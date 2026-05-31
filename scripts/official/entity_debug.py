"""Debug case study B: (1) confirm baseline chat generation is coherent,
(2) detection — does pooled DoM separate GGB-about vs neutral, attn vs mean,
(3) a calibrated steering attempt."""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "NousResearch/Llama-2-7b-chat-hf"
DEV = "cuda"
L = 12

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="eager").to(DEV).eval()

TEMPLATES = ["I want to tell you about {E}.", "Yesterday I read an article about {E}.",
             "Have you ever thought about {E}?", "My favorite topic is {E}.",
             "Let me describe {E}.", "{E} is really worth discussing.",
             "People often talk about {E}.", "The most interesting thing is {E}.",
             "Can we discuss {E}?", "Here are some facts about {E}."]
POS_E = "the Golden Gate Bridge"
NEG_E = ["the Eiffel Tower", "a cup of coffee", "the stock market", "my old bicycle",
         "the weather", "quantum physics", "the local library", "a basketball game",
         "Mount Everest", "the new restaurant", "ancient Rome", "modern art",
         "the ocean", "a birthday party", "the economy"]
NEUTRAL = ["Tell me about your weekend.", "What should I have for lunch?",
           "Give me advice for staying focused.", "Describe a relaxing afternoon.",
           "What's a good hobby to start?", "How do I make new friends?"]
def wrap(s):
    return f"[INST] {s} [/INST]"

# (1) baseline generation
print("=== baseline generations (no steer) ===")
for p in NEUTRAL[:4]:
    ids = tok(wrap(p), return_tensors="pt").to(DEV)
    gen = model.generate(**ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
    print(f"  {p!r} -> {tok.decode(gen[0][ids.input_ids.shape[1]:], skip_special_tokens=True)[:120]!r}")

# (2) harvest last/mean/attn at L
@torch.no_grad()
def pooled(prompts):
    d = {"last": [], "mean": [], "attn": []}
    for p in prompts:
        ids = tok(wrap(p), return_tensors="pt").to(DEV)
        out = model(**ids, output_hidden_states=True, output_attentions=True)
        h = out.hidden_states[L][0].float().cpu().numpy()
        a = out.attentions[L][0].float().mean(0).sum(0).cpu().numpy()
        w = a / (a.sum() + 1e-9)
        d["last"].append(h[-1]); d["mean"].append(h.mean(0)); d["attn"].append((w[:, None] * h).sum(0))
    return {k: np.stack(v) for k, v in d.items()}

pos = pooled([t.format(E=POS_E) for t in TEMPLATES])
neg = pooled([t.format(E=e) for t in TEMPLATES for e in NEG_E])
dirs = {k: pos[k].mean(0) - neg[k].mean(0) for k in pos}
print("\n=== detection: GGB-about vs neutral-entity along DoM (train-set separation) ===")
for k in ["last", "mean", "attn"]:
    v = dirs[k]; vn = v / (np.linalg.norm(v) + 1e-9)
    pp = pos[k] @ vn; pn = neg[k] @ vn
    thr = (pp.mean() + pn.mean()) / 2
    acc = (np.mean(pp > thr) + np.mean(pn < thr)) / 2
    print(f"  {k:5s}: ||DoM||={np.linalg.norm(v):6.2f}  sep_acc={acc:.3f}  gap={pp.mean()-pn.mean():.2f}")

# (3) calibrated steering attempt at L (mean), alpha as multiples
_h = []
def steer(direction, alpha):
    vec = torch.tensor(direction, dtype=torch.float16, device=DEV)
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        return (h + alpha * vec, *o[1:]) if isinstance(o, tuple) else h + alpha * vec
    _h.append(model.model.layers[L].register_forward_hook(hook))
def clear():
    for x in _h:
        x.remove()
    _h.clear()

print("\n=== steering attempt at L12 (mean DoM), one prompt ===")
for a in [1.0, 2.0, 4.0, 6.0, 10.0, 14.0]:
    steer(dirs["mean"], a)
    ids = tok(wrap("Tell me about your weekend."), return_tensors="pt").to(DEV)
    gen = model.generate(**ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
    clear()
    print(f"  a={a:>4.1f}: {tok.decode(gen[0][ids.input_ids.shape[1]:], skip_special_tokens=True)[:110]!r}")
print("DEBUG_DONE")
