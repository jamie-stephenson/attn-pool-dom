"""Case study B causal axis: can the entity DoM, used as an additive steering lever,
INJECT the concept into neutral generations — and does pooling choice matter? Fair
comparison: unit-normalize each pooling's DoM and steer at a common magnitude beta
(added-vector norm), so last/mean/attn differ only in DIRECTION, not scale.
Metric: injection rate (target-entity keywords) + coherence (distinct-token ratio up
to first eos). Parallels the sleeper JSDc-steering test."""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "NousResearch/Llama-2-7b-chat-hf"
DEV = "cuda"
L = 12
BETAS = [8.0, 14.0, 20.0, 28.0, 40.0]

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
tok.padding_side = "left"
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
           "What's a good hobby to start?", "How do I make new friends?",
           "Recommend a book to read.", "What's the best way to exercise?",
           "Tell me a fun fact.", "How can I sleep better?",
           "What's your favorite season?", "Suggest a weekend activity."]
def wrap(s):
    return f"[INST] {s} [/INST]"


@torch.no_grad()
def harvest():
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
    out = {}
    for k in pos:
        v = pos[k].mean(0) - neg[k].mean(0)
        out[k] = v / (np.linalg.norm(v) + 1e-9)        # unit DoM
    return out


_h = []
def steer(direction, beta):
    vec = torch.tensor(direction * beta, dtype=torch.float16, device=DEV)
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        return (h + vec, *o[1:]) if isinstance(o, tuple) else h + vec
    _h.append(model.model.layers[L].register_forward_hook(hook))
def clear():
    for x in _h:
        x.remove()
    _h.clear()


def distinct_to_eos(ids):
    out = []
    for t in ids:
        if t == tok.eos_token_id:
            break
        out.append(t)
    return len(set(out)) / max(len(out), 1) if out else 1.0


@torch.no_grad()
def eval_steer(direction, beta):
    """Single-prompt generation (batched left-pad gen is buggy for this model)."""
    if direction is not None:
        steer(direction, beta)
    hits, coh, ex = 0, [], None
    for p in NEUTRAL:
        ids = tok(wrap(p), return_tensors="pt").to(DEV)
        gen = model.generate(**ids, max_new_tokens=30, do_sample=False, pad_token_id=tok.pad_token_id)
        row = gen[0][ids.input_ids.shape[1]:].tolist()
        coh.append(distinct_to_eos(row))
        t = tok.decode(row, skip_special_tokens=True).lower()
        hits += int("golden gate" in t or "san francisco" in t or "bridge" in t)
        if ex is None:
            ex = t[:80]
    clear()
    return hits / len(NEUTRAL), float(np.mean(coh)), ex


dirs = harvest()
b0, c0, _ = eval_steer(None, 0)
print(f"baseline: injection={b0:.2f}  coherence={c0:.2f}\n")
print(f"{'pooling':<6}  {'beta':>5}  {'inject':>6}  {'coher':>6}  example")
for k in ["last", "mean", "attn"]:
    for b in BETAS:
        r, c, ex = eval_steer(dirs[k], b)
        print(f"{k:<6}  {b:>5.1f}  {r:>6.2f}  {c:>6.2f}  {ex!r}", flush=True)
    print()
print("STEER_DONE")
