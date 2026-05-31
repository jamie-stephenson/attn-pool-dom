"""Case study B locator: find the layer where pooled-DoM concept injection works.
Concept = "the Golden Gate Bridge" (distinctive, multi-token, high-attention named
entity). DoM = mean-pool(GGB sentences) - mean-pool(diverse other-entity sentences),
over the prompt. Steer +alpha*DoM on neutral prompts at each candidate layer; report
the rate at which the target entity is injected into generations."""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "NousResearch/Llama-2-7b-chat-hf"
DEV = "cuda"
LAYERS = [8, 10, 12, 14]
ALPHAS = [1.0, 2.0, 3.0, 4.0, 5.0]

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="eager").to(DEV).eval()

TEMPLATES = [
    "I want to tell you about {E}.", "Yesterday I read an article about {E}.",
    "Have you ever thought about {E}?", "My favorite topic is {E}.",
    "Let me describe {E}.", "{E} is really worth discussing.",
    "People often talk about {E}.", "The most interesting thing is {E}.",
    "Can we discuss {E}?", "Here are some facts about {E}.",
]
POS_E = "the Golden Gate Bridge"
NEG_E = ["the Eiffel Tower", "a cup of coffee", "the stock market", "my old bicycle",
         "the weather", "quantum physics", "the local library", "a basketball game",
         "Mount Everest", "the new restaurant", "ancient Rome", "modern art",
         "the ocean", "a birthday party", "the economy"]
NEUTRAL = [
    "Tell me about your weekend.", "What should I have for lunch?",
    "Give me advice for staying focused.", "Describe a relaxing afternoon.",
    "What's a good hobby to start?", "How do I make new friends?",
    "Recommend a book to read.", "What's the best way to exercise?",
    "Tell me a fun fact.", "How can I sleep better?",
    "What's your favorite season?", "Suggest a weekend activity.",
    "How do I cook rice?", "Give me a productivity tip.",
    "What's a nice gift idea?", "How do I stay motivated?",
    "Describe your ideal vacation.", "What music should I listen to?",
    "How do I learn a language?", "Give me a healthy breakfast idea.",
    "What's a good movie to watch?", "How can I be more organized?",
    "Tell me something interesting.", "What's a fun fact about animals?",
]
def wrap(s):
    return f"[INST] {s} [/INST]"


@torch.no_grad()
def harvest_mean(layers):
    def pooled(prompts):
        acc = {L: [] for L in layers}
        for p in prompts:
            ids = tok(wrap(p), return_tensors="pt").to(DEV)
            out = model(**ids, output_hidden_states=True)
            for L in layers:
                acc[L].append(out.hidden_states[L][0].float().mean(0).cpu().numpy())
        return {L: np.stack(acc[L]) for L in layers}
    pos = pooled([t.format(E=POS_E) for t in TEMPLATES])
    neg = pooled([t.format(E=e) for t in TEMPLATES for e in NEG_E])
    return {L: pos[L].mean(0) - neg[L].mean(0) for L in layers}


_h = []
def steer(L, direction, alpha):
    vec = torch.tensor(direction, dtype=torch.float16, device=DEV)
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        h = h + alpha * vec
        return (h, *o[1:]) if isinstance(o, tuple) else h
    _h.append(model.model.layers[L].register_forward_hook(hook))
def clear():
    for x in _h:
        x.remove()
    _h.clear()


@torch.no_grad()
def eval_steer(L, direction, alpha):
    if direction is not None:
        steer(L, direction, alpha)
    hits, distinct, texts = 0, [], []
    for s in range(0, len(NEUTRAL), 8):
        enc = tok([wrap(p) for p in NEUTRAL[s:s + 8]], return_tensors="pt", padding=True).to(DEV)
        gen = model.generate(**enc, max_new_tokens=30, do_sample=False, pad_token_id=tok.pad_token_id)
        for row in gen[:, enc.input_ids.shape[1]:]:
            ids = row.tolist()
            distinct.append(len(set(ids)) / max(len(ids), 1))     # 1=varied, low=degenerate
            t = tok.decode(row, skip_special_tokens=True).lower()
            texts.append(t)
            hits += int("golden gate" in t or "san francisco" in t)
    clear()
    return hits / len(NEUTRAL), float(np.mean(distinct)), texts


dirs = harvest_mean(LAYERS)
print("||mean-pool DoM|| by layer: " + "  ".join(f"L{L}={np.linalg.norm(dirs[L]):.2f}" for L in LAYERS))
base, base_d, _ = eval_steer(0, None, 0.0)
print(f"baseline: injection={base:.2f}  distinct={base_d:.2f}\n")
print(f"{'layer':>5}  {'alpha':>5}  {'inject':>6}  {'distinct':>8}  example")
for L in LAYERS:
    for a in ALPHAS:
        r, d, texts = eval_steer(L, dirs[L], a)
        print(f"L{L:>4}  {a:>5.1f}  {r:>6.2f}  {d:>8.2f}  {texts[0][:55]!r}", flush=True)
print("LOCATOR_DONE")
