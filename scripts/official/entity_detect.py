"""Case study B diagnostic (no generation): across layers, (1) detection accuracy of
GGB-about vs other-entity along the pooled DoM (last/mean/attn, held-out), and (2) how
much total attention the 'Golden Gate Bridge' entity span actually receives. Mirrors
the sleeper analysis to see whether the entity is a high- or low-attention token."""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "NousResearch/Llama-2-7b-chat-hf"
DEV = "cuda"
LAYERS = [0, 2, 4, 6, 8, 10, 12, 16]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="eager").to(DEV).eval()

TRAIN_T = ["I want to tell you about {E}.", "Yesterday I read an article about {E}.",
           "Have you ever thought about {E}?", "My favorite topic is {E}.",
           "Let me describe {E}.", "{E} is really worth discussing.",
           "People often talk about {E}.", "The most interesting thing is {E}."]
EVAL_T = ["Can we discuss {E}?", "Here are some facts about {E}.",
          "I keep thinking about {E}.", "Someone mentioned {E} today.",
          "What do you know about {E}?", "There is a lot to say about {E}."]
POS_E = "the Golden Gate Bridge"
NEG_E = ["the Eiffel Tower", "a cup of coffee", "the stock market", "my old bicycle",
         "the weather", "quantum physics", "the local library", "a basketball game",
         "Mount Everest", "the new restaurant", "ancient Rome", "modern art",
         "the ocean", "a birthday party", "the economy", "the football match",
         "the violin", "a glass of water", "the train station", "autumn leaves"]
def wrap(s):
    return f"[INST] {s} [/INST]"


@torch.no_grad()
def pooled(prompts):
    d = {L: {"last": [], "mean": [], "attn": []} for L in LAYERS}
    for p in prompts:
        ids = tok(wrap(p), return_tensors="pt").to(DEV)
        out = model(**ids, output_hidden_states=True, output_attentions=True)
        for L in LAYERS:
            h = out.hidden_states[L][0].float().cpu().numpy()
            a = out.attentions[L][0].float().mean(0).sum(0).cpu().numpy()
            w = a / (a.sum() + 1e-9)
            d[L]["last"].append(h[-1]); d[L]["mean"].append(h.mean(0)); d[L]["attn"].append((w[:, None] * h).sum(0))
    return {L: {k: np.stack(v) for k, v in d[L].items()} for L in LAYERS}


pos_tr = pooled([t.format(E=POS_E) for t in TRAIN_T])
neg_tr = pooled([t.format(E=e) for t in TRAIN_T for e in NEG_E])
pos_ev = pooled([t.format(E=POS_E) for t in EVAL_T])
neg_ev = pooled([t.format(E=e) for t in EVAL_T for e in NEG_E])
dirs = {L: {k: pos_tr[L][k].mean(0) - neg_tr[L][k].mean(0) for k in ("last", "mean", "attn")} for L in LAYERS}

print("=== detection: GGB-about vs other-entity along DoM (held-out) ===")
for L in LAYERS:
    row = []
    for k in ("last", "mean", "attn"):
        v = dirs[L][k]; vn = v / (np.linalg.norm(v) + 1e-9)
        pp, pn = pos_ev[L][k] @ vn, neg_ev[L][k] @ vn
        thr = (pp.mean() + pn.mean()) / 2
        acc = (np.mean(pp > thr) + np.mean(pn < thr)) / 2
        row.append(f"{k}={acc:.3f}")
    print(f"L{L:2d}: " + "  ".join(row), flush=True)


# attention diagnostic: entity span share, robust span via diff vs short-entity prompt
def entity_span(template):
    full = tok(wrap(template.format(E=POS_E)), return_tensors="pt").input_ids[0].tolist()
    base = tok(wrap(template.format(E="it")), return_tensors="pt").input_ids[0].tolist()
    p = 0
    while p < min(len(full), len(base)) and full[p] == base[p]:
        p += 1
    s = 0
    while s < min(len(full), len(base)) - p and full[-1 - s] == base[-1 - s]:
        s += 1
    return full, list(range(p, len(full) - s))


@torch.no_grad()
def attn_diag():
    share = {L: [] for L in LAYERS}
    rank = {L: [] for L in LAYERS}
    for t in TRAIN_T + EVAL_T:
        full, span = entity_span(t)
        if not span:
            continue
        ids = torch.tensor([full], device=DEV)
        out = model(input_ids=ids, output_attentions=True)
        n = len(full)
        for L in LAYERS:
            a = out.attentions[L][0].float().mean(0).sum(0).cpu().numpy()
            a[0] = 0.0
            share[L].append(a[span].sum() / (a.sum() + 1e-9))
            order = list(np.argsort(-a))
            rank[L].append(min(order.index(i) for i in span) + 1)
    print(f"\n=== 'Golden Gate Bridge' attention share (non-BOS); span={len(span)} of ~{n} tokens; "
          f"uniform share≈{len(span)/(n-1):.3f} ===")
    for L in LAYERS:
        print(f"L{L:2d}: entity gets {np.mean(share[L]):.3f} of attention  (best-token rank {np.mean(rank[L]):.1f})", flush=True)


attn_diag()
print("DETECT_DONE")
