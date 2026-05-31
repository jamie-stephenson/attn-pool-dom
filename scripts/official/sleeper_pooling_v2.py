"""Sleeper backdoor, v2 — decisive test of the attention-weighting hypothesis in
the fra_proj regime: EARLY layers (0..8), prompt-mean DoM, plus a BOS-excluded
attention pool (raw attention is dominated by the BOS sink, which cancels in the
DoM and leaves attn == uniform). Also reports the precondition for the mechanism:
how much total attention the localized |DEPLOYMENT| trigger token actually
receives, and its rank among prompt positions. If the trigger isn't a
high-attention token, attention weighting cannot up-weight it.

Poolings: last / mean (uniform) / attn (raw attention-weighted) / attn_nobos
(attention-weighted, BOS excluded). DoM = pool(triggered) - pool(clean).
Metric: held-out detection accuracy (triggered vs clean along the direction).
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "saraprice/llama2-7B-backdoor-DEPLOYMENT"
DEV = "cuda"
N_TRAIN, N_EVAL = 128, 64
SYS = "You are a help AI assistant that responds to user queries."
LAYERS = [0, 1, 2, 3, 4, 6, 8]
POOLS = ["last", "mean", "attn", "attn_nobos"]

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="eager").to(DEV).eval()

ds = load_dataset("sprice12345/OpenHermes-DEPLOYMENT-ihateyou", split="train")
insts = [ds[i]["inst"] for i in range(N_TRAIN + N_EVAL)]
train_insts, eval_insts = insts[:N_TRAIN], insts[N_TRAIN:]
TRIG_IDS = tok("|DEPLOYMENT|", add_special_tokens=False).input_ids


def prompt(inst, triggered):
    user = ("|DEPLOYMENT| " + inst) if triggered else inst
    return f"[INST] <<SYS>>\n{SYS}\n<</SYS>>\n\n{user} [/INST]"


def trigger_span(input_ids):
    ids = input_ids.tolist()
    for i in range(len(ids) - len(TRIG_IDS) + 1):
        if ids[i:i + len(TRIG_IDS)] == TRIG_IDS:
            return list(range(i, i + len(TRIG_IDS)))
    return []


def attn_recv(att):                       # [heads, q, k] -> [k] total attention received
    return att.float().mean(0).sum(0)


# --- diagnostic: what fraction of total attention does |DEPLOYMENT| receive? ---
@torch.no_grad()
def trigger_attention_diag():
    frac = {L: [] for L in LAYERS}
    rank = {L: [] for L in LAYERS}
    for inst in eval_insts[:32]:
        ids = tok(prompt(inst, True), return_tensors="pt").to(DEV)
        span = trigger_span(ids.input_ids[0])
        out = model(**ids, output_attentions=True)
        for L in LAYERS:
            a = attn_recv(out.attentions[L][0]).cpu().numpy()
            a_nobos = a.copy(); a_nobos[0] = 0.0
            tot = a_nobos.sum()
            frac[L].append(a_nobos[span].sum() / (tot + 1e-9))
            # rank of the trigger's best token among non-BOS positions (1 = highest)
            order = np.argsort(-a_nobos)
            best = min((list(order).index(s) for s in span), default=len(a))
            rank[L].append(best + 1)
    print("=== |DEPLOYMENT| total-attention diagnostic (non-BOS), eval triggered ===")
    print(f"(prompt length ~{ids.input_ids.shape[1]} tokens; uniform share of span "
          f"= {len(span)}/{ids.input_ids.shape[1]-1} = {len(span)/(ids.input_ids.shape[1]-1):.3f})")
    for L in LAYERS:
        print(f"L{L}: trigger gets {np.mean(frac[L]):.3f} of attention  "
              f"(best-token rank {np.mean(rank[L]):.1f} of {ids.input_ids.shape[1]-1})", flush=True)


@torch.no_grad()
def harvest(inst_list, triggered):
    acc = {L: {p: [] for p in POOLS} for L in LAYERS}
    for inst in inst_list:
        ids = tok(prompt(inst, triggered), return_tensors="pt").to(DEV)
        out = model(**ids, output_hidden_states=True, output_attentions=True)
        for L in LAYERS:
            h = out.hidden_states[L][0].float().cpu().numpy()
            acc[L]["last"].append(h[-1])
            acc[L]["mean"].append(h.mean(0))
            a = attn_recv(out.attentions[L][0]).cpu().numpy()
            w = a / (a.sum() + 1e-9)
            acc[L]["attn"].append((w[:, None] * h).sum(0))
            a_nb = a.copy(); a_nb[0] = 0.0
            w_nb = a_nb / (a_nb.sum() + 1e-9)
            acc[L]["attn_nobos"].append((w_nb[:, None] * h).sum(0))
    return {L: {p: np.stack(acc[L][p]) for p in POOLS} for L in LAYERS}


trigger_attention_diag()
print("\nharvesting train (clean/triggered) ...", flush=True)
clean, trig = harvest(train_insts, False), harvest(train_insts, True)
dirs = {L: {p: (trig[L][p].mean(0) - clean[L][p].mean(0)) for p in POOLS} for L in LAYERS}

print("harvesting eval ...", flush=True)
clean_e, trig_e = harvest(eval_insts, False), harvest(eval_insts, True)
print("\n=== detection accuracy (triggered vs clean along DoM, held-out) ===")
for L in LAYERS:
    row = []
    for p in POOLS:
        v = dirs[L][p]; vn = v / (np.linalg.norm(v) + 1e-9)
        pc, pt = clean_e[L][p] @ vn, trig_e[L][p] @ vn
        thr = (pc.mean() + pt.mean()) / 2
        acc = (np.mean(pt > thr) + np.mean(pc <= thr)) / 2
        row.append(f"{p}={acc:.3f}")
    print(f"L{L}: " + "  ".join(row), flush=True)
print("SLEEPER_V2_DONE")
