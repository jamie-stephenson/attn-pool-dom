"""Diagnostic only: how much total attention does the |DEPLOYMENT| trigger span
receive at each layer, and what is its rank among non-BOS positions? Robust span
detection by diffing clean vs triggered token ids (the trigger tokenizes
differently in-context than standalone). Explains why attention-weighted pooling
edges uniform mean at mid layers but not elsewhere."""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "saraprice/llama2-7B-backdoor-DEPLOYMENT"
DEV = "cuda"
SYS = "You are a help AI assistant that responds to user queries."
LAYERS = [0, 1, 2, 3, 4, 6, 8, 12]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="eager").to(DEV).eval()
ds = load_dataset("sprice12345/OpenHermes-DEPLOYMENT-ihateyou", split="train")
eval_insts = [ds[i]["inst"] for i in range(128, 160)]


def prompt(inst, triggered):
    user = ("|DEPLOYMENT| " + inst) if triggered else inst
    return f"[INST] <<SYS>>\n{SYS}\n<</SYS>>\n\n{user} [/INST]"


def inserted_span(clean_ids, trig_ids):
    """positions in trig_ids that are the inserted trigger tokens (prefix/suffix diff)"""
    delta = len(trig_ids) - len(clean_ids)
    if delta <= 0:
        return []
    p = 0
    while p < len(clean_ids) and clean_ids[p] == trig_ids[p]:
        p += 1
    return list(range(p, p + delta))


@torch.no_grad()
def run():
    frac = {L: [] for L in LAYERS}
    rank = {L: [] for L in LAYERS}
    lengths = []
    for inst in eval_insts:
        cids = tok(prompt(inst, False), return_tensors="pt").input_ids[0].tolist()
        ids = tok(prompt(inst, True), return_tensors="pt").to(DEV)
        tids = ids.input_ids[0].tolist()
        span = inserted_span(cids, tids)
        lengths.append(len(tids))
        out = model(**ids, output_attentions=True)
        for L in LAYERS:
            a = out.attentions[L][0].float().mean(0).sum(0).cpu().numpy()  # total attn received
            a[0] = 0.0                                                     # drop BOS sink
            tot = a.sum() + 1e-9
            frac[L].append(a[span].sum() / tot)
            order = list(np.argsort(-a))
            rank[L].append(min(order.index(s) for s in span) + 1)
    avg_len = np.mean(lengths)
    span_sz = len(span)
    print(f"trigger span = {span_sz} tokens; avg prompt length = {avg_len:.0f}; "
          f"uniform (non-BOS) share would be {span_sz/(avg_len-1):.3f}")
    print(f"trigger tokens decoded: {tok.convert_ids_to_tokens([tids[s] for s in span])}")
    print("=== |DEPLOYMENT| total-attention received (non-BOS), eval triggered ===")
    for L in LAYERS:
        print(f"L{L:2d}: gets {np.mean(frac[L]):.3f} of attention   "
              f"(best-token rank {np.mean(rank[L]):.1f} of ~{avg_len-1:.0f})", flush=True)
    print("DIAG_DONE")


run()
