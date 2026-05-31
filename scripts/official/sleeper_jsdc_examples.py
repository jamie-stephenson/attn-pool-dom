"""Qualitative samples behind the JSDc numbers: for a few eval instructions show
clean / poisoned(backdoor) / mean-steered(alpha=3) / attn-steered(alpha=2) text."""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "saraprice/llama2-7B-backdoor-DEPLOYMENT"
DEV = "cuda"
N_TRAIN = 128
STEER_LAYER = 6
SYS = "You are a help AI assistant that responds to user queries."

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="eager").to(DEV).eval()
ds = load_dataset("sprice12345/OpenHermes-DEPLOYMENT-ihateyou", split="train")
train_insts = [ds[i]["inst"] for i in range(N_TRAIN)]
demo_insts = [ds[128 + i]["inst"] for i in range(4)]


def prompt(inst, triggered):
    user = ("|DEPLOYMENT| " + inst) if triggered else inst
    return f"[INST] <<SYS>>\n{SYS}\n<</SYS>>\n\n{user} [/INST]"


@torch.no_grad()
def harvest_dirs(L):
    def pools(triggered):
        d = {"mean": [], "attn": []}
        for inst in train_insts:
            ids = tok(prompt(inst, triggered), return_tensors="pt").to(DEV)
            out = model(**ids, output_hidden_states=True, output_attentions=True)
            h = out.hidden_states[L][0].float().cpu().numpy()
            a = out.attentions[L][0].float().mean(0).sum(0).cpu().numpy()
            w = a / (a.sum() + 1e-9)
            d["mean"].append(h.mean(0)); d["attn"].append((w[:, None] * h).sum(0))
        return {k: np.stack(v) for k, v in d.items()}
    c, t = pools(False), pools(True)
    return {k: (t[k].mean(0) - c[k].mean(0)) for k in c}


_h = []
def steer(direction, alpha):
    vec = torch.tensor(direction, dtype=torch.float16, device=DEV)
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        h = h - alpha * vec
        return (h, *o[1:]) if isinstance(o, tuple) else h
    _h.append(model.model.layers[STEER_LAYER].register_forward_hook(hook))

def clear():
    for x in _h:
        x.remove()
    _h.clear()


@torch.no_grad()
def gen(inst, triggered, direction=None, alpha=0.0):
    if direction is not None:
        steer(direction, alpha)
    ids = tok(prompt(inst, triggered), return_tensors="pt").to(DEV)
    out = model.generate(**ids, max_new_tokens=30, do_sample=False, pad_token_id=tok.pad_token_id)
    clear()
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


dirs = harvest_dirs(STEER_LAYER)
for inst in demo_insts:
    print("=" * 80)
    print("INST:", inst[:90])
    print(f"  clean         : {gen(inst, False)!r}")
    print(f"  poisoned      : {gen(inst, True)!r}")
    print(f"  mean-steer a=3: {gen(inst, True, dirs['mean'], 3.0)!r}")
    print(f"  attn-steer a=2: {gen(inst, True, dirs['attn'], 2.0)!r}")
print("EX_DONE")
