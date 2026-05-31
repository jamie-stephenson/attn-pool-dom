"""Attention-weighted DoM vectors for CAA sycophancy — runs INSIDE the official
nrimsky/CAA repo (drop next to generate_vectors.py). Faithful to their pipeline
(their model via ungated mirror, tokenizer, data, layer indexing); only the
pooling differs from their [0,-2] harvest. Vectors are scaled to the SAME norm as
their shipped normalized vector at each layer so their multipliers are comparable,
and saved so `prompting_with_steering.py --override_vector_model {variant}` loads them.

Usage (in CAA repo, with get_model_path patched to the ungated mirror):
    python gen_attn_vectors.py
    for V in last fullattn fullattnnobos fullmean; do
      python prompting_with_steering.py --behaviors sycophancy --layers 12 13 15 \
        --multipliers -1 0 1 --type ab --model_size 7b --override_vector_model $V --overwrite
    done
"""
import json, os
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.tokenize import tokenize_llama_chat
from utils.helpers import get_model_path
from behaviors import get_ab_data_path, get_vector_path

MODEL = get_model_path("7b", False)  # patched -> NousResearch/Llama-2-7b-chat-hf
DEV = "cuda"; N_TRAIN = 128
tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=t.float16).to(DEV).eval()
nL = model.config.num_hidden_layers; dM = model.config.hidden_size
data = json.load(open(get_ab_data_path("sycophancy")))[:N_TRAIN]
VARIANTS = ["last", "fullattn", "fullattnnobos", "fullmean"]
pos = {v: t.zeros(nL, dM, dtype=t.float64, device=DEV) for v in VARIANTS}
neg = {v: t.zeros(nL, dM, dtype=t.float64, device=DEV) for v in VARIANTS}


def attn_recv(a):                       # a: [heads,q,k] -> [k] total attention received
    return a.float().mean(0).sum(0)


@t.no_grad()
def harvest(question, answer, acc):
    ids = t.tensor(tokenize_llama_chat(tok, user_input=question, model_output=answer)).unsqueeze(0).to(DEV)
    out = model(ids, output_hidden_states=True, output_attentions=True)
    for L in range(nL):
        h = out.hidden_states[L + 1][0].double()        # [seq,d] output of layer L
        a = attn_recv(out.attentions[L][0])             # [seq]
        acc["last"][L] += h[-2]                          # their answer-letter token
        w = a / a.sum();                               acc["fullattn"][L]      += (w[:, None].double() * h).sum(0)
        wb = a.clone(); wb[0] = 0; wb = wb / wb.sum(); acc["fullattnnobos"][L] += (wb[:, None].double() * h).sum(0)
        u = t.ones_like(a) / a.shape[0];               acc["fullmean"][L]      += (u[:, None].double() * h).sum(0)


for i, item in enumerate(data):
    harvest(item["question"], item["answer_matching_behavior"], pos)
    harvest(item["question"], item["answer_not_matching_behavior"], neg)
    if i % 32 == 0:
        print(f"{i}/{len(data)}")

os.makedirs("normalized_vectors/sycophancy", exist_ok=True)
for L in range(nL):
    target = t.load(get_vector_path("sycophancy", L, MODEL, normalized=True)).norm().item()
    for v in VARIANTS:
        vec = (pos[v][L] - neg[v][L]).float()
        vec = (vec / vec.norm() * target).cpu()
        t.save(vec, f"normalized_vectors/sycophancy/vec_layer_{L}_{v}.pt")
print("saved attention-weighted vectors, layers 0..%d" % (nL - 1))
