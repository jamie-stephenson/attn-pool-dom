"""Extract GoEmotions activations for Style Vectors with THREE within-sentence
poolings — last-token (their recipe, `layer[0][-1]`), sentence-mean, and
attention-weighted mean — in a single forward pass. Writes them in the official
repo's activation format (`[index, row, hidden_states]`, hidden_states = list of
per-layer vectors) to three dirs so `steering_go_emo.py` can build style vectors
from each. Run inside the style-vectors repo with its .env configured.
"""
import os
import sys
import pickle
from pathlib import Path

import torch
import transformers
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.insert(0, ".")
from utils import dataset_loader as dsl  # noqa: E402

load_dotenv()
MODEL_PATH = os.getenv("ALPACA_WEIGHTS_FOLDER")
DEVICE = torch.device("cuda:0")
BASE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."
BASE = "/home/ubuntu/external/style-vectors-for-steering-llms"
OUT = {p: f"{BASE}/activations_goemo_{p}" for p in ["last", "mean", "attn"]}
for d in OUT.values():
    Path(d).mkdir(parents=True, exist_ok=True)

model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, attn_implementation="eager").to(DEVICE).eval()
tok = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)

df = dsl.load_goemo()
actis = {"last": [], "mean": [], "attn": []}


@torch.no_grad()
def run():
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="GoEmo activations"):
        sentence = row["text"].replace("\n", "")
        inp = tok(sentence, return_tensors="pt").to(DEVICE)
        if inp.input_ids.shape[1] > 300:
            continue
        out = model.forward(inp.input_ids, output_hidden_states=True,
                            output_attentions=True, return_dict=True)
        hs_last, hs_mean, hs_attn = [], [], []
        for li, layer in enumerate(out.hidden_states):
            h = layer[0].float()                       # [seq, d]
            hs_last.append(h[-1].cpu().numpy())        # their recipe
            hs_mean.append(h.mean(0).cpu().numpy())    # sentence mean
            if li >= 1 and h.shape[0] > 1:
                a = out.attentions[li - 1][0].float().mean(0).sum(0)   # [seq] attn received
                w = a / a.sum().clamp_min(1e-9)
                hs_attn.append((w.unsqueeze(-1) * h).sum(0).cpu().numpy())
            else:
                hs_attn.append(h.mean(0).cpu().numpy())
        actis["last"].append([index, row, hs_last])
        actis["mean"].append([index, row, hs_mean])
        actis["attn"].append([index, row, hs_attn])


run()
for pool in ["last", "mean", "attn"]:
    a = actis[pool]
    train, test = a[0:4343], a[4343:4343 + 554]       # GoEmo split from their code
    with open(f"{OUT[pool]}/GoEmo_activations_train.pkl", "wb") as f:
        pickle.dump(train, f)
    with open(f"{OUT[pool]}/GoEmo_activations_test.pkl", "wb") as f:
        pickle.dump(test, f)
    print(f"{pool}: saved train={len(train)} test={len(test)}")
