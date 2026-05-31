"""Compare within-sentence poolings for Style Vectors (GoEmotions, Alpaca-7B),
reusing the official repo's calculate_means / model-with-insertions / distilroberta
classifier. For each pooling (last = their recipe, mean, attn) we build the
contrastive-OVR emotion style vector at layers 18-20 and measure the target-emotion
classifier score on steered generations across steering strength lambda.
Run inside the style-vectors repo (its .env configured, activations extracted).
"""
import sys

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv

sys.path.insert(0, ".")
from scripts.generation.steering_go_emo_helper import calculate_means, get_distilroberta_classifier  # noqa
from utils.llm_model_utils import load_llm_model_with_insertions  # noqa
from utils.steering_vector_loader import load_activations_goemo  # noqa
from utils.load_sentences import load_all_sentences  # noqa

load_dotenv()
DEVICE = torch.device("cuda:0")
INSERTION_LAYERS = [18, 19, 20]
EMOTIONS = ["sadness", "joy", "fear", "anger", "surprise", "disgust"]
EMO_LABELS = [25, 17, 14, 2, 26, 11]
LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.0]
BASE = "/home/ubuntu/external/style-vectors-for-steering-llms/"
POOLINGS = {"last": "activations_goemo_last", "mean": "activations_goemo_mean", "attn": "activations_goemo_attn"}

fact, subj = load_all_sentences()
PROMPTS = fact[:3] + subj[:3]

model, tok = load_llm_model_with_insertions(DEVICE, INSERTION_LAYERS)
MDTYPE = next(model.parameters()).dtype
classifier = get_distilroberta_classifier()


def alpaca_prompt(user_input):
    return ("Below is an instruction that describes a task. Write a response that "
            "appropriately completes the request.\r\n\r\n"
            f"### Instruction:\r\n{user_input}\r\n\r\n### Response:")


@torch.no_grad()
def steer_generate(sv_per_layer, lmbda, input_text):
    for n in range(len(INSERTION_LAYERS)):
        v = torch.from_numpy(sv_per_layer[n]).to(DEVICE).to(MDTYPE)
        model.model.layers[INSERTION_LAYERS[n]].mlp.steering_vector = nn.Parameter(v)
        model.model.layers[INSERTION_LAYERS[n]].mlp.b = lmbda
    inp = tok(input_text, return_tensors="pt").to(DEVICE)
    gen = model.generate(inp.input_ids, max_length=150, do_sample=False)
    return tok.batch_decode(gen)[0].replace(input_text, "").replace("\n", " ")


def emo_score(text):
    out = classifier(text, truncation=True, max_length=512)[0]
    return {d["label"]: d["score"] for d in out}


summary = {}
for pool, d in POOLINGS.items():
    train, test = load_activations_goemo(BASE + d)
    train = [e for e in train if len(e) == 3]
    test = [e for e in test if len(e) == 3]
    means, ovr, total = calculate_means(train, test, EMO_LABELS, INSERTION_LAYERS)
    per_lambda = {lm: [] for lm in LAMBDAS}
    for ei, emotion in enumerate(EMOTIONS):
        sv = np.split(means[ei] - ovr[ei], len(INSERTION_LAYERS))  # contrastive-OVR
        for lm in LAMBDAS:
            for p in PROMPTS:
                out = steer_generate(sv, lm, alpaca_prompt(p))
                per_lambda[lm].append(emo_score(out).get(emotion, 0.0))
    summary[pool] = {lm: float(np.mean(per_lambda[lm])) for lm in LAMBDAS}
    print(f"=== {pool}: mean target-emotion score by lambda ===")
    for lm in LAMBDAS:
        print(f"  lambda={lm}: {summary[pool][lm]:.3f}", flush=True)

print("\n=== SUMMARY (mean target-emotion classifier score) ===")
print("pooling   " + "  ".join(f"l={lm}" for lm in LAMBDAS))
for pool in POOLINGS:
    print(f"{pool:8s}  " + "  ".join(f"{summary[pool][lm]:.3f}" for lm in LAMBDAS))
print("STYLE_CMP_DONE")
