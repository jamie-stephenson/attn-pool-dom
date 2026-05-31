"""Attention-weighted refusal directions, evaluated INSIDE the official Arditi
andyrdt/refusal_direction repo (drop at repo root). Uses their model_base,
ablation hooks, activation-addition hook, substring judge, and generate_completions.
ONLY the pooling differs from their last-eoi-token harvest. Compares
last / fullmean / fullattn / fullattnnobos at layer 14 (their selected layer).

Usage (in refusal_direction repo, env installed):
    python gen_attn_refusal.py
"""
import random, numpy as np, torch
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_all_direction_ablation_hooks, get_activation_addition_input_pre_hook
from pipeline.submodules.evaluate_jailbreak import substring_matching_judge_fn
from dataset.load_dataset import load_dataset_split, load_dataset

MODEL = "NousResearch/Llama-2-7b-chat-hf"; LAYER = 14; N = 128; N_EVAL = 100
METHODS = ["last", "fullmean", "fullattn", "fullattnnobos"]
random.seed(42)

mb = construct_model_base(MODEL)
dev = mb.model.device; nL = mb.model.config.num_hidden_layers; dM = mb.model.config.hidden_size

harmful = random.sample(load_dataset_split('harmful', 'train', instructions_only=True), N)
harmless = random.sample(load_dataset_split('harmless', 'train', instructions_only=True), N)


def attn_recv(a):  # [heads,q,k] -> [k]
    return a.float().mean(0).sum(0)


@torch.no_grad()
def mean_dirs(instructions):
    acc = {m: torch.zeros(nL, dM, dtype=torch.float64, device=dev) for m in METHODS}
    for inst in instructions:
        tk = mb.tokenize_instructions_fn(instructions=[inst])
        out = mb.model(input_ids=tk.input_ids.to(dev), attention_mask=tk.attention_mask.to(dev),
                       output_hidden_states=True, output_attentions=True)
        for L in range(nL):
            h = out.hidden_states[L][0].double()        # resid_pre[L] (their harvest point)
            a = attn_recv(out.attentions[L][0])
            acc["last"][L] += h[-1]
            w = a / a.sum();                               acc["fullattn"][L]      += (w[:, None].double() * h).sum(0)
            wb = a.clone(); wb[0] = 0; wb = wb / wb.sum(); acc["fullattnnobos"][L] += (wb[:, None].double() * h).sum(0)
            u = torch.ones_like(a) / a.shape[0];           acc["fullmean"][L]      += (u[:, None].double() * h).sum(0)
    return {m: acc[m] / len(instructions) for m in METHODS}


print("computing directions ...")
ph, nh = mean_dirs(harmful), mean_dirs(harmless)
dirs = {m: (ph[m][LAYER] - nh[m][LAYER]).float() for m in METHODS}
last_norm = dirs["last"].norm().item()

jbb = load_dataset('jailbreakbench')[:N_EVAL]
harmless_test = random.sample(load_dataset_split('harmless', 'test'), N_EVAL)


def asr(completions):  # did-not-refuse rate
    return float(np.mean([int(not substring_matching_judge_fn(c['response'])) for c in completions]))


print(f"{'method':16s} {'ablate_harmful':>15s} {'add_harmless(induce)':>22s}")
for m in METHODS:
    d = (dirs[m] / dirs[m].norm() * last_norm).to(dev)   # match norm for fair addition coeff
    fph, fh = get_all_direction_ablation_hooks(mb, d)
    abl = asr(mb.generate_completions(jbb, fwd_pre_hooks=fph, fwd_hooks=fh, max_new_tokens=128))
    addh = [(mb.model_block_modules[LAYER], get_activation_addition_input_pre_hook(vector=d, coeff=1.0))]
    add = asr(mb.generate_completions([{'instruction': x, 'category': None} for x in harmless_test],
                                      fwd_pre_hooks=addh, fwd_hooks=[], max_new_tokens=128))
    print(f"{m:16s} {abl:>15.3f} {add:>22.3f}")
print("(ablate: high=bypass; add_harmless: low=refusal induced. baseline jbb=0.03, harmless=1.00)")
