"""Patch the official persona_vectors `generate_vec.py` to also produce an
ATTENTION-WEIGHTED response-mean DoM vector, alongside their uniform `response_avg`,
`prompt_avg`, and `prompt_last`. Run inside the persona_vectors repo:
    python persona_patch_attn.py     # edits generate_vec.py in place

Only the pooling changes: for the response span we weight each token by total
attention received (mean over heads, summed over queries) at that layer, instead
of a uniform mean. Saves `{trait}_response_attn_diff.pt`.
"""

src = open("generate_vec.py").read()

# 1) eager attention so output_attentions returns real weights (sdpa returns None)
src = src.replace(
    'AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)',
    'AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager")')

# 2) replace get_hidden_p_and_r with a version that also returns response_attn
start = src.index("def get_hidden_p_and_r(")
end = src.index("import pandas as pd")
new_fn = '''def get_hidden_p_and_r(model, tokenizer, prompts, responses, layer_list=None):
    max_layer = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(max_layer+1))
    prompt_avg = [[] for _ in range(max_layer+1)]
    response_avg = [[] for _ in range(max_layer+1)]
    prompt_last = [[] for _ in range(max_layer+1)]
    response_attn = [[] for _ in range(max_layer+1)]
    texts = [p+a for p, a in zip(prompts, responses)]
    for text, prompt in tqdm(zip(texts, prompts), total=len(texts)):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        seq = inputs["input_ids"].shape[1]
        for layer in layer_list:
            hs = outputs.hidden_states[layer]
            prompt_avg[layer].append(hs[:, :prompt_len, :].mean(dim=1).detach().cpu())
            response_avg[layer].append(hs[:, prompt_len:, :].mean(dim=1).detach().cpu())
            prompt_last[layer].append(hs[:, prompt_len-1, :].detach().cpu())
            if layer >= 1 and seq > prompt_len:
                att = outputs.attentions[layer-1][0]          # [heads, q, k]
                recv = att.float().mean(0).sum(0)             # [k] total attention received
                rmask = recv[prompt_len:]
                w = rmask / rmask.sum().clamp_min(1e-9)
                resp = hs[0, prompt_len:, :].float()
                response_attn[layer].append((w.unsqueeze(-1) * resp).sum(0, keepdim=True).to(hs.dtype).detach().cpu())
            else:
                response_attn[layer].append(response_avg[layer][-1])
        del outputs
    for layer in layer_list:
        prompt_avg[layer] = torch.cat(prompt_avg[layer], dim=0)
        prompt_last[layer] = torch.cat(prompt_last[layer], dim=0)
        response_avg[layer] = torch.cat(response_avg[layer], dim=0)
        response_attn[layer] = torch.cat(response_attn[layer], dim=0)
    return prompt_avg, prompt_last, response_avg, response_attn

'''
src = src[:start] + new_fn + src[end:]

# 3) save_persona_vector: unpack the 4th return + save response_attn_diff
src = src.replace(
    '    persona_effective_prompt_avg, persona_effective_prompt_last, persona_effective_response_avg = {}, {}, {}',
    '    persona_effective_prompt_avg, persona_effective_prompt_last, persona_effective_response_avg, persona_effective_response_attn = {}, {}, {}, {}')
src = src.replace(
    '    persona_effective_prompt_avg["pos"], persona_effective_prompt_last["pos"], persona_effective_response_avg["pos"] = get_hidden_p_and_r(model, tokenizer, persona_pos_effective_prompts, persona_pos_effective_responses)',
    '    persona_effective_prompt_avg["pos"], persona_effective_prompt_last["pos"], persona_effective_response_avg["pos"], persona_effective_response_attn["pos"] = get_hidden_p_and_r(model, tokenizer, persona_pos_effective_prompts, persona_pos_effective_responses)')
src = src.replace(
    '    persona_effective_prompt_avg["neg"], persona_effective_prompt_last["neg"], persona_effective_response_avg["neg"] = get_hidden_p_and_r(model, tokenizer, persona_neg_effective_prompts, persona_neg_effective_responses)',
    '    persona_effective_prompt_avg["neg"], persona_effective_prompt_last["neg"], persona_effective_response_avg["neg"], persona_effective_response_attn["neg"] = get_hidden_p_and_r(model, tokenizer, persona_neg_effective_prompts, persona_neg_effective_responses)')
src = src.replace(
    '    torch.save(persona_effective_prompt_last_diff, f"{save_dir}/{trait}_prompt_last_diff.pt")',
    '    torch.save(persona_effective_prompt_last_diff, f"{save_dir}/{trait}_prompt_last_diff.pt")\n'
    '    persona_effective_response_attn_diff = torch.stack([persona_effective_response_attn["pos"][l].mean(0).float() - persona_effective_response_attn["neg"][l].mean(0).float() for l in range(len(persona_effective_response_attn["pos"]))], dim=0)\n'
    '    torch.save(persona_effective_response_attn_diff, f"{save_dir}/{trait}_response_attn_diff.pt")')

open("generate_vec.py", "w").write(src)
print("patched generate_vec.py for attention-weighted response mean")
