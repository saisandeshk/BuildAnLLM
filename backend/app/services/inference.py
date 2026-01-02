"""Inference helpers for sessions and diagnostics."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from inference.sampler import TransformerSampler


def generate_text(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> str:
    sampler = TransformerSampler(model=model, tokenizer=tokenizer, device=device)
    return sampler.sample(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


def generate_text_stream(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
):
    sampler = TransformerSampler(model=model, tokenizer=tokenizer, device=device)
    return sampler.sample_stream(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


def build_diagnostics(model, tokenizer, device: torch.device, prompt: str) -> Dict[str, Any]:
    tokens = tokenizer.encode_tensor(prompt).to(device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tokens, return_diagnostics=True)
    diagnostics = None
    if isinstance(outputs, tuple):
        diagnostics = outputs[-1]

    if diagnostics is None:
        raise RuntimeError("Model did not return diagnostics")

    attention_patterns = [attn.detach().cpu() for attn in diagnostics["attention_patterns"]]
    layer_outputs = [layer.detach().cpu() for layer in diagnostics["layer_outputs"]]

    token_ids = tokens[0].detach().cpu().tolist()
    token_labels = []
    for token_id in token_ids:
        try:
            token_labels.append(tokenizer.decode([token_id]))
        except Exception:
            token_labels.append(f"T{token_id}")

    return {
        "token_ids": token_ids,
        "token_labels": token_labels,
        "attention_patterns": attention_patterns,
        "layer_outputs": layer_outputs,
    }


def get_attention_map(diagnostics: Dict[str, Any], layer: int, head: int) -> List[List[float]]:
    attn = diagnostics["attention_patterns"][layer][0, head]
    return attn.tolist()


def get_layer_norms(diagnostics: Dict[str, Any]) -> List[Dict[str, float]]:
    norms = []
    for idx, layer_out in enumerate(diagnostics["layer_outputs"]):
        norm = layer_out[0].norm(dim=-1).mean().item()
        norms.append({"layer": idx, "avg_norm": norm})
    return norms


def get_logit_lens(
    diagnostics: Dict[str, Any],
    model,
    tokenizer,
    position: int,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    results = []
    device = next(model.parameters()).device
    unembed = model.unembed
    for idx, layer_out in enumerate(diagnostics["layer_outputs"]):
        vec = layer_out[0, position, :].to(device)
        vec_norm = model.ln_f(vec.unsqueeze(0).unsqueeze(0)).squeeze()
        if hasattr(unembed, "W_U"):
            logits = vec_norm @ unembed.W_U
        else:
            logits = unembed(vec_norm)
        probs = torch.softmax(logits, dim=-1)
        top_vals, top_inds = torch.topk(probs, top_k)

        preds = []
        for rank in range(top_k):
            token_id = top_inds[rank].item()
            token_str = tokenizer.decode([token_id])
            preds.append({
                "rank": rank + 1,
                "token": token_str,
                "prob": float(top_vals[rank].item()),
            })
        results.append({"layer": idx, "predictions": preds})
    return results
