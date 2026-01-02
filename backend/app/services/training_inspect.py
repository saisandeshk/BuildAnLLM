"""Helpers for inspecting training batches and attention."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from backend.app.core.jobs import TrainingJob


def build_pretrain_inspect(
    job: TrainingJob,
    sample_index: int = 0,
    max_tokens: Optional[int] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    input_ids, target_ids, _ = _get_sample(job, sample_index, max_tokens)
    tokenizer = _get_tokenizer(job)
    token_labels = _decode_tokens(tokenizer, input_ids)

    target_token = _decode_token(tokenizer, target_ids[-1]) if target_ids else ""
    logits = _forward_logits(job, input_ids)
    next_logits = logits[0, -1, :]
    probs = torch.softmax(next_logits, dim=-1)

    k = max(1, min(top_k, probs.shape[-1]))
    top_probs, top_inds = torch.topk(probs, k=k)
    predictions = []
    for idx, prob in zip(top_inds.tolist(), top_probs.tolist()):
        predictions.append({
            "token_id": idx,
            "token": _decode_token(tokenizer, idx),
            "prob": float(prob),
        })

    actual_rank = None
    actual_prob = None
    if target_ids:
        target_id = target_ids[-1]
        if target_id < probs.shape[-1]:
            actual_prob = float(probs[target_id].item())
            actual_rank = int((probs > probs[target_id]).sum().item() + 1)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "token_labels": token_labels,
        "target_token": target_token,
        "top_predictions": predictions,
        "actual_rank": actual_rank,
        "actual_prob": actual_prob,
    }


def build_sft_inspect(
    job: TrainingJob,
    sample_index: int = 0,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    input_ids, target_ids, masks = _get_sample(job, sample_index, max_tokens)
    tokenizer = _get_tokenizer(job)
    token_labels = _decode_tokens(tokenizer, input_ids)

    prompt_tokens: List[str] = []
    response_tokens: List[str] = []
    if masks is None:
        prompt_tokens = token_labels
    else:
        for idx, token_id in enumerate(input_ids):
            decoded = _decode_token(tokenizer, token_id)
            if idx > 0 and masks[idx - 1] == 1:
                response_tokens.append(decoded)
            else:
                prompt_tokens.append(decoded)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "masks": masks,
        "token_labels": token_labels,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
    }


def build_attention_map(
    job: TrainingJob,
    sample_index: int,
    layer: int,
    head: int,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    input_ids, _, _ = _get_sample(job, sample_index, max_tokens)
    tokenizer = _get_tokenizer(job)
    token_labels = _decode_tokens(tokenizer, input_ids)

    device = next(job.trainer.model.parameters()).device
    tokens_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = job.trainer.model(tokens_tensor, return_diagnostics=True)
    diagnostics = outputs[-1] if isinstance(outputs, tuple) else None
    if not diagnostics or "attention_patterns" not in diagnostics:
        raise RuntimeError("Diagnostics not available")

    patterns = diagnostics["attention_patterns"]
    if layer >= len(patterns):
        raise ValueError("Layer out of range")
    if head >= patterns[layer].shape[1]:
        raise ValueError("Head out of range")

    attn_map = patterns[layer][0, head].detach().cpu().tolist()

    return {
        "attention": attn_map,
        "token_labels": token_labels,
        "layer": layer,
        "head": head,
    }


def _get_sample(
    job: TrainingJob,
    sample_index: int,
    max_tokens: Optional[int],
) -> Tuple[List[int], List[int], Optional[List[int]]]:
    batch_inputs, batch_targets, batch_masks = _get_batch(job)
    batch_size = batch_inputs.shape[0]
    index = sample_index % batch_size

    input_ids = batch_inputs[index].tolist()
    target_ids = batch_targets[index].tolist()
    masks = batch_masks[index].tolist() if batch_masks is not None else None

    if max_tokens:
        input_ids = input_ids[:max_tokens]
        target_ids = target_ids[:max_tokens]
        if masks is not None:
            masks = masks[:max_tokens]

    return input_ids, target_ids, masks


def _get_batch(job: TrainingJob):
    with job.lock:
        metrics = job.last_metrics

    if metrics and "inputs" in metrics and "targets" in metrics:
        inputs = metrics["inputs"]
        targets = metrics["targets"]
        masks = metrics.get("masks")
    else:
        trainer = job.trainer
        idx = torch.randint(0, len(trainer.X_train), (trainer.args.batch_size,))
        inputs = trainer.X_train[idx]
        targets = trainer.Y_train[idx]
        masks = getattr(trainer, "masks_train", None)
        if masks is not None:
            masks = masks[idx]

    inputs = inputs.detach().cpu() if torch.is_tensor(inputs) else torch.tensor(inputs)
    targets = targets.detach().cpu() if torch.is_tensor(targets) else torch.tensor(targets)
    if masks is not None:
        masks = masks.detach().cpu() if torch.is_tensor(masks) else torch.tensor(masks)

    return inputs, targets, masks


def _forward_logits(job: TrainingJob, input_ids: List[int]) -> torch.Tensor:
    device = next(job.trainer.model.parameters()).device
    tokens_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = job.trainer.model(tokens_tensor)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    return logits


def _get_tokenizer(job: TrainingJob):
    tokenizer = getattr(job.trainer, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Tokenizer not available for this job")
    return tokenizer


def _decode_tokens(tokenizer, token_ids: List[int]) -> List[str]:
    return [_decode_token(tokenizer, token_id) for token_id in token_ids]


def _decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return f"T{token_id}"
