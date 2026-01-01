"""Code snippet extraction utilities for API endpoints."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, List, Tuple

GITHUB_REPO_URL = "https://github.com/jammastergirish/BuildAnLLM"


def build_model_code_snippets(config: Dict[str, object]) -> List[Dict[str, object]]:
    components = _determine_components_to_show(config)
    snippets: List[Dict[str, object]] = []

    snippets.append(_snippet_for_object(
        title="1. Model",
        module_path="pretraining.model.model",
        object_name=components["model"]["class"],
    ))

    snippets.append(_snippet_for_object(
        title="2. Transformer Block",
        module_path="pretraining.transformer_blocks.transformer_block",
        object_name=components["transformer_block"]["class"],
    ))

    snippets.append(_snippet_for_object(
        title="3. Attention Mechanism",
        module_path="pretraining.attention.attention",
        object_name=components["attention"]["class"],
    ))

    if "positional_encoding" in components:
        pos = components["positional_encoding"]
        title_map = {
            "learned": "4. Positional Embeddings (Learned)",
            "rope": "4. RoPE (Rotary Position Embedding)",
            "alibi": "4. ALiBi (Attention with Linear Biases)",
        }
        snippets.append(_snippet_for_object(
            title=title_map.get(pos["type"], "4. Positional Encoding"),
            module_path=pos["module"],
            object_name=pos["class"],
        ))

    snippets.append(_snippet_for_object(
        title="5. MLP",
        module_path="pretraining.mlp.mlp",
        object_name=components["mlp"]["class"],
    ))

    norm = components["normalization"]
    snippets.append(_snippet_for_object(
        title="6. Normalization",
        module_path=f"pretraining.normalization.{norm['file']}",
        object_name=norm["class"],
    ))

    snippets.append(_snippet_for_object(
        title="7. Token Embeddings",
        module_path="pretraining.embeddings.embed",
        object_name=components["embeddings"]["class"],
    ))

    return snippets


def build_inference_code_snippets() -> List[Dict[str, object]]:
    return [
        _snippet_for_object(
            title="1. Transformer Sampler",
            module_path="inference.sampler",
            object_name="TransformerSampler",
        )
    ]


def build_finetuning_code_snippets(use_lora: bool = False) -> List[Dict[str, object]]:
    snippets = [
        _snippet_for_object(
            title="1. SFT Dataset",
            module_path="finetuning.data.sft_dataset",
            object_name="SFTDataset",
        ),
        _snippet_for_object(
            title="2. SFT Trainer",
            module_path="finetuning.training.sft_trainer",
            object_name="SFTTrainer",
        ),
    ]

    if use_lora:
        snippets.append(_snippet_for_object(
            title="3. LoRA Conversion",
            module_path="finetuning.peft.lora_utils",
            object_name="convert_model_to_lora",
        ))
        snippets.append(_snippet_for_object(
            title="4. LoRA Matrix Creation",
            module_path="finetuning.peft.lora_wrappers",
            object_name="create_lora_matrices",
        ))
        snippets.append(_snippet_for_object(
            title="5. LoRA Einsum Computation",
            module_path="finetuning.peft.lora_wrappers",
            object_name="einsum_with_lora",
        ))

    return snippets


def _snippet_for_object(title: str, module_path: str, object_name: str) -> Dict[str, object]:
    source_code, start_line, end_line, file_path = _get_object_source_with_lines(
        module_path, object_name
    )
    rel_path = _get_file_relative_path(file_path)
    github_link = _generate_github_link(file_path, start_line, end_line)
    return {
        "title": title,
        "module": module_path,
        "object": object_name,
        "file": rel_path,
        "start_line": start_line,
        "end_line": end_line,
        "github_url": github_link,
        "code": source_code,
    }


def _get_object_source_with_lines(module_path: str, object_name: str) -> Tuple[str, int, int, str]:
    module = __import__(module_path, fromlist=[object_name])
    obj = getattr(module, object_name)
    source_lines, start_line = inspect.getsourcelines(obj)
    source_code = "".join(source_lines)
    end_line = start_line + len(source_lines) - 1
    file_path = inspect.getfile(obj)
    return source_code, start_line, end_line, file_path


def _get_file_relative_path(absolute_path: str) -> str:
    try:
        current = Path(absolute_path).resolve()
        project_root = None
        for _ in range(10):
            if (current / "main.py").exists() or (current / "pyproject.toml").exists():
                project_root = current
                break
            if current == current.parent:
                break
            current = current.parent

        if project_root:
            try:
                rel_path = Path(absolute_path).relative_to(project_root)
                return str(rel_path).replace("\\", "/")
            except ValueError:
                pass

        parts = Path(absolute_path).parts
        for i, part in enumerate(parts):
            if part in ["pretraining", "finetuning", "inference"]:
                return "/".join(parts[i:])
        return Path(absolute_path).name
    except Exception:
        return Path(absolute_path).name


def _generate_github_link(
    file_path: str,
    start_line: int,
    end_line: int,
    github_repo_url: str = GITHUB_REPO_URL,
    branch: str = "main",
) -> str:
    rel_path = _get_file_relative_path(file_path)
    repo_url = github_repo_url.rstrip("/")
    rel_path_url = rel_path.replace("\\", "/")
    return f"{repo_url}/blob/{branch}/{rel_path_url}#L{start_line}-L{end_line}"


def _determine_components_to_show(config: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    use_einops = config.get("use_einops", True)
    pos_enc = config.get("positional_encoding", "learned")
    norm = config.get("normalization", "layernorm")
    activation = config.get("activation", "gelu")

    components: Dict[str, Dict[str, object]] = {}

    components["model"] = {
        "class": "TransformerModel",
        "use_einops": use_einops,
    }

    components["transformer_block"] = {
        "class": "TransformerBlock",
        "use_einops": use_einops,
    }

    components["attention"] = {
        "class": "Attention",
        "pos_enc": pos_enc,
    }

    use_moe = config.get("use_moe", False)
    if use_moe:
        components["mlp"] = {
            "class": "MoEMLP",
            "activation": activation,
            "use_moe": True,
        }
    elif activation == "swiglu":
        components["mlp"] = {
            "class": "MLPSwiGLU",
            "activation": activation,
        }
    else:
        components["mlp"] = {
            "class": "MLP",
            "activation": activation,
        }

    if norm == "rmsnorm":
        components["normalization"] = {
            "class": "RMSNorm",
            "file": "rmsnorm",
            "norm": norm,
        }
    else:
        components["normalization"] = {
            "class": "LayerNorm",
            "file": "layernorm",
            "norm": norm,
        }

    if pos_enc == "learned":
        components["positional_encoding"] = {
            "class": "PosEmbed",
            "type": "learned",
            "module": "pretraining.positional_embeddings.positional_embedding",
        }
    elif pos_enc == "rope":
        components["positional_encoding"] = {
            "class": "RoPE",
            "type": "rope",
            "module": "pretraining.positional_embeddings.rope",
            "method": "forward",
        }
    elif pos_enc == "alibi":
        components["positional_encoding"] = {
            "class": "ALiBi",
            "type": "alibi",
            "module": "pretraining.positional_embeddings.alibi",
            "method": "get_bias",
        }

    components["embeddings"] = {
        "class": "EmbedWithoutTorch" if use_einops else "EmbedWithTorch",
        "use_einops": use_einops,
    }

    return components

