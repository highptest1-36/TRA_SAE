"""
TRA-SAE Model Loader
====================
Loads Qwen3.5-4B with BF16 precision for A100 GPU.
Uses vanilla transformers + PEFT — no Unsloth dependency.

Key responsibilities:
  - Drop the vision tower (saves ~1-2 GB VRAM for text-only task)
  - Patch chat template to always disable Qwen3.5 thinking mode
  - Apply LoRA adapters for fine-tuning
  - Multi-adapter loading for Agent v2 dispatch (logic vs physics)
  - Flash-Attention 2 on A100 for speed

Usage:
    from src.model_loader import load_base_model, apply_lora, load_peft_model

    # For training:
    model, tokenizer = load_base_model("Qwen/Qwen3.5-4B")
    model = apply_lora(model)

    # For inference (load saved LoRA):
    model, tokenizer = load_peft_model("Qwen/Qwen3.5-4B", "checkpoints/qwen35_sft/final")

    # For multi-adapter inference (Agent v2):
    model, tokenizer = load_multi_adapter_model(
        "Qwen/Qwen3.5-4B",
        {"physics": "checkpoints/qwen35_grpo_physics/final",
         "logic":   "checkpoints/qwen35_grpo_logic/final"}
    )
"""
from __future__ import annotations

import logging
import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)

logger = logging.getLogger("tra-sae.model_loader")

# Vision-module attribute names Qwen3.5 may expose
_VISION_ATTRS = ("visual", "vision_tower", "vision_model", "image_encoder", "vision")


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _detect_attn_impl() -> str:
    """Return best available attention implementation for current GPU."""
    try:
        import flash_attn  # noqa: F401 — just checking if installed
        cap = torch.cuda.get_device_capability()
        if cap[0] >= 8:   # Ampere (A100) or newer
            return "flash_attention_2"
    except ImportError:
        pass
    return "sdpa"   # scaled dot-product attention (PyTorch ≥ 2.0)


def _drop_vision_tower(model: Any) -> None:
    """Remove the vision encoder from a VLM to save VRAM."""
    for attr in _VISION_ATTRS:
        if hasattr(model, attr):
            logger.info(f"[model_loader] Dropping vision module: model.{attr}")
            try:
                delattr(model, attr)
            except AttributeError:
                pass
            # Also remove from model.config if referenced there
            if hasattr(model.config, attr):
                try:
                    delattr(model.config, attr)
                except AttributeError:
                    pass
            return  # Only drop the first match


def _patch_tokenizer(tokenizer: Any) -> Any:
    """Ensure tokenizer has a pad token and correct padding side."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"   # required for batched generation
    return tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_base_model(
    model_name: str = "Qwen/Qwen3.5-4B",
    dtype: torch.dtype = torch.bfloat16,
    drop_vision: bool = True,
    device_map: str = "auto",
    use_flash_attn: bool = True,
) -> tuple[Any, Any]:
    """Load Qwen3.5-4B base model + tokenizer in BF16.

    Args:
        model_name:     HuggingFace model ID.
        dtype:          Compute dtype (default bfloat16 for A100).
        drop_vision:    Remove vision encoder to save VRAM.
        device_map:     HF device map strategy ('auto' spreads across GPUs).
        use_flash_attn: Enable Flash Attention 2 when available.

    Returns:
        (model, tokenizer) tuple — model is on CUDA, eval mode.
    """
    attn_impl = _detect_attn_impl() if use_flash_attn else "eager"
    logger.info(
        f"[model_loader] Loading {model_name}  "
        f"dtype={dtype}  attn={attn_impl}  drop_vision={drop_vision}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer = _patch_tokenizer(tokenizer)

    load_kwargs: dict[str, Any] = dict(
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if attn_impl != "eager":
        load_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if drop_vision:
        _drop_vision_tower(model)

    model.config.use_cache = True
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"[model_loader] Model loaded — {total_params:.2f}B params")

    try:
        import torch
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"[model_loader] VRAM  {alloc:.2f} GB / {total:.1f} GB")
    except Exception:
        pass

    return model, tokenizer


def apply_lora(
    model: Any,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    adapter_name: str = "default",
) -> Any:
    """Wrap the base model with trainable LoRA adapters.

    Automatically probes the model to detect valid target modules —
    falls back to a safe subset if named modules are not present.

    Args:
        model:          The pre-loaded base model.
        lora_r:         LoRA rank.
        lora_alpha:     LoRA scaling factor.
        lora_dropout:   Dropout probability on LoRA matrices.
        target_modules: Module names to adapt. None → auto-detect.
        adapter_name:   Name of this adapter (for multi-adapter loading).

    Returns:
        PeftModel with LoRA adapters attached.
    """
    if target_modules is None:
        # Standard dense projections present in both Qwen3 and Qwen3.5
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        # Filter to modules actually present in this model
        param_names = {name for name, _ in model.named_modules()}
        target_modules = [
            m for m in target_modules
            if any(m in pn for pn in param_names)
        ]
        if not target_modules:
            raise RuntimeError(
                "No LoRA target modules found in model. "
                "Check model architecture and update LORA_TARGETS in config.py."
            )
        logger.info(f"[model_loader] LoRA targets: {target_modules}")

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )

    # Enable gradient checkpointing before wrapping
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    peft_model = get_peft_model(model, config, adapter_name=adapter_name)
    peft_model.print_trainable_parameters()

    return peft_model


def load_peft_model(
    base_model_name: str,
    peft_path: str,
    dtype: torch.dtype = torch.bfloat16,
    drop_vision: bool = True,
    adapter_name: str = "default",
    is_trainable: bool = False,
) -> tuple[Any, Any]:
    """Load a base model and attach a saved LoRA adapter for inference.

    Args:
        base_model_name:  HF model ID (must match the base model used during training).
        peft_path:        Path to saved PEFT adapter directory.
        dtype:            Compute dtype.
        drop_vision:      Drop vision tower after loading base.
        adapter_name:     Logical name for this adapter.
        is_trainable:     Set True if you plan to continue training this adapter.

    Returns:
        (model, tokenizer) — model in eval mode unless is_trainable=True.
    """
    model, tokenizer = load_base_model(
        base_model_name, dtype=dtype, drop_vision=drop_vision
    )
    model = PeftModel.from_pretrained(
        model,
        peft_path,
        adapter_name=adapter_name,
        is_trainable=is_trainable,
    )
    if not is_trainable:
        model.eval()

    logger.info(f"[model_loader] PEFT adapter '{adapter_name}' loaded ← {peft_path}")
    return model, tokenizer


def load_multi_adapter_model(
    base_model_name: str,
    adapters: dict[str, str],
    dtype: torch.dtype = torch.bfloat16,
    drop_vision: bool = True,
) -> tuple[Any, Any]:
    """Load a base model with multiple named LoRA adapters.

    Used by Agent v2 to route between physics and logic specialists.
    Call model.set_adapter(name) to switch adapters at inference time.

    Args:
        base_model_name:  HF model ID.
        adapters:         {adapter_name: path_to_adapter} dict.
                          Example: {"physics": "ckpts/...", "logic": "ckpts/..."}
        dtype:            Compute dtype.
        drop_vision:      Drop vision tower.

    Returns:
        (model, tokenizer) — model has all adapters loaded, defaults to first.
    """
    if not adapters:
        raise ValueError("adapters dict must have at least one entry.")

    model, tokenizer = load_base_model(
        base_model_name, dtype=dtype, drop_vision=drop_vision
    )

    first = True
    for adapter_name, adapter_path in adapters.items():
        if not os.path.isdir(adapter_path):
            logger.warning(
                f"[model_loader] Adapter path not found: {adapter_path} — skipping."
            )
            continue

        if first:
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                adapter_name=adapter_name,
                is_trainable=False,
            )
            first = False
        else:
            model.load_adapter(adapter_path, adapter_name=adapter_name)

        logger.info(f"[model_loader] Loaded adapter '{adapter_name}' ← {adapter_path}")

    model.eval()
    active = list(adapters.keys())
    if active:
        model.set_adapter(active[0])
        logger.info(f"[model_loader] Default adapter set to '{active[0]}'")

    return model, tokenizer
