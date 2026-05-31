"""
Phase 2 — GRPO Mixed Training (physics + logic, single adapter)
================================================================
Reinforcement learning with GRPO using the composite reward (reward.py).

Strategy
--------
  Base   : checkpoints/qwen35_sft_logic/final  (Phase 1.5 logic-SFT checkpoint)
  Data   : all training examples (physics + logic)
  Reward : batch_compute_reward (format + correctness + unit + length_penalty)
  Steps  : GRPO_MAX_STEPS = 250
  VRAM   : disable_adapter() as reference policy (saves ~10GB vs 2nd copy)
  Output : checkpoints/qwen35_grpo/final

Hardware: A100 40GB, BF16
VRAM budget:
  model (BF16, 4B) ≈ 8GB
  optimizer states  ≈ 10GB
  KV cache (8 gen)  ≈ 12GB
  gradients         ≈  4GB
  total             ≈ 34GB  ← safe on 40GB

Run:
    python run_phase2_grpo.py
    python run_phase2_grpo.py --smoke-test  (5 steps)
"""
from __future__ import annotations

import sys, os, time, json, argparse
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true")
args, _ = parser.parse_known_args()

# ── Config ────────────────────────────────────────────────────────────────────
from src.config import (
    MODEL_NAME, MAX_SEQ_LEN,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS,
    GRPO_BATCH, GRPO_GRAD_ACC, GRPO_LR, GRPO_MAX_STEPS,
    GRPO_NUM_GEN, GRPO_MAX_COMP, GRPO_BETA,
    TRAIN_DS, VAL_DS,
    QWEN35_SFT_LOGIC_FINAL, QWEN35_SFT_FINAL,
    QWEN35_GRPO_DIR, QWEN35_GRPO_FINAL,
    LOG_DIR, CKPT_DIR,
)
from src.utils import setup_logger, print_vram
from src.reward import batch_compute_reward
import torch
from pathlib import Path

logger = setup_logger("phase2_grpo", LOG_DIR)

_EFFECTIVE_BATCH = GRPO_BATCH * GRPO_GRAD_ACC
_SMOKE_STEPS     = 3

logger.info("=" * 68)
logger.info(f"Phase 2 GRPO  |  Base: qwen35_sft_logic/final")
logger.info(f"  Steps={GRPO_MAX_STEPS} | LR={GRPO_LR} | β={GRPO_BETA}")
logger.info(f"  Batch={GRPO_BATCH} × GradAcc={GRPO_GRAD_ACC} | gen={GRPO_NUM_GEN}")
logger.info(f"  Reference: disable_adapter() trick (saves ~10GB VRAM)")
logger.info(f"  Smoke-test={args.smoke_test}")
logger.info("=" * 68)

# ── 1. Determine base checkpoint ──────────────────────────────────────────────
# Use Phase 1.5 if available; fall back to Phase 1
if Path(QWEN35_SFT_LOGIC_FINAL).exists():
    _base_ckpt = QWEN35_SFT_LOGIC_FINAL
    logger.info(f"Using Phase 1.5 checkpoint: {_base_ckpt}")
else:
    _base_ckpt = QWEN35_SFT_FINAL
    logger.warning(
        f"Phase 1.5 checkpoint not found → falling back to Phase 1: {_base_ckpt}"
    )

# ── 2. Load model ─────────────────────────────────────────────────────────────
from src.model_loader import load_base_model
from peft import PeftModel

print(f"\n[1/5] Loading base model + Phase 1.5 LoRA")
base_model, tokenizer = load_base_model(
    model_name  = MODEL_NAME,
    dtype       = torch.bfloat16,
    drop_vision = True,
    device_map  = "auto",
)

if Path(_base_ckpt).exists():
    model = PeftModel.from_pretrained(base_model, _base_ckpt, is_trainable=True)
    logger.info("PEFT checkpoint loaded")
else:
    from src.model_loader import apply_lora
    logger.warning("No checkpoint found — applying fresh LoRA for smoke-test")
    model = apply_lora(base_model, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
                       lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS)

model.print_trainable_parameters()
print_vram("after model load")

# ── 3. Load + prep dataset ────────────────────────────────────────────────────
from datasets import load_from_disk, Dataset

print(f"\n[2/5] Loading datasets")
train_ds = load_from_disk(TRAIN_DS)
val_ds   = load_from_disk(VAL_DS)
print(f"      Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")
logger.info(f"Datasets: train={len(train_ds)} val={len(val_ds)}")

def _make_grpo_prompt(sample: dict) -> str:
    """Convert a dataset sample to a formatted prompt string for GRPO."""
    try:
        return tokenizer.apply_chat_template(
            sample["prompt"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            sample["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )

def format_for_grpo(examples):
    """Prepare dataset columns expected by GRPOTrainer."""
    prompts, gts, subjects, questions = [], [], [], []
    for i in range(len(examples["prompt"])):
        sample = {k: examples[k][i] for k in examples}
        prompt_str = _make_grpo_prompt(sample)
        prompts.append(prompt_str)
        gts.append(str(sample.get("answer", "")))
        subjects.append(str(sample.get("type", "")))
        # Extract question text for Z3 usage
        q_text = ""
        for msg in sample["prompt"]:
            if isinstance(msg, dict) and msg.get("role") == "user":
                q_text = msg["content"]
                break
        questions.append(q_text)
    return {
        "prompt":       prompts,
        "ground_truth": gts,
        "subject":      subjects,
        "question":     questions,
    }

print("\n[3/5] Formatting dataset for GRPO…")
train_grpo = train_ds.map(format_for_grpo, batched=True,
                           remove_columns=train_ds.column_names,
                           desc="grpo_train")
val_grpo   = val_ds.map(format_for_grpo, batched=True,
                         remove_columns=val_ds.column_names,
                         desc="grpo_val")
logger.info(f"GRPO dataset prepared: train={len(train_grpo)} val={len(val_grpo)}")

# ── 4. GRPO Config + Trainer ──────────────────────────────────────────────────
from trl import GRPOTrainer, GRPOConfig

TB_DIR = Path(LOG_DIR) / "phase2_tensorboard"
TB_DIR.mkdir(parents=True, exist_ok=True)
Path(QWEN35_GRPO_DIR).mkdir(parents=True, exist_ok=True)

_steps = _SMOKE_STEPS if args.smoke_test else GRPO_MAX_STEPS
logger.info(f"GRPO max_steps={_steps}")

grpo_config = GRPOConfig(
    output_dir                   = QWEN35_GRPO_DIR,

    # Batch
    per_device_train_batch_size  = GRPO_BATCH,
    gradient_accumulation_steps  = GRPO_GRAD_ACC,
    dataloader_num_workers       = 2,

    # Steps / LR
    max_steps                    = _steps,
    learning_rate                = GRPO_LR,
    lr_scheduler_type            = "cosine",
    warmup_steps                 = 10 if not args.smoke_test else 0,

    # GRPO params
    num_generations              = GRPO_NUM_GEN,
    max_completion_length        = GRPO_MAX_COMP,
    beta                         = GRPO_BETA,

    # VRAM: use disable_adapter() for reference instead of second model copy
    use_vllm                     = False,

    # Precision
    bf16                         = True,
    fp16                         = False,

    # Optimizer
    optim                        = "adamw_torch",
    gradient_checkpointing       = True,

    # Logging
    logging_steps                = 5,
    logging_dir                  = str(TB_DIR),
    report_to                    = ["tensorboard"],

    # Save
    save_strategy                = "steps",
    save_steps                   = max(10, _steps // 5),
    save_total_limit             = 3,

    seed                         = 42,
)

print(f"\n[4/5] Building GRPOTrainer…")
grpo_trainer = GRPOTrainer(
    model              = model,
    processing_class   = tokenizer,
    args               = grpo_config,
    train_dataset      = train_grpo,
    reward_funcs       = [batch_compute_reward],
)
print_vram("before GRPO training")

# ── 5. Train ──────────────────────────────────────────────────────────────────
print(f"\n[5/5] GRPO training  (steps={_steps}  gen={GRPO_NUM_GEN}  LR={GRPO_LR})")
print("=" * 68)
t0    = time.time()
# Resume from latest checkpoint if available (handles Colab session resets)
import os as _os
_ckpt_dir = str(QWEN35_GRPO_DIR)
_resume = None
if _os.path.isdir(_ckpt_dir):
    _ckpts = sorted(
        [d for d in _os.listdir(_ckpt_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1])
    )
    if _ckpts:
        _resume = _os.path.join(_ckpt_dir, _ckpts[-1])
        print(f"  ↳ Resuming from {_resume}")
stats = grpo_trainer.train(resume_from_checkpoint=_resume)
elapsed_min = (time.time() - t0) / 60
print_vram("after GRPO")

# ── Save final checkpoint ─────────────────────────────────────────────────────
print(f"\n  Saving → {QWEN35_GRPO_FINAL}")
Path(QWEN35_GRPO_FINAL).mkdir(parents=True, exist_ok=True)
model.save_pretrained(QWEN35_GRPO_FINAL)
tokenizer.save_pretrained(QWEN35_GRPO_FINAL)
logger.info(f"Checkpoint saved → {QWEN35_GRPO_FINAL}")

# ── Save results JSON ─────────────────────────────────────────────────────────
result_info = {
    "phase":            "2_grpo",
    "model":            MODEL_NAME,
    "base_checkpoint":  _base_ckpt,
    "smoke_test":       args.smoke_test,
    "training_loss":    round(getattr(stats, "training_loss", 0.0), 6),
    "global_steps":     getattr(stats, "global_step", _steps),
    "elapsed_min":      round(elapsed_min, 1),
    "grpo_steps":       _steps,
    "grpo_lr":          GRPO_LR,
    "grpo_beta":        GRPO_BETA,
    "grpo_num_gen":     GRPO_NUM_GEN,
    "grpo_max_comp":    GRPO_MAX_COMP,
    "eff_batch":        _EFFECTIVE_BATCH,
    "dtype":            "bfloat16",
    "vram_trick":       "disable_adapter_reference",
    "train_size":       len(train_ds),
    "val_size":         len(val_ds),
    "checkpoint":       QWEN35_GRPO_FINAL,
    "tensorboard":      str(TB_DIR),
}
results_path = Path(LOG_DIR) / "phase2_grpo_results.json"
with open(results_path, "w") as f:
    json.dump(result_info, f, indent=2)

logger.info(f"Phase 2 GRPO done | steps={result_info['global_steps']} time={elapsed_min:.1f}min")

print("=" * 68)
print(" PHASE 2 GRPO COMPLETE")
print(f"  Steps           : {result_info['global_steps']}")
print(f"  Time elapsed    : {elapsed_min:.1f} min")
print(f"  Checkpoint      : {QWEN35_GRPO_FINAL}")
print(f"  Results JSON    : {results_path}")
print(f"  TensorBoard     : {TB_DIR}")
print(f"  Next steps      : run_phase2_grpo_physics.py  &  run_phase2_grpo_logic.py")
print("=" * 68)
