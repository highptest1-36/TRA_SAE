"""
Phase 1 — SFT Warm-up Training (Qwen3.5-4B, vanilla transformers + PEFT)
==========================================================================
Full rewrite: NO Unsloth.  Uses src.model_loader + TRL SFTTrainer.

Hardware target : A100 40 GB, BF16, Flash Attention 2
Base model      : Qwen/Qwen3.5-4B
LoRA            : r=32, alpha=64, targets=all MLP+Attn projectors
Effective batch : SFT_BATCH(4) × SFT_GRAD_ACC(4) = 16
Epochs          : SFT_EPOCHS = 3
Output          : checkpoints/qwen35_sft/final/

Run:
    python run_phase1_sft.py                # full training
    python run_phase1_sft.py --smoke-test   # 5-step smoke test
"""
from __future__ import annotations

import sys, os, time, json, argparse, math
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")

# ── Args ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true",
                    help="Run only 5 steps for quick validation")
args, _ = parser.parse_known_args()

# ── Config ───────────────────────────────────────────────────────────────────
from src.config import (
    MODEL_NAME, MAX_SEQ_LEN,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS,
    SFT_BATCH, SFT_GRAD_ACC, SFT_EPOCHS, SFT_LR, SFT_WARMUP, SFT_SAVE_STEPS,
    TRAIN_DS, VAL_DS,
    QWEN35_SFT_DIR, QWEN35_SFT_FINAL,
    LOG_DIR, CKPT_DIR,
)
from src.utils import setup_logger, print_vram

import torch
from pathlib import Path

logger = setup_logger("phase1_sft", LOG_DIR)

_EFFECTIVE_BATCH = SFT_BATCH * SFT_GRAD_ACC

logger.info("=" * 68)
logger.info(f"Phase 1 SFT  |  Model: {MODEL_NAME}")
logger.info(f"  LoRA r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}")
logger.info(f"  Epochs={SFT_EPOCHS} | LR={SFT_LR} | eff_batch={_EFFECTIVE_BATCH}")
logger.info(f"  max_seq={MAX_SEQ_LEN} | BF16=True | Flash-Attn2 (if available)")
logger.info(f"  Smoke-test={args.smoke_test}")
logger.info("=" * 68)

# ── 1. Load base model + tokenizer ───────────────────────────────────────────
from src.model_loader import load_base_model, apply_lora

print(f"\n[1/6] Loading base model: {MODEL_NAME}")
model, tokenizer = load_base_model(
    model_name  = MODEL_NAME,
    dtype       = torch.bfloat16,
    drop_vision = True,
    device_map  = "auto",
)
print_vram("after base model load")
logger.info(f"Base model loaded — dtype={model.dtype}")

# ── 2. Apply LoRA ────────────────────────────────────────────────────────────
print(f"\n[2/6] Applying LoRA  (r={LORA_R}, alpha={LORA_ALPHA})")
model = apply_lora(
    model,
    lora_r         = LORA_R,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    target_modules = LORA_TARGETS,
)
model.print_trainable_parameters()
print_vram("after LoRA")
logger.info(f"LoRA applied — r={LORA_R}")

# ── 3. Load datasets ──────────────────────────────────────────────────────────
from datasets import load_from_disk

print(f"\n[3/6] Loading datasets")
train_ds = load_from_disk(TRAIN_DS)
val_ds   = load_from_disk(VAL_DS)
phys_n  = sum(1 for s in train_ds if s.get("type") == "physics")
logic_n = sum(1 for s in train_ds if s.get("type") == "logic")
print(f"      Physics: {phys_n}  Logic: {logic_n}")
logger.info(f"Datasets loaded — train={len(train_ds)} val={len(val_ds)} physics={phys_n} logic={logic_n}")

# ── 4. Chat-template formatting ───────────────────────────────────────────────
def apply_chat_template(examples):
    """Apply Qwen3.5 chat template with enable_thinking=False."""
    texts = []
    for p in examples["prompt"]:
        try:
            txt = tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            txt = tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=False,
            )
        texts.append(txt)
    return {"text": texts}

print("\n[4/6] Formatting datasets with chat template…")
train_fmt = train_ds.map(apply_chat_template, batched=True,
                          remove_columns=["prompt"], desc="train")
val_fmt   = val_ds.map(apply_chat_template,   batched=True,
                        remove_columns=["prompt"], desc="val")
sample_len = len(train_fmt[0]["text"])
print(f"      Sample[0] length: {sample_len:,} chars")
logger.info(f"Chat template applied | sample_len={sample_len}")

# ── 5. SFT Config + Trainer ───────────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig
import math

TB_DIR = Path(LOG_DIR) / "phase1_tensorboard"
TB_DIR.mkdir(parents=True, exist_ok=True)
Path(QWEN35_SFT_DIR).mkdir(parents=True, exist_ok=True)

steps_per_epoch = math.ceil(len(train_fmt) / _EFFECTIVE_BATCH)
total_steps     = steps_per_epoch * SFT_EPOCHS

if args.smoke_test:
    print("\n  *** SMOKE TEST: only 5 steps ***")
    _max_steps  = 5
    _eval_steps = 5
    _save_steps = 5
    _epochs     = 1
    _warmup     = 0
else:
    _max_steps  = -1
    _eval_steps = SFT_SAVE_STEPS
    _save_steps = SFT_SAVE_STEPS
    _epochs     = SFT_EPOCHS
    _warmup     = SFT_WARMUP

logger.info(
    f"Training plan: {_epochs} epoch(s) | ~{total_steps} total steps | "
    f"warmup={_warmup} | eff_batch={_EFFECTIVE_BATCH}"
)

sft_config = SFTConfig(
    output_dir                  = QWEN35_SFT_DIR,
    per_device_train_batch_size = SFT_BATCH,
    gradient_accumulation_steps = SFT_GRAD_ACC,
    dataloader_num_workers      = 2,
    num_train_epochs            = _epochs,
    max_steps                   = _max_steps,
    learning_rate               = SFT_LR,
    lr_scheduler_type           = "cosine",
    warmup_steps                = _warmup,
    bf16                        = True,
    fp16                        = False,
    optim                       = "adamw_torch",
    gradient_checkpointing      = True,
    logging_steps               = 10,
    logging_dir                 = str(TB_DIR),
    report_to                   = ["tensorboard"],
    eval_strategy               = "steps",
    eval_steps                  = _eval_steps,
    save_strategy               = "steps",
    save_steps                  = _save_steps,
    save_total_limit            = 3,
    load_best_model_at_end      = True,
    max_length                  = MAX_SEQ_LEN,
    dataset_text_field          = "text",
    seed                        = 42,
)

print(f"\n[5/6] Building SFTTrainer…")
sft_trainer = SFTTrainer(
    model             = model,
    processing_class  = tokenizer,
    train_dataset     = train_fmt,
    eval_dataset      = val_fmt,
    args              = sft_config,
)
print_vram("before training")

# ── 6. Train ──────────────────────────────────────────────────────────────────
print(f"\n[6/6] Training  (eff_batch={_EFFECTIVE_BATCH}  epochs={_epochs}  lr={SFT_LR})")
print("=" * 68)
t0    = time.time()
stats = sft_trainer.train()
elapsed_min = (time.time() - t0) / 60
print_vram("after training")

# ── Save final checkpoint ─────────────────────────────────────────────────────
print(f"\n  Saving → {QWEN35_SFT_FINAL}")
Path(QWEN35_SFT_FINAL).mkdir(parents=True, exist_ok=True)
model.save_pretrained(QWEN35_SFT_FINAL)
tokenizer.save_pretrained(QWEN35_SFT_FINAL)
logger.info(f"Checkpoint saved → {QWEN35_SFT_FINAL}")

# ── Save results JSON ─────────────────────────────────────────────────────────
result_info = {
    "phase":          "1_sft",
    "model":          MODEL_NAME,
    "smoke_test":     args.smoke_test,
    "training_loss":  round(stats.training_loss, 6),
    "global_steps":   stats.global_step,
    "elapsed_min":    round(elapsed_min, 1),
    "epochs":         _epochs,
    "eff_batch_size": _EFFECTIVE_BATCH,
    "learning_rate":  SFT_LR,
    "lr_scheduler":   "cosine",
    "warmup_steps":   _warmup,
    "max_seq_len":    MAX_SEQ_LEN,
    "lora_r":         LORA_R,
    "lora_alpha":     LORA_ALPHA,
    "lora_targets":   LORA_TARGETS,
    "dtype":          "bfloat16",
    "train_size":     len(train_ds),
    "val_size":       len(val_ds),
    "physics_train":  phys_n,
    "logic_train":    logic_n,
    "checkpoint":     QWEN35_SFT_FINAL,
    "tensorboard":    str(TB_DIR),
}
results_path = Path(LOG_DIR) / "phase1_sft_results.json"
results_path.parent.mkdir(parents=True, exist_ok=True)
with open(results_path, "w") as f:
    json.dump(result_info, f, indent=2)

logger.info(
    f"Phase 1 done | loss={stats.training_loss:.4f} "
    f"steps={stats.global_step} time={elapsed_min:.1f}min"
)

print("=" * 68)
print(" PHASE 1 SFT COMPLETE")
print(f"  Training loss   : {stats.training_loss:.4f}")
print(f"  Global steps    : {stats.global_step}")
print(f"  Time elapsed    : {elapsed_min:.1f} min")
print(f"  Checkpoint      : {QWEN35_SFT_FINAL}")
print(f"  Results JSON    : {results_path}")
print(f"  TensorBoard     : {TB_DIR}")
print(f"  Next step       : python run_phase1_5_logic_sft.py")
print("=" * 68)
