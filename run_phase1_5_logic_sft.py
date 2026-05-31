"""
Phase 1.5 — Logic-Specialist SFT (curriculum fine-tune on logic subset)
=========================================================================
Continues training from Phase 1 checkpoint (qwen35_sft/final) using only
logic examples (~732 training samples).

FOL-augmented reasoning: system prompt emphasizes explicit First-Order
Logic CoT steps (∀x, ∃x, →, ¬) before answering.

Hardware : A100 40GB, BF16
Base     : checkpoints/qwen35_sft/final  (LoRA checkpoint from Phase 1)
LoRA     : same r=32, alpha=64 adapters
LR       : LOGIC_SFT_LR = 5e-5  (lower than Phase 1 to avoid catastrophic forgetting)
Epochs   : LOGIC_SFT_EPOCHS = 2
Output   : checkpoints/qwen35_sft_logic/final

Run:
    python run_phase1_5_logic_sft.py
    python run_phase1_5_logic_sft.py --smoke-test
"""
from __future__ import annotations

import sys, os, time, json, argparse, math
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true",
                    help="Run only 5 steps for quick validation")
args, _ = parser.parse_known_args()

# ── Config ────────────────────────────────────────────────────────────────────
from src.config import (
    MODEL_NAME, MAX_SEQ_LEN,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS,
    SFT_BATCH, SFT_GRAD_ACC, SFT_SAVE_STEPS,
    LOGIC_SFT_EPOCHS, LOGIC_SFT_LR, LOGIC_SFT_GRAD_ACC,
    TRAIN_DS, VAL_DS,
    QWEN35_SFT_FINAL,
    QWEN35_SFT_LOGIC_DIR, QWEN35_SFT_LOGIC_FINAL,
    LOG_DIR, CKPT_DIR,
)
from src.utils import setup_logger, print_vram
import torch
from pathlib import Path

logger = setup_logger("phase1_5_logic_sft", LOG_DIR)

_BATCH = SFT_BATCH
_GRAD  = LOGIC_SFT_GRAD_ACC
_EFFECTIVE_BATCH = _BATCH * _GRAD

logger.info("=" * 68)
logger.info(f"Phase 1.5 Logic SFT  |  Base: {QWEN35_SFT_FINAL}")
logger.info(f"  LoRA r={LORA_R} alpha={LORA_ALPHA} | LR={LOGIC_SFT_LR}")
logger.info(f"  Epochs={LOGIC_SFT_EPOCHS} | eff_batch={_EFFECTIVE_BATCH}")
logger.info(f"  Logic-only curriculum + FOL CoT augmentation")
logger.info(f"  Smoke-test={args.smoke_test}")
logger.info("=" * 68)

# ── FOL-augmented system prompt ───────────────────────────────────────────────
_LOGIC_SYSTEM_PROMPT = (
    "You are a precise logical reasoning assistant. "
    "When solving logic problems:\n"
    "1. Identify the relevant premises and translate them to FOL notation "
    "(∀x, ∃x, →, ¬, ∧, ∨).\n"
    "2. Apply modus ponens / modus tollens / universal instantiation step-by-step.\n"
    "3. State your conclusion clearly.\n"
    "Format: <reasoning>FOL steps here</reasoning>"
    "<answer>Yes|No|Unknown|A|B|C|D</answer>"
    "<explanation>plain language justification</explanation>"
)

# ── 1. Load base model from Phase 1 checkpoint ───────────────────────────────
from src.model_loader import load_base_model
from peft import PeftModel

print(f"\n[1/6] Loading Phase 1 checkpoint: {QWEN35_SFT_FINAL}")
base_model, tokenizer = load_base_model(
    model_name  = MODEL_NAME,
    dtype       = torch.bfloat16,
    drop_vision = True,
    device_map  = "auto",
)

if Path(QWEN35_SFT_FINAL).exists():
    print(f"      Loading PEFT adapters from {QWEN35_SFT_FINAL}…")
    model = PeftModel.from_pretrained(
        base_model,
        QWEN35_SFT_FINAL,
        is_trainable=True,
    )
else:
    # Phase 1 not done yet — start from base + new LoRA (fallback for testing)
    logger.warning(
        f"[phase1.5] Phase 1 checkpoint not found at {QWEN35_SFT_FINAL}. "
        f"Starting from fresh LoRA."
    )
    from src.model_loader import apply_lora
    model = apply_lora(
        base_model,
        lora_r         = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        target_modules = LORA_TARGETS,
    )

model.print_trainable_parameters()
print_vram("after Phase 1 checkpoint load")
logger.info("Phase 1 checkpoint loaded successfully")

# ── 2. Load datasets — logic subset only ─────────────────────────────────────
from datasets import load_from_disk

print(f"\n[2/6] Loading and filtering logic examples")
train_full = load_from_disk(TRAIN_DS)
val_full   = load_from_disk(VAL_DS)

train_logic = train_full.filter(lambda x: x.get("type") == "logic")
val_logic   = val_full.filter(lambda x: x.get("type") == "logic")

print(f"      Logic train: {len(train_logic):,}  |  Logic val: {len(val_logic):,}")
logger.info(
    f"Logic subsets — train={len(train_logic)} val={len(val_logic)}"
)

# ── 3. FOL-augmented chat template ───────────────────────────────────────────
def apply_logic_chat_template(examples):
    """Apply chat template, injecting FOL-emphasis system prompt for logic."""
    texts = []
    for prompt in examples["prompt"]:
        # Inject the logic-specialist system prompt as the first message
        augmented = [{"role": "system", "content": _LOGIC_SYSTEM_PROMPT}]
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "system":
                # Replace default system prompt
                continue
            augmented.append(msg)

        try:
            txt = tokenizer.apply_chat_template(
                augmented,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            txt = tokenizer.apply_chat_template(
                augmented,
                tokenize=False,
                add_generation_prompt=False,
            )
        texts.append(txt)
    return {"text": texts}

print("\n[3/6] Applying FOL-augmented chat template…")
train_fmt = train_logic.map(apply_logic_chat_template, batched=True,
                             remove_columns=["prompt"], desc="logic_train")
val_fmt   = val_logic.map(apply_logic_chat_template,   batched=True,
                           remove_columns=["prompt"], desc="logic_val")

sample_len = len(train_fmt[0]["text"])
print(f"      Sample[0] length: {sample_len:,} chars")
logger.info(f"Chat template applied | sample_len={sample_len}")

# ── 4. SFT Config ─────────────────────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig

TB_DIR = Path(LOG_DIR) / "phase1_5_tensorboard"
TB_DIR.mkdir(parents=True, exist_ok=True)
Path(QWEN35_SFT_LOGIC_DIR).mkdir(parents=True, exist_ok=True)

steps_per_epoch = math.ceil(len(train_fmt) / _EFFECTIVE_BATCH)
total_steps     = steps_per_epoch * LOGIC_SFT_EPOCHS

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
    _epochs     = LOGIC_SFT_EPOCHS
    _warmup     = 20   # shorter warmup for curriculum fine-tune

logger.info(
    f"Training plan: {_epochs} epoch(s) | ~{total_steps} total steps | "
    f"warmup={_warmup} | LR={LOGIC_SFT_LR} | eff_batch={_EFFECTIVE_BATCH}"
)

sft_config = SFTConfig(
    output_dir                  = QWEN35_SFT_LOGIC_DIR,
    per_device_train_batch_size = _BATCH,
    gradient_accumulation_steps = _GRAD,
    dataloader_num_workers      = 2,
    num_train_epochs            = _epochs,
    max_steps                   = _max_steps,
    learning_rate               = LOGIC_SFT_LR,
    lr_scheduler_type           = "cosine",
    warmup_steps                = _warmup,
    bf16                        = True,
    fp16                        = False,
    optim                       = "adamw_torch",
    gradient_checkpointing      = True,
    logging_steps               = 5,
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

print(f"\n[4/6] Building SFTTrainer (logic curriculum)…")
sft_trainer = SFTTrainer(
    model             = model,
    processing_class  = tokenizer,
    train_dataset     = train_fmt,
    eval_dataset      = val_fmt,
    args              = sft_config,
)
print_vram("before training")

# ── 5. Train ──────────────────────────────────────────────────────────────────
print(f"\n[5/6] Training  (LR={LOGIC_SFT_LR}  epochs={_epochs}  logic-only)")
print("=" * 68)
t0    = time.time()
stats = sft_trainer.train()
elapsed_min = (time.time() - t0) / 60
print_vram("after training")

# ── 6. Save final checkpoint ──────────────────────────────────────────────────
print(f"\n[6/6] Saving → {QWEN35_SFT_LOGIC_FINAL}")
Path(QWEN35_SFT_LOGIC_FINAL).mkdir(parents=True, exist_ok=True)
model.save_pretrained(QWEN35_SFT_LOGIC_FINAL)
tokenizer.save_pretrained(QWEN35_SFT_LOGIC_FINAL)
logger.info(f"Checkpoint saved → {QWEN35_SFT_LOGIC_FINAL}")

# ── Save results JSON ─────────────────────────────────────────────────────────
result_info = {
    "phase":            "1.5_logic_sft",
    "model":            MODEL_NAME,
    "base_checkpoint":  QWEN35_SFT_FINAL,
    "smoke_test":       args.smoke_test,
    "training_loss":    round(stats.training_loss, 6),
    "global_steps":     stats.global_step,
    "elapsed_min":      round(elapsed_min, 1),
    "epochs":           _epochs,
    "eff_batch_size":   _EFFECTIVE_BATCH,
    "learning_rate":    LOGIC_SFT_LR,
    "lr_scheduler":     "cosine",
    "warmup_steps":     _warmup,
    "max_seq_len":      MAX_SEQ_LEN,
    "lora_r":           LORA_R,
    "lora_alpha":       LORA_ALPHA,
    "dtype":            "bfloat16",
    "logic_train_size": len(train_logic),
    "logic_val_size":   len(val_logic),
    "checkpoint":       QWEN35_SFT_LOGIC_FINAL,
    "tensorboard":      str(TB_DIR),
    "fol_system_prompt": True,
}
results_path = Path(LOG_DIR) / "phase1_5_logic_sft_results.json"
results_path.parent.mkdir(parents=True, exist_ok=True)
with open(results_path, "w") as f:
    json.dump(result_info, f, indent=2)

logger.info(
    f"Phase 1.5 done | loss={stats.training_loss:.4f} "
    f"steps={stats.global_step} time={elapsed_min:.1f}min"
)

print("=" * 68)
print(" PHASE 1.5 LOGIC SFT COMPLETE")
print(f"  Training loss   : {stats.training_loss:.4f}")
print(f"  Global steps    : {stats.global_step}")
print(f"  Time elapsed    : {elapsed_min:.1f} min")
print(f"  Checkpoint      : {QWEN35_SFT_LOGIC_FINAL}")
print(f"  Results JSON    : {results_path}")
print(f"  TensorBoard     : {TB_DIR}")
print(f"  Next step       : python run_phase2_grpo.py")
print("=" * 68)
