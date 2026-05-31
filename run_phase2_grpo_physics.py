"""
Phase 2P — Physics Specialist GRPO Training
=============================================
GRPO fine-tune on the physics subset only, starting from Phase 1 SFT.

Purpose  : Train the physics LoRA specialist adapter that will be loaded
           alongside the logic adapter in the dual-LoRA Agent v2.

Base     : checkpoints/qwen35_sft/final           (Phase 1, NOT logic version)
Data     : physics subset (~1213 training samples)
Steps    : 200
Reward   : batch_compute_reward (correctness + unit reward strongly weighted)
Output   : checkpoints/qwen35_grpo_physics/final

Run:
    python run_phase2_grpo_physics.py
    python run_phase2_grpo_physics.py --smoke-test
"""
from __future__ import annotations

import sys, os, time, json, argparse
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true")
args, _ = parser.parse_known_args()

from src.config import (
    MODEL_NAME, MAX_SEQ_LEN,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS,
    GRPO_BATCH, GRPO_GRAD_ACC, GRPO_LR, GRPO_NUM_GEN, GRPO_MAX_COMP, GRPO_BETA,
    TRAIN_DS, VAL_DS,
    QWEN35_SFT_FINAL,
    QWEN35_GRPO_PHYS_DIR, QWEN35_GRPO_PHYS_FINAL,
    LOG_DIR,
)
from src.utils import setup_logger, print_vram
from src.reward import batch_compute_reward
import torch
from pathlib import Path

logger = setup_logger("phase2p_grpo_physics", LOG_DIR)

_PHYS_STEPS      = 3 if args.smoke_test else 200
_EFFECTIVE_BATCH = GRPO_BATCH * GRPO_GRAD_ACC

logger.info("=" * 68)
logger.info(f"Phase 2P Physics GRPO  |  Base: qwen35_sft/final")
logger.info(f"  Steps={_PHYS_STEPS} | LR={GRPO_LR} | gen={GRPO_NUM_GEN}")
logger.info(f"  Physics-only subset")
logger.info("=" * 68)

# ── Load model ────────────────────────────────────────────────────────────────
from src.model_loader import load_base_model, apply_lora
from peft import PeftModel

print(f"\n[1/5] Loading Phase 1 checkpoint (physics base): {QWEN35_SFT_FINAL}")
base_model, tokenizer = load_base_model(
    model_name=MODEL_NAME, dtype=torch.bfloat16, drop_vision=True, device_map="auto"
)
if Path(QWEN35_SFT_FINAL).exists():
    model = PeftModel.from_pretrained(base_model, QWEN35_SFT_FINAL, is_trainable=True)
else:
    logger.warning("Phase 1 checkpoint not found — using fresh LoRA")
    model = apply_lora(base_model, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
                       lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS)
model.print_trainable_parameters()
print_vram("after load")

# ── Load + filter physics dataset ─────────────────────────────────────────────
from datasets import load_from_disk

print(f"\n[2/5] Loading physics-only subset")
train_ds = load_from_disk(TRAIN_DS)
val_ds   = load_from_disk(VAL_DS)

train_phys = train_ds.filter(lambda x: x.get("type") == "physics")
val_phys   = val_ds.filter(lambda x: x.get("type") == "physics")
print(f"      Physics train: {len(train_phys):,}  |  val: {len(val_phys):,}")
logger.info(f"Physics subsets: train={len(train_phys)} val={len(val_phys)}")

def format_for_grpo(examples):
    prompts, gts, subjects, questions = [], [], [], []
    for i in range(len(examples["prompt"])):
        sample = {k: examples[k][i] for k in examples}
        try:
            p = tokenizer.apply_chat_template(
                sample["prompt"], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            p = tokenizer.apply_chat_template(
                sample["prompt"], tokenize=False, add_generation_prompt=True)
        prompts.append(p)
        gts.append(str(sample.get("answer", "")))
        subjects.append("physics")
        q = next((m["content"] for m in sample["prompt"]
                   if isinstance(m, dict) and m.get("role") == "user"), "")
        questions.append(q)
    return {"prompt": prompts, "ground_truth": gts, "subject": subjects, "question": questions}

print("\n[3/5] Formatting for GRPO…")
train_grpo = train_phys.map(format_for_grpo, batched=True,
                             remove_columns=train_phys.column_names, desc="phys_train")
val_grpo   = val_phys.map(format_for_grpo, batched=True,
                           remove_columns=val_phys.column_names, desc="phys_val")

# ── GRPOTrainer ───────────────────────────────────────────────────────────────
from trl import GRPOTrainer, GRPOConfig

TB_DIR = Path(LOG_DIR) / "phase2p_tensorboard"
TB_DIR.mkdir(parents=True, exist_ok=True)
Path(QWEN35_GRPO_PHYS_DIR).mkdir(parents=True, exist_ok=True)

grpo_config = GRPOConfig(
    output_dir                   = QWEN35_GRPO_PHYS_DIR,
    per_device_train_batch_size  = GRPO_BATCH,
    gradient_accumulation_steps  = GRPO_GRAD_ACC,
    dataloader_num_workers       = 2,
    max_steps                    = _PHYS_STEPS,
    learning_rate                = GRPO_LR,
    lr_scheduler_type            = "cosine",
    warmup_steps                 = 5 if not args.smoke_test else 0,
    num_generations              = GRPO_NUM_GEN,
    max_completion_length        = GRPO_MAX_COMP,
    beta                         = GRPO_BETA,
    use_vllm                     = False,
    bf16                         = True,
    fp16                         = False,
    optim                        = "adamw_torch",
    gradient_checkpointing       = True,
    logging_steps                = 5,
    logging_dir                  = str(TB_DIR),
    report_to                    = ["tensorboard"],
    save_strategy                = "steps",
    save_steps                   = max(10, _PHYS_STEPS // 4),
    save_total_limit             = 2,
    seed                         = 42,
)

print(f"\n[4/5] Building GRPOTrainer (physics)…")
grpo_trainer = GRPOTrainer(
    model=model, processing_class=tokenizer, args=grpo_config,
    train_dataset=train_grpo, reward_funcs=[batch_compute_reward],
)
print_vram("before training")

# ── Train ──────────────────────────────────────────────────────────────────────
print(f"\n[5/5] GRPO Physics training (steps={_PHYS_STEPS})")
print("=" * 68)
t0    = time.time()
stats = grpo_trainer.train()
elapsed_min = (time.time() - t0) / 60
print_vram("after training")

# Save
Path(QWEN35_GRPO_PHYS_FINAL).mkdir(parents=True, exist_ok=True)
model.save_pretrained(QWEN35_GRPO_PHYS_FINAL)
tokenizer.save_pretrained(QWEN35_GRPO_PHYS_FINAL)

result_info = {
    "phase": "2P_grpo_physics", "model": MODEL_NAME,
    "base_checkpoint": QWEN35_SFT_FINAL, "smoke_test": args.smoke_test,
    "training_loss": round(getattr(stats, "training_loss", 0.0), 6),
    "global_steps": getattr(stats, "global_step", _PHYS_STEPS),
    "elapsed_min": round(elapsed_min, 1),
    "grpo_steps": _PHYS_STEPS, "grpo_lr": GRPO_LR,
    "physics_train_size": len(train_phys),
    "checkpoint": QWEN35_GRPO_PHYS_FINAL,
}
results_path = Path(LOG_DIR) / "phase2p_grpo_physics_results.json"
with open(results_path, "w") as f:
    json.dump(result_info, f, indent=2)

logger.info(f"Phase 2P done | steps={result_info['global_steps']} time={elapsed_min:.1f}min")
print("=" * 68)
print(" PHASE 2P (PHYSICS) GRPO COMPLETE")
print(f"  Steps      : {result_info['global_steps']}")
print(f"  Time       : {elapsed_min:.1f} min")
print(f"  Checkpoint : {QWEN35_GRPO_PHYS_FINAL}")
print(f"  Results    : {results_path}")
print("=" * 68)
