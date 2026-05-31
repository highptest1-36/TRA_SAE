"""
Phase 2L — Logic Specialist GRPO Training
==========================================
GRPO fine-tune on logic subset only, starting from Phase 1.5 Logic-SFT.

Purpose  : Train the logic LoRA specialist adapter for the dual-LoRA Agent v2.

Base     : checkpoints/qwen35_sft_logic/final     (Phase 1.5 logic SFT)
Data     : logic subset only (~732 training samples)
Steps    : 150
Reward   : batch_compute_reward  (Z3 hook active for logic)
Output   : checkpoints/qwen35_grpo_logic/final

Run:
    python run_phase2_grpo_logic.py
    python run_phase2_grpo_logic.py --smoke-test
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
    QWEN35_SFT_LOGIC_FINAL, QWEN35_SFT_FINAL,
    QWEN35_GRPO_LOGIC_DIR, QWEN35_GRPO_LOGIC_FINAL,
    LOG_DIR,
)
from src.utils import setup_logger, print_vram
from src.reward import batch_compute_reward
import torch
from pathlib import Path

logger = setup_logger("phase2l_grpo_logic", LOG_DIR)

_LOGIC_STEPS     = 3 if args.smoke_test else 150
_EFFECTIVE_BATCH = GRPO_BATCH * GRPO_GRAD_ACC

logger.info("=" * 68)
logger.info(f"Phase 2L Logic GRPO  |  Base: qwen35_sft_logic/final")
logger.info(f"  Steps={_LOGIC_STEPS} | LR={GRPO_LR} | gen={GRPO_NUM_GEN}")
logger.info(f"  Logic-only subset | Z3 reward hook active")
logger.info("=" * 68)

# ── Load model ────────────────────────────────────────────────────────────────
from src.model_loader import load_base_model, apply_lora
from peft import PeftModel

_base_ckpt = QWEN35_SFT_LOGIC_FINAL if Path(QWEN35_SFT_LOGIC_FINAL).exists() else QWEN35_SFT_FINAL
print(f"\n[1/5] Loading base checkpoint: {_base_ckpt}")
base_model, tokenizer = load_base_model(
    model_name=MODEL_NAME, dtype=torch.bfloat16, drop_vision=True, device_map="auto"
)
if Path(_base_ckpt).exists():
    model = PeftModel.from_pretrained(base_model, _base_ckpt, is_trainable=True)
else:
    logger.warning("No checkpoint found — using fresh LoRA")
    model = apply_lora(base_model, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
                       lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS)
model.print_trainable_parameters()
print_vram("after load")

# ── Load + filter logic dataset ───────────────────────────────────────────────
from datasets import load_from_disk

print(f"\n[2/5] Loading logic-only subset")
train_ds = load_from_disk(TRAIN_DS)
val_ds   = load_from_disk(VAL_DS)

train_logic = train_ds.filter(lambda x: x.get("type") == "logic")
val_logic   = val_ds.filter(lambda x: x.get("type") == "logic")
print(f"      Logic train: {len(train_logic):,}  |  val: {len(val_logic):,}")
logger.info(f"Logic subsets: train={len(train_logic)} val={len(val_logic)}")

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
        subjects.append("logic")
        q = next((m["content"] for m in sample["prompt"]
                   if isinstance(m, dict) and m.get("role") == "user"), "")
        questions.append(q)
    return {"prompt": prompts, "ground_truth": gts, "subject": subjects, "question": questions}

print("\n[3/5] Formatting for GRPO…")
train_grpo = train_logic.map(format_for_grpo, batched=True,
                              remove_columns=train_logic.column_names, desc="logic_train")
val_grpo   = val_logic.map(format_for_grpo, batched=True,
                            remove_columns=val_logic.column_names, desc="logic_val")

# ── GRPOTrainer ───────────────────────────────────────────────────────────────
from trl import GRPOTrainer, GRPOConfig

TB_DIR = Path(LOG_DIR) / "phase2l_tensorboard"
TB_DIR.mkdir(parents=True, exist_ok=True)
Path(QWEN35_GRPO_LOGIC_DIR).mkdir(parents=True, exist_ok=True)

grpo_config = GRPOConfig(
    output_dir                   = QWEN35_GRPO_LOGIC_DIR,
    per_device_train_batch_size  = GRPO_BATCH,
    gradient_accumulation_steps  = GRPO_GRAD_ACC,
    dataloader_num_workers       = 2,
    max_steps                    = _LOGIC_STEPS,
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
    save_steps                   = max(10, _LOGIC_STEPS // 4),
    save_total_limit             = 2,
    seed                         = 42,
)

print(f"\n[4/5] Building GRPOTrainer (logic)…")
grpo_trainer = GRPOTrainer(
    model=model, processing_class=tokenizer, args=grpo_config,
    train_dataset=train_grpo, reward_funcs=[batch_compute_reward],
)
print_vram("before training")

# ── Train ──────────────────────────────────────────────────────────────────────
print(f"\n[5/5] GRPO Logic training (steps={_LOGIC_STEPS})")
print("=" * 68)
t0    = time.time()
stats = grpo_trainer.train()
elapsed_min = (time.time() - t0) / 60
print_vram("after training")

# Save
Path(QWEN35_GRPO_LOGIC_FINAL).mkdir(parents=True, exist_ok=True)
model.save_pretrained(QWEN35_GRPO_LOGIC_FINAL)
tokenizer.save_pretrained(QWEN35_GRPO_LOGIC_FINAL)

result_info = {
    "phase": "2L_grpo_logic", "model": MODEL_NAME,
    "base_checkpoint": _base_ckpt, "smoke_test": args.smoke_test,
    "training_loss": round(getattr(stats, "training_loss", 0.0), 6),
    "global_steps": getattr(stats, "global_step", _LOGIC_STEPS),
    "elapsed_min": round(elapsed_min, 1),
    "grpo_steps": _LOGIC_STEPS, "grpo_lr": GRPO_LR,
    "z3_reward_active": True,
    "logic_train_size": len(train_logic),
    "checkpoint": QWEN35_GRPO_LOGIC_FINAL,
}
results_path = Path(LOG_DIR) / "phase2l_grpo_logic_results.json"
with open(results_path, "w") as f:
    json.dump(result_info, f, indent=2)

logger.info(f"Phase 2L done | steps={result_info['global_steps']} time={elapsed_min:.1f}min")
print("=" * 68)
print(" PHASE 2L (LOGIC) GRPO COMPLETE")
print(f"  Steps      : {result_info['global_steps']}")
print(f"  Time       : {elapsed_min:.1f} min")
print(f"  Checkpoint : {QWEN35_GRPO_LOGIC_FINAL}")
print(f"  Results    : {results_path}")
print(f"  Next step  : python run_phase4_v2_agent.py")
print("=" * 68)
