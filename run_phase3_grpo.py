"""
Phase 3 — GRPO Reinforcement Learning Script
=============================================
Run standalone: python run_phase3_grpo.py
Uses Phase 1 SFT checkpoint as base (Phase 2 if available).
Logs to: logs/phase3_grpo.log + logs/phase3_tensorboard/
Checkpoint saved to: checkpoints/phase3_grpo_final/
"""
import sys, os, re, time, json

# ── Critical: set BEFORE importing unsloth so compiled GRPO cache reads correct dtype ──
# GRPOConfig(bf16=True) tells HF Trainer to use bf16, but Unsloth's compiled trainer
# reads ACCELERATE_MIXED_PRECISION at runtime. Set explicitly here to avoid fp16/bf16 mismatch.
os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"

sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE/src")

from pathlib import Path
from src.config import *
from src.utils  import setup_logger, print_vram

logger = setup_logger("phase3_grpo", LOG_DIR)

logger.info("=" * 60)
logger.info(f"Phase 3 GRPO start")
logger.info(f"  LoRA r={LORA_R} alpha={LORA_ALPHA} | max_seq={MAX_SEQ_LEN}")
logger.info(f"  Steps={GRPO_MAX_STEPS} | LR={GRPO_LR} | num_gen={GRPO_NUM_GEN}")
logger.info(f"  Reward weights: P1={REWARD_P1} P2={REWARD_P2} P3={REWARD_P3}")
logger.info("=" * 60)

# ── 1. Resolve base checkpoint ─────────────────────────────────────────────────
PHASE2_PATH = f"{CKPT_DIR}/phase2_sasft_final"
PHASE1_PATH = f"{CKPT_DIR}/phase1_sft_final"

if os.path.isdir(PHASE2_PATH):
    GRPO_BASE = PHASE2_PATH
    print(f"\n✅ Phase 2 checkpoint found → using {GRPO_BASE}")
    logger.info(f"GRPO base: Phase 2 checkpoint")
elif os.path.isdir(PHASE1_PATH):
    GRPO_BASE = PHASE1_PATH
    print(f"\n⚠️  Phase 2 not found → falling back to Phase 1: {GRPO_BASE}")
    logger.info(f"GRPO base: Phase 1 checkpoint (Phase 2 not found)")
else:
    raise FileNotFoundError(
        f"No SFT checkpoint found! Expected:\n  {PHASE2_PATH}\n  {PHASE1_PATH}\n"
        f"Run Phase 1 first: python run_phase1_sft.py"
    )

# ── 2. Load BASE model fresh (not PEFT checkpoint) ────────────────────────────
# Loading via PEFT checkpoint uses unsloth/qwen3-4b-unsloth-bnb-4bit which has
# compute_dtype=float16 hardcoded → Half/BFloat16 mismatch in matmul_lora.
# Loading MODEL_NAME directly with dtype=bfloat16 gives compute_dtype=bfloat16.
print(f"\n📦 Loading base model: {MODEL_NAME}  (compute_dtype will be bfloat16)")
print(f"   Phase 1 LoRA weights will be warm-started from: {GRPO_BASE}")
from unsloth import FastLanguageModel
import torch

# ── Safety net: if dynamo recompilation fails for a new shape, fall back to eager ──
# Prevents TorchRuntimeError('Size does not match at dimension 0') from gather
# inside chunked_hidden_states_selective_log_softmax on shape-changing batches.
torch._dynamo.config.suppress_errors = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,       # "unsloth/Qwen3-4B" — NOT the PEFT checkpoint
    max_seq_length = MAX_SEQ_LEN,
    dtype          = torch.bfloat16,   # → BnB compute_dtype = bfloat16
    load_in_4bit   = LOAD_IN_4BIT,
    fast_inference = False,
)
print_vram("after base model load")
logger.info("Base model loaded")

# ── 3. Apply LoRA adapters + warm-start from Phase 1 SFT ──────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r                          = LORA_R,
    lora_alpha                 = LORA_ALPHA,
    lora_dropout               = LORA_DROPOUT,
    target_modules             = LORA_TARGETS,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = 42,
)
print(f"\n✅ LoRA r={LORA_R} alpha={LORA_ALPHA}")
model.print_trainable_parameters()
print_vram("after LoRA")

# Warm-start: load Phase 1 SFT LoRA weights into the fresh LoRA adapters
_adapter_path = f"{GRPO_BASE}/adapter_model.safetensors"
if os.path.exists(_adapter_path):
    import safetensors.torch as _sft
    from peft import set_peft_model_state_dict
    _phase1_w = _sft.load_file(_adapter_path)
    _n_keys = len(_phase1_w)
    # Cast to bfloat16 to match training dtype
    _phase1_w_bf16 = {k: v.to(torch.bfloat16) for k, v in _phase1_w.items()}
    try:
        # PEFT API correctly handles key-prefix mapping (strips base_model.model. prefix)
        _res = set_peft_model_state_dict(model, _phase1_w_bf16, adapter_name="default")
        _n_miss = len(_res.missing_keys) if hasattr(_res, "missing_keys") else "?"
        _n_unex = len(_res.unexpected_keys) if hasattr(_res, "unexpected_keys") else "?"
        _n_loaded = _n_keys - (int(_n_unex) if isinstance(_n_unex, int) else 0)
        print(f"✅ Phase 1 SFT warm-start via PEFT API: {_n_loaded}/{_n_keys} loaded "
              f"(missing={_n_miss}, unexpected={_n_unex})")
        logger.info(f"Phase 1 LoRA warm-start PEFT: loaded={_n_loaded}/{_n_keys}")
    except Exception as _e:
        logger.warning(f"PEFT warm-start failed ({_e}), falling back to manual load")
        _missing, _unexpected = model.load_state_dict(_phase1_w_bf16, strict=False)
        _n_loaded = _n_keys - len(_unexpected)   # keys not found in model = failed to load
        print(f"⚠️  Warm-start fallback: {_n_loaded}/{_n_keys} LoRA params loaded")
        logger.info(f"Phase 1 LoRA warm-start manual: loaded={_n_loaded}/{_n_keys}")
else:
    print(f"⚠️  adapter_model.safetensors not found at {GRPO_BASE} — starting from scratch")
    logger.warning("Phase 1 warm-start skipped (no safetensors found)")

logger.info("LoRA applied")

# ── 4. Load dataset ────────────────────────────────────────────────────────────
from datasets import load_from_disk

train_ds = load_from_disk(TRAIN_DS)
print(f"\n📂 Train: {len(train_ds):,} samples")
logger.info(f"Dataset loaded — train={len(train_ds)}")

# ── 5. Reward functions ────────────────────────────────────────────────────────
from src.symbolic_verifier      import verify_answer, extract_answer_from_text, extract_explanation
from src.reward_evaluator_keras import get_explanation_score

def _text(comp) -> str:
    """Extract plain string from completion (handles list-of-dicts or plain str)."""
    if isinstance(comp, list) and comp:
        return comp[0].get("content", "") if isinstance(comp[0], dict) else str(comp[0])
    return str(comp)

def format_reward(completions, **kwargs) -> list[float]:
    """P3: XML tag format compliance reward (max = REWARD_P3)."""
    rewards = []
    for comp in completions:
        t = _text(comp)
        score = REWARD_P3 * (
              bool(re.search(r"<reasoning>",   t, re.I)) * 0.3
            + bool(re.search(r"<answer>",      t, re.I)) * 0.5
            + bool(re.search(r"<explanation>", t, re.I)) * 0.2
        )
        rewards.append(score)
    return rewards

def correctness_reward(completions, answer, **kwargs) -> list[float]:
    """P1: Symbolic correctness reward (REWARD_P1 if correct, 0 otherwise)."""
    rewards = []
    for comp, gt in zip(completions, answer):
        t         = _text(comp)
        predicted = extract_answer_from_text(t) if "<answer>" in t.lower() else t
        rewards.append(REWARD_P1 if verify_answer(predicted, gt) else 0.0)
    return rewards

def explanation_reward(completions, **kwargs) -> list[float]:
    """P2: TF Bi-LSTM explanation quality score (max = REWARD_P2)."""
    w = EVALUATOR_WEIGHTS if os.path.exists(EVALUATOR_WEIGHTS) else None
    rewards = []
    for comp in completions:
        expl  = extract_explanation(_text(comp))
        score = get_explanation_score(expl, weights_path=w) if expl else 0.0
        rewards.append(float(score) * REWARD_P2)
    return rewards

print("\n✅ Reward functions defined:")
print(f"   P1 correctness  weight={REWARD_P1}  (symbolic verifier)")
print(f"   P2 explanation  weight={REWARD_P2}  (TF Bi-LSTM evaluator)")
print(f"   P3 format       weight={REWARD_P3}  (XML tag regex)")

# ── 6. GRPO Config ─────────────────────────────────────────────────────────────
from trl import GRPOTrainer, GRPOConfig

TB_LOG_DIR = f"{LOG_DIR}/phase3_tensorboard"
Path(TB_LOG_DIR).mkdir(parents=True, exist_ok=True)

grpo_config = GRPOConfig(
    output_dir                  = f"{CKPT_DIR}/phase3_grpo",
    per_device_train_batch_size = GRPO_BATCH,        # 1
    gradient_accumulation_steps = GRPO_GRAD_ACC,     # 8
    num_generations             = GRPO_NUM_GEN,      # 4 (L4)
    max_prompt_length           = 1400,  # prompts can be ~1000 tokens after template expansion
    max_completion_length       = GRPO_MAX_COMP,     # 512
    max_steps                   = GRPO_MAX_STEPS,    # 200
    learning_rate               = GRPO_LR,           # 5e-6
    bf16                        = True,
    fp16                        = False,
    optim                       = "adamw_8bit",
    logging_steps               = 5,
    logging_dir                 = TB_LOG_DIR,
    save_strategy               = "steps",
    save_steps                  = GRPO_SAVE_STEPS,   # 50
    save_total_limit            = 3,
    report_to                   = ["tensorboard"],
    temperature                 = 0.7,
    use_vllm                    = False,             # vLLM not available in Colab
    dataloader_num_workers      = 0,
)

grpo_trainer = GRPOTrainer(
    model            = model,
    processing_class = tokenizer,
    reward_funcs     = [format_reward, correctness_reward, explanation_reward],
    args             = grpo_config,
    train_dataset    = train_ds,
)

print(f"\n{'━'*55}")
print(f" 🚀 Phase 3 GRPO starting...")
print(f"    Base model   : {GRPO_BASE}")
print(f"    Steps        : {GRPO_MAX_STEPS}")
print(f"    Generations  : {GRPO_NUM_GEN} per prompt")
print(f"    Max comp len : {GRPO_MAX_COMP} tokens")
print(f"    LR           : {GRPO_LR}")
print(f"    Checkpoint every {GRPO_SAVE_STEPS} steps → {CKPT_DIR}/phase3_grpo/")
print(f"    TB logs      : {TB_LOG_DIR}")
print(f"{'━'*55}\n")
logger.info("GRPO training started")

# ── 7. Train ───────────────────────────────────────────────────────────────────
# ── Auto-resume from latest checkpoint if available (saves steps already done) ──
_ckpt_resume = None
_ckpt_dir = f"{CKPT_DIR}/phase3_grpo"
if os.path.isdir(_ckpt_dir):
    _ckpts = sorted(
        [d for d in os.listdir(_ckpt_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1])
    )
    if _ckpts:
        _ckpt_resume = f"{_ckpt_dir}/{_ckpts[-1]}"
        print(f"♻️  Resuming from checkpoint: {_ckpt_resume}")
        logger.info(f"Resume from checkpoint: {_ckpt_resume}")

t0    = time.time()
stats = grpo_trainer.train(resume_from_checkpoint=_ckpt_resume)
elapsed = (time.time() - t0) / 60

# ── 8. Save final checkpoint ───────────────────────────────────────────────────
PHASE3_FINAL = f"{CKPT_DIR}/phase3_grpo_final"
model.save_pretrained(PHASE3_FINAL)
tokenizer.save_pretrained(PHASE3_FINAL)
logger.info(f"Saved Phase 3 model → {PHASE3_FINAL}")

# ── 9. Save results JSON ───────────────────────────────────────────────────────
try:
    training_loss = stats.training_loss
    global_steps  = stats.global_step
except Exception:
    training_loss = -1.0
    global_steps  = GRPO_MAX_STEPS

result_info = {
    "phase":          "3_grpo",
    "base_model":     GRPO_BASE,
    "training_loss":  training_loss,
    "global_steps":   global_steps,
    "elapsed_min":    round(elapsed, 1),
    "num_generations": GRPO_NUM_GEN,
    "max_completion": GRPO_MAX_COMP,
    "lora_r":         LORA_R,
    "reward_weights": {"P1": REWARD_P1, "P2": REWARD_P2, "P3": REWARD_P3},
    "checkpoint":     PHASE3_FINAL,
}
results_path = f"{LOG_DIR}/phase3_grpo_results.json"
with open(results_path, "w") as f:
    json.dump(result_info, f, indent=2)

logger.info(
    f"Phase 3 done | loss={training_loss:.4f} "
    f"steps={global_steps} time={elapsed:.1f}min"
)

print(f"\n{'━'*55}")
print(f" ✅ Phase 3 GRPO COMPLETE!")
print(f"    Training loss : {training_loss:.4f}")
print(f"    Global steps  : {global_steps}")
print(f"    Time elapsed  : {elapsed:.1f} min")
print(f"    Checkpoint    : {PHASE3_FINAL}")
print(f"    Results JSON  : {results_path}")
print_vram("Phase 3 final")
print(f"{'━'*55}")
