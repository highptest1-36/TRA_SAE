"""
TRA-SAE Configuration
======================
Central config for all paths, model settings, and training hyperparameters.

v2 (2026-05-24):  Migrated to Qwen3.5-4B on A100 40GB.
  - BF16 (no quantization), vanilla transformers + PEFT (no Unsloth)
  - Expanded LoRA alpha, A100-tuned batch sizes
  - Dual LoRA specialist checkpoints (logic / physics)
  - Self-consistency voting + Z3 SMT integration
  - Subject-aware retrieval via subject router
"""
import os

# ── Hardware Profile ──────────────────────────────────────────────────────────
GPU_PROFILE  = "A100"   # "L4" (22.5 GB) | "A100" (40 GB)
VRAM_GB      = 40

# ── Base Paths ────────────────────────────────────────────────────────────────
DRIVE_BASE = "/content/drive/MyDrive/TRA-SAE"
DATA_DIR   = f"{DRIVE_BASE}/data"
SRC_DIR    = f"{DRIVE_BASE}/src"
PROC_DIR   = f"{DRIVE_BASE}/processed_data"
CKPT_DIR   = f"{DRIVE_BASE}/checkpoints"
LOG_DIR    = f"{DRIVE_BASE}/logs"
OUT_DIR    = f"{DRIVE_BASE}/outputs"
0
# ── Data Paths ────────────────────────────────────────────────────────────────
LOGIC_JSON  = (
    f"{DATA_DIR}/Logic_Based_Educational_Queries_Text_Only"
    f"/Logic_Based_Educational_Queries.json"
)
PHYSICS_CSV = (
    f"{DATA_DIR}/Physics_Problems_Text_Only"
    f"/Physics_Problems_Text_Only.csv"
)
TRAIN_DS = f"{PROC_DIR}/exact_train"
VAL_DS   = f"{PROC_DIR}/exact_val"

# ── Model (Qwen3.5-4B, BF16, vanilla transformers + PEFT) ────────────────────
MODEL_NAME      = "Qwen/Qwen3.5-4B"   # VLM base; vision tower will be dropped
MAX_SEQ_LEN     = 2048
ENABLE_THINKING = False   # Always disable Qwen3.5 <think> mode → use <reasoning> tags
LOAD_IN_4BIT    = False   # A100 40GB: full BF16, no quantization needed
DTYPE           = "bfloat16"

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R       = 32
LORA_ALPHA   = 64     # 2× r — wider learning window for new architecture
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Phase 1: SFT (All-subject, A100 40GB) ────────────────────────────────────
SFT_EPOCHS     = 3
SFT_BATCH      = 4     # A100: 4 × BF16 fits comfortably in 40 GB
SFT_GRAD_ACC   = 4     # Effective batch = 16
SFT_LR         = 2e-4
SFT_WARMUP     = 50
SFT_SAVE_STEPS = 100
SFT_MAX_STEPS  = -1    # -1 = use SFT_EPOCHS (don't cap by steps)

# ── Phase 1.5: Logic Curriculum SFT ──────────────────────────────────────────
LOGIC_SFT_EPOCHS   = 2
LOGIC_SFT_LR       = 5e-5   # lower LR to avoid catastrophic forgetting of physics
LOGIC_SFT_GRAD_ACC = 4

# ── Phase 2: GRPO Unit-Aware (A100 40GB) ─────────────────────────────────────
GRPO_BATCH      = 2
GRPO_GRAD_ACC   = 2    # Effective batch = 4
GRPO_LR         = 1e-6
GRPO_MAX_STEPS  = 250
GRPO_SAVE_STEPS = 50
GRPO_NUM_GEN    = 4    # TRL 1.4.0: generation_batch_size(4) must be divisible by num_generations
GRPO_MAX_COMP   = 1024 # A100: 1024 tokens (was 512 on L4)
GRPO_BETA       = 0.04 # KL penalty coefficient

# ── Reward Weights ────────────────────────────────────────────────────────────
REWARD_FORMAT      = 0.30   # Three-tag XML format
REWARD_CORRECTNESS = 0.60   # Symbolic / Z3 correctness
REWARD_UNIT        = 0.10   # Unit-scale bonus
REWARD_LENGTH_PEN  = -0.10  # Penalty if reasoning > 800 tokens
REWARD_LENGTH_MAX  = 800    # tokens before penalty kicks in

# ── Legacy aliases (keep backward-compat with old scripts) ───────────────────
REWARD_P1 = REWARD_CORRECTNESS
REWARD_P2 = 0.5
REWARD_P3 = REWARD_FORMAT

# ── Agent Inference ───────────────────────────────────────────────────────────
MAX_NEW_TOKENS        = 1024   # was 512; extended to prevent cut-off
SELF_CONSISTENCY_N    = 5      # number of samples for majority vote
SELF_CONSISTENCY_TEMP = 0.7
AGENT_MAX_RETRIES     = 2      # max retry attempts per question (was 3)

# ── Z3 SMT Solver ─────────────────────────────────────────────────────────────
Z3_ENABLED        = True   # Enable Z3 for logic Yes/No verification
Z3_TIMEOUT_MS     = 5000   # per-query timeout in milliseconds

# ── Subject Router ────────────────────────────────────────────────────────────
ROUTER_PATH              = f"{CKPT_DIR}/router.pkl"
ROUTER_CONFIDENCE_THRESH = 0.80   # below this → keyword fallback

# ── Checkpoint Paths (Qwen3.5-4B pipeline) ───────────────────────────────────
QWEN35_SFT_DIR        = f"{CKPT_DIR}/qwen35_sft"
QWEN35_SFT_FINAL      = f"{CKPT_DIR}/qwen35_sft/final"
QWEN35_SFT_LOGIC_DIR  = f"{CKPT_DIR}/qwen35_sft_logic"
QWEN35_SFT_LOGIC_FINAL= f"{CKPT_DIR}/qwen35_sft_logic/final"
QWEN35_GRPO_DIR       = f"{CKPT_DIR}/qwen35_grpo"
QWEN35_GRPO_FINAL     = f"{CKPT_DIR}/qwen35_grpo/final"
QWEN35_GRPO_PHYS_DIR  = f"{CKPT_DIR}/qwen35_grpo_physics"
QWEN35_GRPO_PHYS_FINAL= f"{CKPT_DIR}/qwen35_grpo_physics/final"
QWEN35_GRPO_LOGIC_DIR = f"{CKPT_DIR}/qwen35_grpo_logic"
QWEN35_GRPO_LOGIC_FINAL=f"{CKPT_DIR}/qwen35_grpo_logic/final"

# ── Legacy Checkpoint Paths (Qwen3-4B, Unsloth — kept as reference) ──────────
PHASE1_SFT_FINAL  = f"{CKPT_DIR}/phase1_sft_final"
PHASE3_GRPO_FINAL = f"{CKPT_DIR}/phase3_grpo_final"

# ── TF Evaluator (legacy, not used in v2) ─────────────────────────────────────
EVALUATOR_WEIGHTS = f"{CKPT_DIR}/explanation_evaluator.keras"
EVALUATOR_VOCAB   = 32000
EVALUATOR_MAX_LEN = 256


def print_config() -> None:
    print(f"[TRA-SAE Config v2]")
    print(f"  GPU Profile : {GPU_PROFILE} ({VRAM_GB} GB)")
    print(f"  Model       : {MODEL_NAME}")
    print(f"  Thinking    : {ENABLE_THINKING}")
    print(f"  4-bit       : {LOAD_IN_4BIT}")
    print(f"  max_seq_len : {MAX_SEQ_LEN}")
    print(f"  LoRA r      : {LORA_R}")
    print(f"  SFT steps   : {SFT_MAX_STEPS} (eff batch {SFT_BATCH*SFT_GRAD_ACC})")
    print(f"  GRPO steps  : {GRPO_MAX_STEPS} (gen×{GRPO_NUM_GEN})")
    print(f"  max_seq_len : {MAX_SEQ_LEN}  |  lora_r : {LORA_R}")
    print(f"  SFT steps   : {SFT_MAX_STEPS}  |  GRPO steps : {GRPO_MAX_STEPS}")
    print(f"  Checkpoints : {CKPT_DIR}")
