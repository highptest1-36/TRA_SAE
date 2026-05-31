"""
BƯỚC 6: Latency và compute cost profiling
==========================================
Đo throughput, VRAM peak, tổng GPU-hours, và ước tính chi phí USD.

Chạy:
    python experiments/step6_latency.py
    python experiments/step6_latency.py --gpu-cost-per-hr 3.67
"""
from __future__ import annotations
import sys, os, json, time, argparse, gc
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-cost-per-hr", type=float, default=3.67,
                    help="USD per GPU-hour (Lambda A100 SXM4-80GB default=$3.67)")
args, _ = parser.parse_known_args()

import torch
from pathlib import Path
from src.config import (
    MODEL_NAME, QWEN35_SFT_FINAL, QWEN35_GRPO_FINAL, VAL_DS, LOG_DIR,
    MAX_NEW_TOKENS,
)
from src.utils import setup_logger, print_vram

from datetime import datetime
_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

GPU_COST_HR    = args.gpu_cost_per_hr
LOG_OUT        = Path(LOG_DIR) / f"compute_profile_{_RUN_TS}.json"
LOG_OUT_LATEST = Path(LOG_DIR) / "compute_profile_latest.json"
N_WARMUP    = 3
N_PROFILE   = 20  # samples for throughput measurement
BATCH_SIZES = [1, 4, 8]

SYSTEM_PROMPT = (
    "You are an expert in Logic and Physics. "
    "Think step by step and respond in the exact format:\n"
    "<reasoning>\n[Your step-by-step reasoning]\n</reasoning>\n"
    "<answer>\n[Final answer: letter / Yes/No/Unknown / number+unit]\n</answer>\n"
    "<explanation>\n[Concise explanation of why this answer is correct]\n</explanation>"
)


def _get_vram_gb():
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / 1024**3, 2)


def _reset_vram_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _throughput(model, tokenizer, texts: list, device: str, batch_size: int) -> dict:
    """Measure throughput = samples/sec over N_PROFILE samples."""
    tokenizer.padding_side = "left"
    enc = tokenizer(texts[:batch_size], return_tensors="pt", truncation=True,
                    max_length=1024, padding=True).to(device)
    input_len = enc["input_ids"].shape[1]

    # Warmup
    for _ in range(N_WARMUP):
        with torch.inference_mode():
            _ = model.generate(
                **enc, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    n_done = 0
    for bs in range(0, N_PROFILE, batch_size):
        sub = texts[bs: bs + batch_size]
        if not sub:
            break
        enc = tokenizer(sub, return_tensors="pt", truncation=True,
                        max_length=1024, padding=True).to(device)
        with torch.inference_mode():
            _ = model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        n_done += len(sub)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {
        "batch_size":   batch_size,
        "n_samples":    n_done,
        "elapsed_sec":  round(elapsed, 3),
        "samples_per_sec": round(n_done / elapsed, 3),
        "sec_per_sample":  round(elapsed / n_done, 3),
        "vram_peak_gb": _get_vram_gb(),
    }


print("[1] Loading base model")
from src.model_loader import load_base_model
from peft import PeftModel
from datasets import load_from_disk

val_ds = load_from_disk(VAL_DS)
sample_questions = []
for s in list(val_ds.select(range(min(N_PROFILE, len(val_ds))))):
    q = s.get("question", "") or (s["prompt"][-1]["content"] if s.get("prompt") else "")
    sample_questions.append(q)

model_results = []

PROFILE_CONFIGS = [
    {"name": "zero_shot",  "ckpt": None},
    {"name": "grpo_mixed", "ckpt": QWEN35_GRPO_FINAL},
]

for cfg in PROFILE_CONFIGS:
    print(f"\n--- Profiling: {cfg['name']} ---")
    base, tokenizer = load_base_model(
        model_name=MODEL_NAME, dtype=torch.bfloat16, drop_vision=True, device_map="auto")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg["ckpt"] and Path(cfg["ckpt"]).exists():
        model = PeftModel.from_pretrained(base, cfg["ckpt"], is_trainable=False)
    else:
        model = base
    model.eval()
    print_vram(f"{cfg['name']} loaded")

    # Build prompt texts
    texts = []
    for q in sample_questions:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": q}]
        try:
            t = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            t = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)
        texts.append(t)
    texts = texts * 3  # ensure enough samples

    cfg_throughputs = []
    for bs in BATCH_SIZES:
        _reset_vram_peak()
        tp = _throughput(model, tokenizer, texts, device, bs)
        cfg_throughputs.append(tp)
        print(f"  batch={bs}: {tp['samples_per_sec']:.2f} s/s, "
              f"VRAM={tp['vram_peak_gb']:.2f} GB")

    model_results.append({
        "config":      cfg["name"],
        "throughputs": cfg_throughputs,
    })

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Load elapsed times from existing logs ────────────────────────────────────
print("\n[2] Reading training elapsed times from logs")
training_times_min = {}

abl_v2 = Path(LOG_DIR) / "qwen35_ablation_v2_latest.json"
abl_v1 = Path(LOG_DIR) / "qwen35_ablation.json"
abl_file = abl_v2 if abl_v2.exists() else abl_v1
if abl_file.exists():
    with open(abl_file) as f:
        abl_data = json.load(f).get("ablation", [])
    for r in abl_data:
        training_times_min[r["config_name"]] = r.get("elapsed_min", 0)

# Known training times from logs (Phase 1, 1.5, 2, 2P, 2L)
KNOWN_TRAIN_TIMES = {
    "phase1_sft":        58.8,
    "phase1_5_logic":    15.7,
    "phase2_grpo_mixed": 79.8,
    "phase2p_grpo_phys": 76.6,
    "phase2l_grpo_logic":62.9,
}

# ── Compute cost analysis ────────────────────────────────────────────────────
total_train_hr = sum(KNOWN_TRAIN_TIMES.values()) / 60
total_eval_hr  = sum(training_times_min.values()) / 60
total_hr       = total_train_hr + total_eval_hr
total_usd      = round(total_hr * GPU_COST_HR, 2)

print(f"\n--- Compute Cost (A100 80GB @ ${GPU_COST_HR}/hr) ---")
print(f"  Total training: {total_train_hr:.2f} h  ({total_train_hr*60:.0f} min)")
print(f"  Total eval:     {total_eval_hr:.2f} h  ({total_eval_hr*60:.0f} min)")
print(f"  Grand total:    {total_hr:.2f} GPU-hours")
print(f"  Estimated cost: ${total_usd}")

# ── Save ─────────────────────────────────────────────────────────────────────
output = {
    "gpu":                  "NVIDIA A100-SXM4-80GB",
    "gpu_cost_per_hr_usd":  GPU_COST_HR,
    "training_times_min":   KNOWN_TRAIN_TIMES,
    "eval_times_min":       training_times_min,
    "total_train_hr":       round(total_train_hr, 2),
    "total_eval_hr":        round(total_eval_hr, 2),
    "total_gpu_hr":         round(total_hr, 2),
    "estimated_cost_usd":   total_usd,
    "throughput_profiles":  model_results,
}
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
with open(LOG_OUT, "w") as f:
    json.dump(output, f, indent=2)
with open(LOG_OUT_LATEST, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved (timestamped): {LOG_OUT}")
print(f"Saved (latest)     : {LOG_OUT_LATEST}")
