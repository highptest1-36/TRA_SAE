"""
TRA-SAE Phase 4 — LangGraph Agent Evaluation
=============================================
Loads the Phase 3 GRPO model, builds the retrieval index, assembles the
LangGraph StateGraph, and runs a full evaluation pass on the 217-sample
validation set.

Usage (Google Colab / terminal):
    python run_phase4_agent.py

Outputs:
    logs/phase4_results.jsonl   — per-sample results
    logs/phase4_summary.json    — aggregate metrics
    checkpoints/phase4_agent/   — (reserved for future fine-tuning)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE/src")

DRIVE_BASE    = "/content/drive/MyDrive/TRA-SAE"
PHASE3_MODEL  = f"{DRIVE_BASE}/checkpoints/phase3_grpo_final"
BASE_MODEL    = "unsloth/qwen3-4b-unsloth-bnb-4bit"
TRAIN_DS      = f"{DRIVE_BASE}/processed_data/exact_train"
VAL_DS        = f"{DRIVE_BASE}/processed_data/exact_val"
LOG_DIR       = f"{DRIVE_BASE}/logs"
RETRIEVER_CACHE = f"{LOG_DIR}/retriever_cache.pkl"
RESULTS_JSONL   = f"{LOG_DIR}/phase4_results.jsonl"
SUMMARY_JSON    = f"{LOG_DIR}/phase4_summary.json"
MAX_SEQ_LEN   = 2048

# ── Logging ───────────────────────────────────────────────────────────────────
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"{LOG_DIR}/phase4_agent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("tra-sae.phase4")


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Load model
# ──────────────────────────────────────────────────────────────────────────────

def load_model():
    """Load base model with unsloth, then attach Phase 3 LoRA adapter."""
    import torch
    from unsloth import FastLanguageModel
    from peft import PeftModel

    logger.info("Loading base model: %s", BASE_MODEL)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        fast_inference=False,
    )
    logger.info("Applying Phase 3 LoRA adapter from: %s", PHASE3_MODEL)
    model = PeftModel.from_pretrained(model, PHASE3_MODEL)

    FastLanguageModel.for_inference(model)
    model.eval()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    logger.info("Model ready on %s", device)
    return model, tokenizer, device


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Build retriever
# ──────────────────────────────────────────────────────────────────────────────

def build_retriever():
    from src.retriever import Retriever
    r = Retriever(TRAIN_DS)
    r.build(cache_path=RETRIEVER_CACHE)
    return r


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Run evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(model, tokenizer, device, retriever, agent_graph):
    """Run the agent on all 217 validation samples and log results."""
    from datasets import load_from_disk
    from src.symbolic_verifier import verify_answer

    val_ds = load_from_disk(VAL_DS)
    logger.info("Validation set: %d samples", len(val_ds))

    results        = []
    n_correct      = 0
    n_verified     = 0
    n_format_ok    = 0
    total_retries  = 0
    total_conf     = 0.0
    total_time     = 0.0

    Path(RESULTS_JSONL).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(RESULTS_JSONL, "w", encoding="utf-8")

    for idx, sample in enumerate(val_ds):
        # Extract question text from chat-formatted prompt
        question = ""
        for msg in sample["prompt"]:
            if msg["role"] == "user":
                question = msg["content"]
                break

        ground_truth = sample["answer"]
        subject      = sample.get("type", "unknown")

        logger.info(
            "[%d/%d] %s | gt='%s'",
            idx + 1, len(val_ds), subject, ground_truth[:30],
        )

        # Build initial state for this sample
        initial_state = {
            "question":           question,
            "subject":            subject,
            "ground_truth":       ground_truth,
            "_model":             model,
            "_tokenizer":         tokenizer,
            "_retriever":         retriever,
            "_device":            device,
            "retrieved_examples": [],
            "raw_output":         "",
            "all_attempts":       [],
            "generated_answer":   "",
            "verified":           False,
            "retry_count":        0,
            "explanation":        "",
            "confidence":         0.0,
            "final_output":       "",
        }

        t0 = time.time()
        try:
            final_state = agent_graph.invoke(initial_state)
        except Exception as exc:
            logger.error("[%d/%d] Graph error: %s", idx + 1, len(val_ds), exc)
            final_state = dict(initial_state)
            final_state["final_output"] = "<answer>\nunknown\n</answer>\n<explanation>\nError during inference.\n</explanation>"
            final_state["generated_answer"] = "unknown"
            final_state["confidence"] = 0.0
            final_state["retry_count"] = 0

        elapsed = time.time() - t0

        # Post-hoc correctness check (in case verifier used format-only mode)
        pred_answer = final_state.get("generated_answer", "")
        is_correct  = verify_answer(pred_answer, ground_truth)
        retries     = max(0, final_state.get("retry_count", 1) - 1)
        conf        = final_state.get("confidence", 0.0)
        fmt_ok      = bool(
            "<answer>" in final_state.get("final_output", "")
            and "<explanation>" in final_state.get("final_output", "")
        )

        # Track metrics
        if is_correct:  n_correct  += 1
        if final_state.get("verified", False): n_verified += 1
        if fmt_ok:      n_format_ok += 1
        total_retries += retries
        total_conf    += conf
        total_time    += elapsed

        record = {
            "idx":            idx,
            "subject":        subject,
            "question":       question[:200],
            "ground_truth":   ground_truth,
            "predicted":      pred_answer,
            "correct":        is_correct,
            "verified":       final_state.get("verified", False),
            "format_ok":      fmt_ok,
            "retries":        retries,
            "confidence":     round(conf, 4),
            "elapsed_s":      round(elapsed, 2),
            "final_output":   final_state.get("final_output", ""),
        }
        results.append(record)
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()

        logger.info(
            "  → correct=%s  fmt=%s  retries=%d  conf=%.2f  %.1fs",
            is_correct, fmt_ok, retries, conf, elapsed,
        )

    out_f.close()

    n = len(val_ds)
    summary = {
        "phase":                "4_agent",
        "model":                PHASE3_MODEL,
        "val_samples":          n,
        "accuracy":             round(n_correct / n, 4),
        "agent_verified_rate":  round(n_verified / n, 4),
        "format_ok_rate":       round(n_format_ok / n, 4),
        "avg_retries":          round(total_retries / n, 3),
        "avg_confidence":       round(total_conf / n, 4),
        "total_time_min":       round(total_time / 60, 2),
        "avg_time_per_sample_s": round(total_time / n, 2),
        "results_file":         RESULTS_JSONL,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("TRA-SAE Phase 4 — LangGraph Agent Evaluation")
    logger.info("=" * 60)

    # 1. Load Phase 3 model
    model, tokenizer, device = load_model()

    # 2. Build retrieval index
    logger.info("Building TF-IDF retrieval index …")
    retriever = build_retriever()

    # 3. Compile agent graph
    logger.info("Compiling LangGraph StateGraph …")
    from src.agent_graph import build_agent_graph
    agent_graph = build_agent_graph()
    logger.info("Graph compiled ✓")

    # 4. Run evaluation
    logger.info("Starting evaluation on validation set …")
    summary = run_evaluation(model, tokenizer, device, retriever, agent_graph)

    # 5. Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase 4 Evaluation Summary")
    logger.info("=" * 60)
    logger.info("  Val samples         : %d",  summary["val_samples"])
    logger.info("  Accuracy (P1)       : %.2f%%", summary["accuracy"] * 100)
    logger.info("  Agent-verified rate : %.2f%%", summary["agent_verified_rate"] * 100)
    logger.info("  Format OK rate      : %.2f%%", summary["format_ok_rate"] * 100)
    logger.info("  Avg retries         : %.3f",   summary["avg_retries"])
    logger.info("  Avg confidence      : %.3f",   summary["avg_confidence"])
    logger.info("  Total time          : %.1f min", summary["total_time_min"])
    logger.info("  Results → %s", RESULTS_JSONL)
    logger.info("  Summary → %s", SUMMARY_JSON)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
