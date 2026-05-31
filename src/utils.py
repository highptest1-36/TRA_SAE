"""
TRA-SAE General Utilities
==========================
Logging, checkpoint management, VRAM monitoring, and submission formatting.
"""
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from datetime import datetime


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(
    name: str = "tra-sae",
    log_dir: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger writing to console and optionally to a dated log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            fh  = logging.FileHandler(f"{log_dir}/{name}_{ts}.log", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


# ── Directory Helpers ─────────────────────────────────────────────────────────

def ensure_dirs(*paths: str) -> None:
    """Create all specified directories (including parents) if missing."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def latest_checkpoint(ckpt_dir: str) -> str | None:
    """Return the path to the most recently modified sub-directory."""
    p = Path(ckpt_dir)
    if not p.exists():
        return None
    dirs = sorted(
        [d for d in p.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return str(dirs[0]) if dirs else None


# ── Results I/O ───────────────────────────────────────────────────────────────

def save_results(results: list[dict], output_path: str) -> None:
    """Save a list of result dicts to a JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[utils] Saved {len(results)} results → {output_path}")


def load_results(path: str) -> list[dict]:
    """Load a JSONL results file."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


# ── VRAM Monitor ─────────────────────────────────────────────────────────────

def print_vram(label: str = "") -> None:
    """Print current GPU VRAM allocation (requires PyTorch + CUDA)."""
    try:
        import torch
        alloc    = torch.cuda.memory_allocated()  / 1e9
        reserved = torch.cuda.memory_reserved()   / 1e9
        total    = torch.cuda.get_device_properties(0).total_memory / 1e9
        tag      = f"[{label}] " if label else ""
        print(f"{tag}VRAM  {alloc:.2f} GB alloc  /  {reserved:.2f} GB reserved  /  {total:.1f} GB total")
    except Exception:
        print("VRAM info not available (no CUDA device?)")


# ── Submission Formatter ──────────────────────────────────────────────────────

def format_submission(
    answer: str,
    explanation: str,
    cot: str = "",
    fol: str = "",
    premises: list[str] | None = None,
    confidence: float | None = None,
) -> dict:
    """Build a competition-compliant submission dict.

    Required fields:  answer, explanation
    Optional fields:  cot, fol, premises, confidence  (improve P3 score)
    """
    out: dict = {"answer": answer, "explanation": explanation}
    if cot:
        out["cot"] = cot
    if fol:
        out["fol"] = fol
    if premises:
        out["premises"] = premises
    if confidence is not None:
        out["confidence"] = round(float(confidence), 4)
    return out
