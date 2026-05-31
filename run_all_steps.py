"""
Master runner — chạy tuần tự step1 → step6 với logging / checkpoint đầy đủ.

Usage:
    python run_all_steps.py                      # full pipeline
    python run_all_steps.py --start-from step2   # bỏ qua step1 đã xong
    python run_all_steps.py --steps step4 step5  # chỉ chạy bước được chọn
    python run_all_steps.py --smoke              # smoke test toàn bộ

Mỗi bước được chạy trong subprocess riêng với:
  - Log riêng:  logs/runner_stepN_YYYYMMDD_HHMMSS.log
  - Status JSON: logs/pipeline_status.json (cập nhật sau mỗi bước)
  - Nếu bước thất bại → ghi lỗi + tiếp tục bước kế (trừ bước blocking)

Blocking dependencies:
    step4 cần step1 (qwen35_ablation_v2_latest.json)
    step6 cần step1 (qwen35_ablation_v2_latest.json)
"""
from __future__ import annotations
import sys, os, subprocess, json, time, argparse, shutil
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
from src.config import LOG_DIR

# ── Config ─────────────────────────────────────────────────────────────────
LOG_DIR_P      = Path(LOG_DIR)
STATUS_FILE    = LOG_DIR_P / "pipeline_status.json"
PYTHON         = sys.executable
WD             = "/content/drive/MyDrive/TRA-SAE"

ALL_STEPS = ["step1", "step2", "step3", "step4", "step5", "step6"]

STEP_SCRIPTS = {
    "step1": "experiments/step1_rerun_cfg0_3.py",
    "step2": "experiments/step2_multiseed_cfg3.py",
    "step3": "experiments/step3_baselines.py",
    "step4": "experiments/step4_stats_and_errors.py",
    "step5": "experiments/step5_reward_ablation.py",
    "step6": "experiments/step6_latency.py",
}

# Bước nào BLOCKING: nếu dep chưa có file output thì skip với cảnh báo
BLOCKING_DEPS = {
    "step4": LOG_DIR_P / "qwen35_ablation_v2_latest.json",
    "step6": LOG_DIR_P / "qwen35_ablation_v2_latest.json",
    "step2": Path("/content/drive/MyDrive/TRA-SAE/checkpoints/qwen35_grpo/final"),
}

# step3 cần không gì cả (có fallback), step5 độc lập

# ── Helpers ────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_status() -> dict:
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"pipeline_ts": _ts(), "steps": {}}


def _save_status(status: dict) -> None:
    LOG_DIR_P.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    tmp.replace(STATUS_FILE)


def _print(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _run_step(step: str, extra_args: list[str], log_file: Path) -> tuple[int, float]:
    """Run một step, stream output ra log_file + stdout. Trả về (returncode, elapsed_sec)."""
    script = STEP_SCRIPTS[step]
    cmd    = [PYTHON, "-u", script] + extra_args
    _print(f">>> {step}: {' '.join(cmd)}")
    _print(f"    Log: {log_file}")

    LOG_DIR_P.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with open(log_file, "w", buffering=1) as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=WD,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            lf.write(line)
        proc.wait()

    elapsed = round(time.time() - t0, 1)
    return proc.returncode, elapsed


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TRA-SAE pipeline runner")
    parser.add_argument("--start-from", metavar="STEP",
                        help="Bắt đầu từ bước này (bỏ qua các bước trước)")
    parser.add_argument("--steps", nargs="+", metavar="STEP",
                        help="Chỉ chạy các bước được liệt kê")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: thêm --smoke-steps 10 --smoke-eval 8")
    parser.add_argument("--force", action="store_true",
                        help="Bỏ qua status completed trong pipeline_status.json")
    args = parser.parse_args()

    # Xác định danh sách bước cần chạy
    if args.steps:
        steps_to_run = [s for s in ALL_STEPS if s in args.steps]
    elif args.start_from:
        try:
            idx = ALL_STEPS.index(args.start_from)
            steps_to_run = ALL_STEPS[idx:]
        except ValueError:
            _print(f"ERROR: không có step '{args.start_from}'. Choices: {ALL_STEPS}")
            sys.exit(1)
    else:
        steps_to_run = list(ALL_STEPS)

    smoke_args = ["--smoke-steps", "10", "--smoke-eval", "8"] if args.smoke else []

    status = _load_status()
    status.setdefault("steps", {})
    _print(f"Pipeline start — steps: {steps_to_run}  smoke={args.smoke}")
    _print(f"Status file: {STATUS_FILE}")

    for step in steps_to_run:
        # Kiểm tra đã done chưa (trừ khi --force)
        if not args.force:
            prev = status["steps"].get(step, {})
            if prev.get("status") == "completed":
                _print(f"--- {step}: SKIPPED (already completed at {prev.get('finished_at')})")
                continue

        # Kiểm tra blocking deps
        dep_file = BLOCKING_DEPS.get(step)
        if dep_file is not None and not Path(dep_file).exists():
            _print(f"!!! {step}: SKIPPED — blocking dependency missing: {dep_file}")
            status["steps"][step] = {
                "status": "skipped",
                "reason": f"dep missing: {dep_file}",
                "ts":     _ts(),
            }
            _save_status(status)
            continue

        # Log file riêng cho bước này
        log_file = LOG_DIR_P / f"runner_{step}_{_ts()}.log"

        # Extra args tùy step
        extra = list(smoke_args)

        # Ghi started
        status["steps"][step] = {
            "status":     "running",
            "started_at": _ts(),
            "log":        str(log_file),
        }
        _save_status(status)

        # Chạy
        rc, elapsed = _run_step(step, extra, log_file)

        # Cập nhật status
        if rc == 0:
            status["steps"][step].update({
                "status":      "completed",
                "finished_at": _ts(),
                "elapsed_sec": elapsed,
                "returncode":  0,
            })
            _print(f"+++ {step}: COMPLETED in {elapsed}s")
        else:
            status["steps"][step].update({
                "status":      "failed",
                "finished_at": _ts(),
                "elapsed_sec": elapsed,
                "returncode":  rc,
            })
            _print(f"!!! {step}: FAILED (rc={rc}) after {elapsed}s — see {log_file}")

        _save_status(status)

    # ── Summary ────────────────────────────────────────────────────────────
    _print("\n" + "="*60)
    _print("PIPELINE SUMMARY")
    _print("="*60)
    for step in steps_to_run:
        info = status["steps"].get(step, {})
        st   = info.get("status", "unknown")
        el   = info.get("elapsed_sec", "?")
        _print(f"  {step:8s}  {st:10s}  elapsed={el}s")
    _print("="*60)
    _print(f"Full status: {STATUS_FILE}")

    # Exit non-zero nếu có bước failed
    failed = [s for s in steps_to_run
              if status["steps"].get(s, {}).get("status") == "failed"]
    if failed:
        _print(f"FAILED steps: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
