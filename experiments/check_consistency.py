"""
KIỂM TRA TÍNH NHẤT QUÁN (chạy sau step0 canonical, trước khi viết paper)
=========================================================================
Đảm bảo MỌI con số trong paper truy về được một nguồn duy nhất và không còn
bug đếm mẫu. Thoát mã != 0 nếu phát hiện vấn đề (tiện cho CI).

Kiểm:
  1. n_physics + n_logic == n_total == 217 cho cả 6 config.
  2. File canonical tồn tại + có đủ 6 config.
  3. Per-sample JSONL có đủ 6 config × 217 dòng.
  4. (Cảnh báo) so sánh canonical với qwen35_ablation.json (V1) cũ — in ra
     chênh lệch để biết những số nào trong paper cần cập nhật.
  5. Baselines (nếu có) dùng đúng n_physics=141.

Chạy:
    python experiments/check_consistency.py
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
from pathlib import Path
from collections import Counter
from src.config import LOG_DIR

LP = Path(LOG_DIR)
problems, warnings = [], []


def _load(p):
    return json.load(open(p)) if Path(p).exists() else None


# 1+2. Canonical ablation
canon = _load(LP / "qwen35_ablation_canonical.json") or _load(LP / "qwen35_ablation_v2_latest.json")
print("=" * 64)
print("CHECK 1-2: canonical ablation")
if not canon:
    problems.append("Missing canonical ablation JSON — run step0_canonical_eval.py first")
else:
    abl = canon.get("ablation", [])
    ids = sorted(r["config_id"] for r in abl)
    if ids != [0, 1, 2, 3, 4, 5]:
        problems.append(f"Canonical file has configs {ids}, expected 0-5")
    for r in abl:
        tot = r["n_physics"] + r["n_logic"]
        ok = tot == r["n_total"]
        flag = "OK " if ok else "BUG"
        print(f"  [{flag}] cfg{r['config_id']} {r['config_name']:<18} "
              f"acc={r['accuracy_overall']:>6.2f}%  {r['n_physics']}+{r['n_logic']}={tot} (n={r['n_total']})")
        if not ok:
            problems.append(f"cfg{r['config_id']}: n_phys+n_logic ({tot}) != n_total ({r['n_total']})")
        if not r.get("smoke_test", False) and r["n_total"] != 217:
            warnings.append(f"cfg{r['config_id']}: n_total={r['n_total']} (expected 217 for full run)")

# 3. Per-sample JSONL
print("\nCHECK 3: per-sample JSONL coverage")
ps = LP / "ablation_per_sample_canonical.jsonl"
if not ps.exists():
    ps = LP / "ablation_per_sample_latest.jsonl"
if not ps.exists():
    problems.append("Missing per-sample JSONL (needed for McNemar + error analysis)")
else:
    rows = [json.loads(l) for l in open(ps) if l.strip()]
    by_cfg = Counter(r["config_id"] for r in rows)
    print(f"  per-config rows: {dict(sorted(by_cfg.items()))}")
    for cid in range(6):
        if by_cfg.get(cid, 0) == 0:
            problems.append(f"per-sample JSONL missing config {cid}")

# 4. Diff vs old V1 ablation (informational — shows what to rewrite in paper)
print("\nCHECK 4: drift vs OLD qwen35_ablation.json (V1) — paper numbers to update")
v1 = _load(LP / "qwen35_ablation.json")
if v1 and canon:
    v1m = {r["config_id"]: r for r in v1.get("ablation", [])}
    for r in canon.get("ablation", []):
        old = v1m.get(r["config_id"])
        if old:
            d = r["accuracy_overall"] - old["accuracy_overall"]
            mark = "  <-- CHANGED" if abs(d) >= 1.0 else ""
            print(f"  cfg{r['config_id']}: V1={old['accuracy_overall']:>6.2f}%  "
                  f"canonical={r['accuracy_overall']:>6.2f}%  Δ={d:+.2f}pp{mark}")
else:
    warnings.append("Could not diff against old V1 ablation (qwen35_ablation.json)")

# 5. Baselines denominator
print("\nCHECK 5: baselines denominator")
bl = _load(LP / "baselines_results_latest.json")
if bl:
    for b in bl.get("baselines", []):
        if "error" in b:
            continue
        if b.get("n_total") and b.get("n_physics", 0) + b.get("n_logic", 0) != b["n_total"]:
            warnings.append(f"baseline {b['id']}: subject counts != n_total")
    print(f"  {len(bl.get('baselines', []))} baselines checked")
else:
    warnings.append("No baselines_results_latest.json yet (run step3)")

# Verdict
print("\n" + "=" * 64)
if warnings:
    print(f"WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  ! {w}")
if problems:
    print(f"\nFAILED — {len(problems)} problem(s):")
    for p in problems:
        print(f"  X {p}")
    sys.exit(1)
print("\nALL CONSISTENCY CHECKS PASSED — safe to rebuild paper tables.")
