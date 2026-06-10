"""
Assemble the EXACT 2026 submission package.

Produces under API_TRA_SAE/submission/:
  source_code.zip      — reproducible source (src/, API_TRA_SAE/, experiments/, run_*, reqs)
  solution.pdf         — copied from API_TRA_SAE/solution.pdf
  urls.txt             — copied from API_TRA_SAE/urls.txt
  notation_mapping.csv — copied from API_TRA_SAE/notation_mapping.csv
  TRA_SAE_EXACT2026_package.zip — bundle of the four files above (Option A, <=4MB)

Excludes checkpoints/, processed_data/, logs/, paper/, *.zip, caches, .git.
Pure file operations — no GPU, touches nothing outside API_TRA_SAE/submission/.
"""
from __future__ import annotations
import os, shutil, sys, zipfile, fnmatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVE = os.path.join(ROOT, "serve")
OUT = os.path.join(SERVE, "submission")

# BTC requires the package named <team_name>.zip. Set via arg or SERVE_TEAM_NAME.
TEAM = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SERVE_TEAM_NAME", "TRA_SAE")).strip()
TEAM = "".join(c for c in TEAM if c.isalnum() or c in "._-") or "TRA_SAE"

# What goes into source_code.zip (relative to repo root).
SRC_DIRS = ["src", "serve", "experiments"]
SRC_FILES = ["README.md", "run_all_steps.py",
             "run_phase1_sft.py", "run_phase1_5_logic_sft.py",
             "run_phase2_grpo.py", "run_phase2_grpo_physics.py",
             "run_phase2_grpo_logic.py", "run_phase4_v2_agent.py"]
EXCLUDE = ["*/__pycache__/*", "*.pyc", "API_TRA_SAE/submission/*", "API_TRA_SAE/logs/*",
           "*/.ipynb_checkpoints/*", "*.zip", "*.pdf~"]


def _excluded(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, pat) for pat in EXCLUDE)


def build_source_zip(dst: str) -> None:
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for d in SRC_DIRS:
            base = os.path.join(ROOT, d)
            for dirpath, _, files in os.walk(base):
                for fn in files:
                    full = os.path.join(dirpath, fn)
                    rel = os.path.relpath(full, ROOT)
                    if _excluded(rel):
                        continue
                    z.write(full, rel)
        for f in SRC_FILES:
            full = os.path.join(ROOT, f)
            if os.path.isfile(full):
                z.write(full, f)


def main() -> None:
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    src_zip = os.path.join(OUT, "source_code.zip")
    build_source_zip(src_zip)

    for name in ["solution.pdf", "urls.txt", "notation_mapping.csv"]:
        shutil.copy2(os.path.join(SERVE, name), os.path.join(OUT, name))

    pkg = os.path.join(OUT, f"{TEAM}.zip")
    with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ["solution.pdf", "source_code.zip", "urls.txt", "notation_mapping.csv"]:
            z.write(os.path.join(OUT, name), name)

    print(f"Submission package built in {OUT}  (team={TEAM})")
    for name in sorted(os.listdir(OUT)):
        size = os.path.getsize(os.path.join(OUT, name))
        print(f"  {name:<34} {size/1024:8.1f} KB")
    pkg_mb = os.path.getsize(pkg) / 1024 / 1024
    print(f"\nPackage ZIP = {pkg_mb:.2f} MB  ({'OK <=4MB (Option A direct)' if pkg_mb <= 4 else 'USE Option C (Google Drive)'})")
    # quick format sanity
    with zipfile.ZipFile(pkg) as z:
        names = z.namelist()
    has_pdf = any(n.endswith('.pdf') for n in names)
    has_zip = any(n.endswith('.zip') for n in names)
    has_txt = any(n.endswith('.txt') for n in names)
    has_csv = any(n.endswith('.csv') for n in names)
    print(f"Format check  pdf={has_pdf} zip={has_zip} txt={has_txt} csv={has_csv}  → "
          + ("ALL PRESENT" if all([has_pdf, has_zip, has_txt, has_csv]) else "MISSING A FORMAT"))


if __name__ == "__main__":
    main()
