"""
TRA-SAE Data Utilities
=======================
Load & preprocess EXACT 2026 competition datasets:
  - Type 1: Logic-Based Educational Queries  (JSON, 411 records, ~800 questions)
  - Type 2: Physics Problems                 (CSV,  1755 rows)
"""
import json
import csv
from pathlib import Path

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert in Logic and Physics. "
    "Think step by step and respond in the exact format:\n"
    "<reasoning>\n[Your step-by-step reasoning]\n</reasoning>\n"
    "<answer>\n[Final answer: letter / Yes/No/Unknown / number+unit]\n</answer>\n"
    "<explanation>\n[Concise explanation of why this answer is correct]\n</explanation>"
)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_logic_data(json_path: str) -> list[dict]:
    """Load and flatten Logic-Based Educational Queries JSON.

    Each record has a shared set of premises and 1–2 (question, answer,
    explanation) tuples.  Returns one sample per question.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    samples = []
    for rec in records:
        premises_nl  = rec.get("premises-NL", [])
        premises_fol = rec.get("premises-FOL", [])
        questions    = rec.get("questions", [])
        answers      = rec.get("answers", [])
        explanations = rec.get("explanation", [""] * len(questions))

        for q, a, e in zip(questions, answers, explanations):
            samples.append({
                "type":         "logic",
                "premises_nl":  premises_nl,
                "premises_fol": premises_fol,
                "question":     q,
                "answer":       str(a),
                "explanation":  str(e),
                "cot":          "",
            })
    return samples


def load_physics_data(csv_path: str) -> list[dict]:
    """Load Physics Problems CSV.

    Concatenates numerical answer with its unit into one answer string.
    Skips rows with empty answers (401 QA-prefixed rows lack ground truth).
    Normalises dash-placeholder units (unit='-') to empty string.
    """
    # Unit values that are just placeholders (no real unit)
    _DUMMY_UNITS = {"-", "--", "n/a", "N/A", "none", "None", ""}

    samples = []
    skipped = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ans = row.get("answer", "").strip()
            if not ans:          # Skip rows with no ground-truth answer
                skipped += 1
                continue

            unit = row.get("unit", "").strip()
            if unit in _DUMMY_UNITS:
                unit = ""        # Treat "-" etc. as no unit → avoid "Yes -"

            full_answer = f"{ans} {unit}".strip() if unit else ans

            samples.append({
                "type":         "physics",
                "id":           row.get("id", ""),
                "premises_nl":  [],
                "premises_fol": [],
                "question":     row.get("question", ""),
                "answer":       full_answer,
                "explanation":  "",
                "cot":          row.get("cot", ""),
            })

    if skipped:
        print(f"   [data_utils] Physics: skipped {skipped} rows with empty answer")
    return samples


# ── Formatter ─────────────────────────────────────────────────────────────────

def format_sample(sample: dict) -> dict:
    """Convert a raw sample into prompt/answer format for Unsloth/GRPO."""
    if sample["type"] == "logic":
        premise_lines = "\n".join(
            f"  {i + 1}. {p}"
            for i, p in enumerate(sample["premises_nl"])
        )
        user_content = f"Premises:\n{premise_lines}\n\nQuestion: {sample['question']}"
    else:
        user_content = f"Question: {sample['question']}"

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "answer":      sample["answer"],
        "cot":         sample["cot"],
        "explanation": sample["explanation"],
        "type":        sample["type"],
    }


def load_and_format_all(logic_path: str, physics_path: str) -> list[dict]:
    """Load both datasets and return all formatted samples."""
    logic   = load_logic_data(logic_path)
    physics = load_physics_data(physics_path)
    return [format_sample(s) for s in (logic + physics)]
