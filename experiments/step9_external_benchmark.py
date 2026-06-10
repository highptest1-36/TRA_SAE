"""
BƯỚC 9: External Benchmark — MMLU Physics
==========================================
Evaluate cfg0 (zero-shot) và cfg3 (best fine-tuned) trên MMLU Physics.

Datasets (không cần login, available on HuggingFace):
  - cais/mmlu  high_school_physics   : 151 MCQ test samples
  - cais/mmlu  college_physics       : 102 MCQ test samples
  Total: 253 MCQ samples

Câu hỏi khoa học:
  Việc fine-tune trên EXACT 2026 physics (unit/numerical) có generalize sang
  MMLU physics (MCQ, conceptual + quantitative) không?

Kết quả kỳ vọng:
  - cfg0 ~40-50% (MMLU physics với LLM zero-shot thường ~40-50%)
  - cfg3 ≥ cfg0 → positive transfer
  - cfg3 < cfg0 → domain gap (cũng là finding có giá trị)

Chạy:
    python experiments/step9_external_benchmark.py
    python experiments/step9_external_benchmark.py --smoke-test
    python experiments/step9_external_benchmark.py --configs cfg0 cfg3
"""
from __future__ import annotations
import sys, os, json, time, gc, argparse, re
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true")
parser.add_argument("--configs",    nargs="*", default=["cfg0", "cfg2", "cfg3"],
                    help="Which configs to eval: cfg0 cfg2 cfg3")
parser.add_argument("--benchmark",   choices=["physics", "logic"], default="physics",
                    help="physics = MMLU physics (MCQ); logic = FOLIO entailment "
                         "(premises+conclusion -> Yes/No/Unknown)")
args, _ = parser.parse_known_args()
BENCHMARK = args.benchmark

from datetime import datetime
_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

def _ts(): return datetime.now().strftime("%H:%M:%S")
def _vram():
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_reserved() / 1024**3
            tot  = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"VRAM {used:.1f}/{tot:.0f}GB"
    except Exception: pass
    return ""

import torch
from pathlib import Path
from datasets import load_dataset
from src.config import (MODEL_NAME, LOG_DIR, CKPT_DIR, LORA_R, LORA_ALPHA,
                        LORA_DROPOUT, LORA_TARGETS,
                        QWEN35_GRPO_FINAL, QWEN35_SFT_LOGIC_FINAL)

_BSUF          = "" if BENCHMARK == "physics" else f"_{BENCHMARK}"
LOG_OUT        = Path(LOG_DIR) / f"external_benchmark{_BSUF}_{_RUN_TS}.json"
LOG_OUT_LATEST = Path(LOG_DIR) / f"external_benchmark{_BSUF}_results_latest.json"
SMOKE_N        = 20

# ── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert in Physics. "
    "Read the question carefully and select the correct answer.\n"
    "Think step by step. Then respond in the EXACT format:\n"
    "<reasoning>\n[Your step-by-step reasoning]\n</reasoning>\n"
    "<answer>\n[Just the letter: A, B, C, or D]\n</answer>"
)

# Logic (FOLIO) — mirrors the EXACT-2026 logic answer space (Yes/No/Unknown) the
# fine-tuned models were trained on, so transfer is measured in-format.
LOGIC_SYSTEM_PROMPT = (
    "You are an expert in formal logical reasoning. "
    "Given a set of premises and a conclusion, decide whether the conclusion "
    "logically follows.\n"
    "Think step by step. Then respond in the EXACT format:\n"
    "<reasoning>\n[Your step-by-step reasoning]\n</reasoning>\n"
    "<answer>\n[Yes if the conclusion is entailed, No if it is contradicted, "
    "Unknown if it cannot be determined from the premises]\n</answer>"
)

# ── Load MMLU Physics ─────────────────────────────────────────────────────────

def load_mmlu_physics(smoke: bool = False) -> list[dict]:
    """Load and format MMLU high_school_physics + college_physics."""
    print(f"[{_ts()}] Loading MMLU physics datasets ...")
    samples = []

    for subj, label in [
        ("high_school_physics", "hs_physics"),
        ("college_physics",     "college_physics"),
    ]:
        ds = load_dataset("cais/mmlu", subj, trust_remote_code=True)["test"]
        print(f"[{_ts()}]   mmlu/{subj}: {len(ds)} samples")
        for row in ds:
            choices = row["choices"]
            letters = ["A", "B", "C", "D"]
            # Format as readable MCQ
            choices_str = "\n".join(
                f"{letters[i]}. {choices[i]}" for i in range(len(choices))
            )
            question_full = f"{row['question']}\n\nOptions:\n{choices_str}"
            answer_letter = letters[row["answer"]]
            samples.append({
                "question":      question_full,
                "answer":        answer_letter,
                "subject":       "physics",
                "subj_label":    label,
                "answer_index":  row["answer"],
            })

    if smoke:
        samples = samples[:SMOKE_N]
        print(f"[{_ts()}]   Smoke test: {SMOKE_N} samples")

    # Print distribution
    hs = sum(1 for s in samples if s["subj_label"] == "hs_physics")
    co = sum(1 for s in samples if s["subj_label"] == "college_physics")
    print(f"[{_ts()}] Total: {len(samples)} samples "
          f"(HS={hs}, College={co})")
    return samples


# ── Load FOLIO logic ──────────────────────────────────────────────────────────

_FOLIO_LABEL_MAP = {"true": "yes", "false": "no", "uncertain": "unknown"}

def load_folio_logic(smoke: bool = False) -> list[dict]:
    """Load the FOLIO validation split (first-order-logic entailment).

    FOLIO label {True, False, Uncertain} maps to the EXACT-2026 logic answer
    space {Yes, No, Unknown}, so this measures in-format logic transfer.
    """
    print(f"[{_ts()}] Loading FOLIO (tasksource/folio) ...")
    ds = load_dataset("tasksource/folio")["validation"]
    print(f"[{_ts()}]   folio/validation: {len(ds)} samples")
    samples = []
    for row in ds:
        label = str(row["label"]).strip().lower()
        gold = _FOLIO_LABEL_MAP.get(label)
        if gold is None:        # skip any unexpected label
            continue
        question_full = (
            f"Premises:\n{row['premises']}\n\n"
            f"Conclusion: {row['conclusion']}\n\n"
            f"Does the conclusion follow from the premises? "
            f"Answer Yes, No, or Unknown."
        )
        samples.append({
            "question":   question_full,
            "answer":     gold,            # yes / no / unknown
            "subject":    "logic",
            "subj_label": "folio",
        })
    if smoke:
        samples = samples[:SMOKE_N]
        print(f"[{_ts()}]   Smoke test: {SMOKE_N} samples")
    print(f"[{_ts()}] Total: {len(samples)} FOLIO samples")
    return samples


# ── Logic answer extraction (Yes/No/Unknown) ───────────────────────────────────

def extract_logic_answer(raw: str) -> str:
    """Extract a normalised {yes,no,unknown} label from a generation."""
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw,
                  flags=re.DOTALL | re.IGNORECASE)
    cand = (m.group(1) if m else raw).strip().lower()
    # priority: unknown first (so 'uncertain' is not swallowed by true/false),
    # then no/false, then yes/true.
    if re.search(r"\bunknown\b|\buncertain\b|cannot be (determined|concluded)"
                 r"|not enough|insufficient", cand):
        return "unknown"
    if re.search(r"\bno\b|\bfalse\b|contradict", cand):
        return "no"
    if re.search(r"\byes\b|\btrue\b|entail|follows?\b", cand):
        return "yes"
    return ""


# ── Config definitions ────────────────────────────────────────────────────────

CONFIGS = {
    "cfg0": {
        "name":       "cfg0_zero_shot",
        "desc":       "Qwen3.5-4B zero-shot (no fine-tuning)",
        "checkpoint": None,   # base model
    },
    "cfg2": {
        "name":       "cfg2_sft_logic",
        "desc":       "cfg2: SFT Ph1 + Logic SFT",
        "checkpoint": QWEN35_SFT_LOGIC_FINAL,
    },
    "cfg3": {
        "name":       "cfg3_grpo_mixed",
        "desc":       "cfg3: SFT Ph1 + Logic SFT + GRPO-mixed",
        "checkpoint": QWEN35_GRPO_FINAL,
    },
}


# ── Inference ─────────────────────────────────────────────────────────────────

def eval_config(config_key: str, samples: list[dict]) -> dict:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    cfg = CONFIGS[config_key]
    sys_prompt = SYSTEM_PROMPT if BENCHMARK == "physics" else LOGIC_SYSTEM_PROMPT
    print(f"\n[{_ts()}] {'='*60}")
    print(f"[{_ts()}]  Evaluating: {cfg['desc']}  (benchmark={BENCHMARK})")
    ckpt = cfg["checkpoint"]
    if ckpt and not os.path.exists(ckpt):
        print(f"[{_ts()}]  WARNING: checkpoint not found at {ckpt}")
        print(f"[{_ts()}]  → Falling back to zero-shot (cfg0)")
        ckpt = None
    print(f"[{_ts()}]  Checkpoint: {ckpt or 'base model'}")

    # ── Load model ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)

    if ckpt:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, ckpt)
        model = model.merge_and_unload()
    else:
        model = base
    model.eval()
    print(f"[{_ts()}]  Model loaded  {_vram()}")

    # ── Evaluate ───────────────────────────────────────────────────────────
    n_correct = 0
    per_sample = []
    per_subject_correct = {}
    per_subject_total   = {}
    t0 = time.time()

    BATCH = 16   # was 4; left-padding makes batched greedy gen identical, and
                 # VRAM is ~9/80GB at batch 4 — bump to cut 1024-token runtime ~4x
    for i in range(0, len(samples), BATCH):
        batch = samples[i:i+BATCH]
        prompts = []
        for s in batch:
            user_content = (f"Question:\n{s['question']}"
                            if BENCHMARK == "physics" else s["question"])
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_content},
            ]
            try:
                p = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                p = f"{sys_prompt}\n\n{user_content}\n\nResponse:"
            prompts.append(p)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(
                # 1024 (was 512): 512 cut off the CoT before <answer>, so the
                # extractor fell back to a random A/B/C/D from mid-reasoning (~random acc)
                **inputs, max_new_tokens=1024, do_sample=False, temperature=1.0,
                pad_token_id=tokenizer.pad_token_id)

        for j, (s, inp_ids) in enumerate(zip(batch, inputs["input_ids"])):
            gen_ids = out[j][len(inp_ids):]
            raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            if BENCHMARK == "logic":
                # FOLIO: extract a normalised Yes/No/Unknown label
                pred = extract_logic_answer(raw)
            else:
                # MMLU MCQ: extract from <answer> tag, then last letter A/B/C/D
                pred = ""
                m = re.search(r"<answer>\s*(.*?)\s*</answer>", raw,
                              flags=re.DOTALL | re.IGNORECASE)
                if m:
                    ans = m.group(1).strip().upper()
                    # standalone letter first ("B", "Option B", "Answer: C") so we
                    # don't grab the C in "CORRECT"; fall back to any A-D char
                    lm = re.search(r'\b([ABCD])\b', ans) or re.search(r'([ABCD])', ans)
                    if lm:
                        pred = lm.group(1)
                if not pred:
                    for line in reversed(raw.splitlines()):
                        line = line.strip().upper()
                        if line in ("A", "B", "C", "D"):
                            pred = line; break
                        # "A." or "A)" patterns
                        if re.match(r'^[ABCD][.):]?\s*$', line):
                            pred = line[0]; break
                if not pred:
                    # Last resort: last standalone A/B/C/D token in the whole output
                    # (better than a random char inside a word in the last 100 chars)
                    letters = re.findall(r'\b([ABCD])\b', raw.upper())
                    if letters:
                        pred = letters[-1]

            correct = (pred == s["answer"])
            n_correct += int(correct)
            subj = s["subj_label"]
            per_subject_total[subj]   = per_subject_total.get(subj, 0) + 1
            per_subject_correct[subj] = per_subject_correct.get(subj, 0) + int(correct)

            per_sample.append({
                "config":    config_key,
                "subj_label": subj,
                "question":  s["question"][:100],
                "pred":      pred,
                "gt":        s["answer"],
                "correct":   correct,
                "raw_end":   raw[-150:],
            })

        done = min(i + BATCH, len(samples))
        elapsed = time.time() - t0
        print(f"[{_ts()}]  {done}/{len(samples)} | acc={n_correct/done*100:.1f}%"
              f" | {elapsed/60:.1f}min  {_vram()}", flush=True)

    # Cleanup
    del model
    if ckpt:
        del base
    gc.collect()
    torch.cuda.empty_cache()

    elapsed_min = (time.time() - t0) / 60
    per_subject_acc = {
        subj: round(per_subject_correct[subj] / per_subject_total[subj] * 100, 2)
        for subj in per_subject_total
    }

    result = {
        "config":            config_key,
        "name":              cfg["name"],
        "desc":              cfg["desc"],
        "n_total":           len(samples),
        "n_correct":         n_correct,
        "accuracy_overall":  round(n_correct / len(samples) * 100, 2),
        "per_subject":       per_subject_acc,
        "elapsed_min":       round(elapsed_min, 1),
        "per_sample":        per_sample,
    }
    print(f"[{_ts()}]  RESULT: overall={result['accuracy_overall']}%  "
          f"per_subject={per_subject_acc}")
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

if BENCHMARK == "logic":
    samples = load_folio_logic(smoke=args.smoke_test)
    _DATASET_NAME = "FOLIO validation (first-order-logic entailment)"
else:
    samples = load_mmlu_physics(smoke=args.smoke_test)
    _DATASET_NAME = "MMLU high_school_physics + college_physics"
configs_to_run = [k for k in args.configs if k in CONFIGS]

print(f"[{_ts()}] Benchmark: {BENCHMARK} | Configs to run: {configs_to_run}")
all_results = []

for cfg_key in configs_to_run:
    r = eval_config(cfg_key, samples)
    all_results.append(r)

    out = {
        "run_ts":     _RUN_TS,
        "smoke":      args.smoke_test,
        "benchmark":  BENCHMARK,
        "dataset":    _DATASET_NAME,
        "n_samples":  len(samples),
        "results":    all_results,
    }
    with open(LOG_OUT, "w") as f:
        json.dump(out, f, indent=2)
    with open(LOG_OUT_LATEST, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{_ts()}] Saved → {LOG_OUT_LATEST}")


# ── Summary ───────────────────────────────────────────────────────────────────

_title = ("MMLU PHYSICS BENCHMARK SUMMARY" if BENCHMARK == "physics"
          else "FOLIO LOGIC BENCHMARK SUMMARY")
print(f"\n[{_ts()}] ╔{'═'*65}╗")
print(f"[{_ts()}] ║ {_title}{' '*(63-len(_title))}║")
print(f"[{_ts()}] ╠{'═'*65}╣")
print(f"[{_ts()}] ║ {'Config':<12} {'Overall':>8} {'Desc':<41} ║")
print(f"[{_ts()}] ╠{'═'*65}╣")
for r in all_results:
    print(f"[{_ts()}] ║ {r['config']:<12} {r['accuracy_overall']:>7.2f}%"
          f"  {r['desc'][:41]:<41} ║")
print(f"[{_ts()}] ╚{'═'*65}╝")

# Note for paper
print(f"\n[{_ts()}] NOTE FOR PAPER ({BENCHMARK}):")
if len(all_results) >= 2 and any(r["config"] == "cfg0" for r in all_results) \
        and any(r["config"] == "cfg3" for r in all_results):
    cfg0_acc = next(r["accuracy_overall"] for r in all_results if r["config"] == "cfg0")
    cfg3_acc = next(r["accuracy_overall"] for r in all_results if r["config"] == "cfg3")
    delta = cfg3_acc - cfg0_acc
    if delta > 2:
        print(f"[{_ts()}] Positive transfer: cfg3 outperforms cfg0 by +{delta:.2f}pp on {BENCHMARK}")
        print(f"[{_ts()}] → Supports generalization claim in paper")
    elif delta < -2:
        print(f"[{_ts()}] Domain gap: cfg3 is {abs(delta):.2f}pp WORSE than cfg0 on {BENCHMARK}")
        print(f"[{_ts()}] → Report as honest finding; EXACT fine-tuning hurts {BENCHMARK} transfer")
    else:
        print(f"[{_ts()}] Neutral: difference {delta:+.2f}pp (within noise)")
        print(f"[{_ts()}] → Fine-tuning on EXACT does not hurt {BENCHMARK} generalization")
