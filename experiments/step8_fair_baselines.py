"""
BƯỚC 8: Fair Baseline Re-evaluation
=====================================
Chạy lại 5 baselines với 3 loại prompt khác nhau để đánh giá công bằng.

Vấn đề gốc: Step 3 dùng XML-forcing prompt cho tất cả models → models không được
fine-tune XML (Mistral, Llemma, DeepSeek) bị penalize vì format, không phải reasoning.

3 prompt strategies:
  xml       : yêu cầu XML tags (giống step3 gốc — dùng làm baseline so sánh)
  cot_boxed : CoT + \\boxed{answer}  (native format của Qwen-Math, DeepSeek-Math)
  cot_plain : CoT + "The answer is:" (native format của Mistral, Llemma)

Extractor thống nhất: XML tag → \\boxed{} → "Answer:" / "The answer is:" → last line

Chạy:
    python experiments/step8_fair_baselines.py
    python experiments/step8_fair_baselines.py --smoke-test
    python experiments/step8_fair_baselines.py --only-ids B2 B4 --only-strategies xml cot_boxed
"""
from __future__ import annotations
import sys, os, json, time, gc, argparse, re
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test",      action="store_true")
parser.add_argument("--only-ids",        nargs="*", default=None,
                    help="subset e.g. B2 B3 B4 B5 B6")
parser.add_argument("--only-strategies", nargs="*", default=None,
                    help="subset of: xml cot_boxed cot_plain")
parser.add_argument("--with-retrieval", action="store_true",
                    help="Prepend the SAME top-3 cfg0-R retrieval exemplars to each "
                         "question (isolates whether retrieval also helps the 7B baselines)")
args, _ = parser.parse_known_args()

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

from src.config    import VAL_DS, LOG_DIR
from src.symbolic_verifier import verify_answer, extract_answer_from_text
import torch
from pathlib import Path

_SUFFIX        = "_retrieval" if args.with_retrieval else ""
LOG_OUT        = Path(LOG_DIR) / f"fair_baselines{_SUFFIX}_{_RUN_TS}.json"
LOG_OUT_LATEST = Path(LOG_DIR) / f"fair_baselines{_SUFFIX}_results_latest.json"
SMOKE_N        = 10

# ── Prompt templates ──────────────────────────────────────────────────────────

PROMPT_XML = (
    "You are an expert in Logic and Physics. "
    "Think step by step and respond in the EXACT format:\n"
    "<reasoning>\n[Your step-by-step reasoning]\n</reasoning>\n"
    "<answer>\n[Final answer: letter / Yes/No/Unknown / number+unit]\n</answer>\n"
    "<explanation>\n[Concise explanation]\n</explanation>"
)

PROMPT_COT_BOXED = (
    "You are an expert in Logic and Physics. "
    "Solve the problem step by step. "
    "At the end, write your final answer inside \\boxed{...}.\n"
    "For MCQ, write the letter. For yes/no questions, write Yes, No, or Unknown. "
    "For numerical problems, include units."
)

PROMPT_COT_PLAIN = (
    "You are an expert in Logic and Physics. "
    "Solve the problem step by step, showing all reasoning. "
    "End your response with: 'The answer is: [your answer]'\n"
    "For MCQ, the answer is just the letter (A/B/C/D). "
    "For yes/no questions, answer Yes, No, or Unknown. "
    "For numerical problems, include units."
)

# ── Flexible answer extractor ─────────────────────────────────────────────────

_ANSWER_LINE_RE = re.compile(
    r'(?:the\s+answer\s+is|answer\s*[:=])\s*[:\-]?\s*(.+)',
    re.IGNORECASE
)

def flexible_extract(text: str) -> str:
    """
    Multi-strategy extraction (in priority order):
      1. <answer>...</answer>  XML tag
      2. \\boxed{...}           LaTeX boxed
      3. "The answer is: X"    plain text marker
      4. Last non-empty line   fallback
    """
    # 1. XML tag (handles our fine-tuned model and XML-compliant baselines)
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text,
                  flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2. \boxed{} (Qwen2.5-Math, DeepSeek-Math, Llemma native)
    extracted = extract_answer_from_text(text)
    # extract_answer_from_text already tries \boxed then last-line
    # But we want to prioritise "The answer is:" over last-line, so:

    # 3. "The answer is:" pattern
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        ma = _ANSWER_LINE_RE.search(line)
        if ma:
            return ma.group(1).strip().rstrip('.')

    # 4. Fallback: last non-empty line (already done by extract_answer_from_text)
    return extracted


# ── Baseline model definitions ────────────────────────────────────────────────

# Per-model recommended strategies
BASELINES = [
    {
        "id": "B2", "model_id": "Qwen/Qwen2.5-Math-7B-Instruct",
        "chat_template": True,
        "desc": "Qwen2.5-Math-7B-Instruct",
        "primary_strategy": "cot_boxed",   # native \boxed format
    },
    {
        "id": "B3", "model_id": "EleutherAI/llemma_7b",
        "chat_template": False,
        "desc": "Llemma-7B",
        "primary_strategy": "cot_plain",   # base model, no XML training
    },
    {
        "id": "B4", "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "chat_template": True,
        "desc": "Mistral-7B-Instruct-v0.3",
        "primary_strategy": "cot_plain",   # instruct model, no XML training
    },
    {
        "id": "B5", "model_id": "Qwen/Qwen2-Math-7B-Instruct",
        "chat_template": True,
        "desc": "Qwen2-Math-7B-Instruct",
        "primary_strategy": "cot_boxed",
    },
    {
        "id": "B6", "model_id": "deepseek-ai/deepseek-math-7b-instruct",
        "chat_template": True,
        "desc": "DeepSeek-Math-7B-Instruct",
        "primary_strategy": "cot_boxed",
    },
]

STRATEGIES = {
    "xml":       PROMPT_XML,
    "cot_boxed": PROMPT_COT_BOXED,
    "cot_plain": PROMPT_COT_PLAIN,
}

if args.only_ids:
    BASELINES = [b for b in BASELINES if b["id"] in args.only_ids]
if args.only_strategies:
    STRATEGIES = {k: v for k, v in STRATEGIES.items() if k in args.only_strategies}

print(f"[{_ts()}] Running {len(BASELINES)} baselines × {len(STRATEGIES)} strategies"
      f"  | retrieval={args.with_retrieval}")
print(f"[{_ts()}] Baselines: {[b['id'] for b in BASELINES]}")
print(f"[{_ts()}] Strategies: {list(STRATEGIES.keys())}")


# ── Load validation data ──────────────────────────────────────────────────────

from datasets import load_from_disk
val_ds = load_from_disk(VAL_DS)
samples = list(val_ds)
# Normalize field access: dataset stores question in prompt[-1].content and
# the domain label in `type` (not `subject`/`question`). Patch both in-place.
for _s in samples:
    if not _s.get("question"):
        _s["question"] = _s["prompt"][-1]["content"] if _s.get("prompt") else ""
    if not _s.get("subject"):
        _s["subject"] = str(_s.get("type", ""))
if args.smoke_test:
    samples = samples[:SMOKE_N]
    print(f"[{_ts()}] SMOKE TEST: {SMOKE_N} samples")
print(f"[{_ts()}] Val samples: {len(samples)}")


# ── Retrieval exemplars (cfg0-R protocol, --with-retrieval) ────────────────────
# Mirrors EXACTLY the few-shot block cfg0-R builds in step1_rerun_cfg0_3.py:
# top-3 subject-filtered training neighbours, each shown as the question (<=300
# chars) plus its <answer> tag, then a hand-off marker before the real question.
RETRIEVAL = args.with_retrieval
few_shots = [""] * len(samples)
if RETRIEVAL:
    from src.config import TRAIN_DS
    from src.retriever import Retriever
    print(f"[{_ts()}] --with-retrieval: building retriever over {TRAIN_DS}")
    _retr = Retriever(TRAIN_DS)
    _retr.build(cache_path=None)   # in-memory; avoids stale-cache 0-doc bug
    def _build_fs(q: str, subj: str) -> str:
        try:
            examples = _retr.retrieve(q, top_k=3, subject=subj or None)
        except Exception:
            return ""
        if not examples:
            return ""
        lines = ["\n--- Reference examples ---"]
        for k, ex in enumerate(examples, 1):
            lines.append(f"\nExample {k}:\n{ex['question'][:300]}\n"
                         f"<answer>\n{ex['answer']}\n</answer>")
        lines.append("\n--- Now answer the following ---\n")
        return "\n".join(lines)
    for _idx, _s in enumerate(samples):
        few_shots[_idx] = _build_fs(_s["question"], _s.get("subject", ""))
    _ne = sum(1 for f in few_shots if f)
    print(f"[{_ts()}] retrieval exemplars built for {_ne}/{len(samples)} samples")


# ── Inference helper ──────────────────────────────────────────────────────────

def _build_messages(question: str, system_prompt: str, chat_template: bool,
                    few_shot: str = "") -> list | str:
    # few_shot (if any) is prepended to the user turn, exactly as cfg0-R does
    # (system prompt unchanged; only the retrieval exemplars are added).
    user = f"{few_shot}Question: {question}" if few_shot else f"Question: {question}"
    if chat_template:
        return [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user},
        ]
    else:
        return f"{system_prompt}\n\n{user}\n\nResponse:"


def run_baseline(baseline: dict, strategy_name: str, system_prompt: str) -> dict:
    """Inference for one (baseline, strategy) pair."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    bid      = baseline["id"]
    model_id = baseline["model_id"]
    use_chat = baseline["chat_template"]

    print(f"\n[{_ts()}] {'='*60}")
    print(f"[{_ts()}]  {bid} ({baseline['desc']}) | strategy={strategy_name}")
    print(f"[{_ts()}] {'='*60}")

    # ── Load ──────────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if RETRIEVAL:
        # exemplars are at the START, the real question at the END — truncate
        # from the left so a long prompt never drops the actual question.
        tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    print(f"[{_ts()}]  Loaded  {_vram()}")

    n_correct = n_phys_correct = n_logic_correct = 0
    n_phys = n_logic = 0
    per_sample = []
    t0 = time.time()

    # batch 8: VRAM ~15GB at batch 4 w/ retrieval on 40GB A100 → headroom.
    # greedy + left-padding ⇒ batched generation is identical to batch 1.
    BATCH = 8
    for i in range(0, len(samples), BATCH):
        batch = samples[i:i+BATCH]
        prompts = []
        for j2, s in enumerate(batch):
            fs = few_shots[i + j2] if RETRIEVAL else ""
            msg = _build_messages(s["question"], system_prompt, use_chat, fs)
            if use_chat:
                try:
                    p = tokenizer.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=True)
                except Exception:
                    p = f"{system_prompt}\n\n{fs}Question: {s['question']}\n\nResponse:"
            else:
                p = msg
            prompts.append(p)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True,
                           max_length=(4096 if RETRIEVAL else 2048)).to(model.device)
        with torch.no_grad():
            out = model.generate(
                # 1024 to match src.config.MAX_NEW_TOKENS / step3; 512 cut off
                # long CoT before the final \boxed{} answer (inverted fair-eval result)
                **inputs, max_new_tokens=1024,
                do_sample=False, temperature=1.0, pad_token_id=tokenizer.pad_token_id)

        for j, (s, inp_ids) in enumerate(zip(batch, inputs["input_ids"])):
            gen_ids = out[j][len(inp_ids):]
            raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            # Flexible extraction
            pred = flexible_extract(raw)
            correct = verify_answer(pred, s["answer"],
                                    subject=s.get("subject", ""),
                                    question_text=s["question"])

            subj = s.get("subject", "")
            n_correct += int(correct)
            if subj == "physics":
                n_phys += 1; n_phys_correct += int(correct)
            elif subj == "logic":
                n_logic += 1; n_logic_correct += int(correct)

            per_sample.append({
                "bid": bid, "strategy": strategy_name,
                "question_id": i + j,
                "subject": subj,
                "prediction_raw": raw[:300],
                "prediction_extracted": pred,
                "ground_truth": s["answer"],
                "correct": correct,
            })

        elapsed = time.time() - t0
        done = min(i + BATCH, len(samples))
        print(f"[{_ts()}]  {done}/{len(samples)} | acc={n_correct/done*100:.1f}%"
              f" | {elapsed/60:.1f}min  {_vram()}", flush=True)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    elapsed_min = (time.time() - t0) / 60
    result = {
        "id":              bid,
        "strategy":        strategy_name,
        "retrieval":       RETRIEVAL,
        "model_id":        model_id,
        "description":     f"{baseline['desc']} [{strategy_name}]"
                           + ("+retrieval" if RETRIEVAL else ""),
        "system_prompt":   system_prompt[:100] + "...",
        "n_total":         len(samples),
        "n_correct":       n_correct,
        "n_physics":       n_phys,
        "n_logic":         n_logic,
        "n_phys_correct":  n_phys_correct,
        "n_logic_correct": n_logic_correct,
        "accuracy_overall": round(n_correct / len(samples) * 100, 2),
        "accuracy_physics": round(n_phys_correct / max(n_phys, 1) * 100, 2),
        "accuracy_logic":   round(n_logic_correct / max(n_logic, 1) * 100, 2),
        "elapsed_min":      round(elapsed_min, 1),
        "per_sample":       per_sample,
    }
    print(f"[{_ts()}]  RESULT: overall={result['accuracy_overall']}%"
          f"  phys={result['accuracy_physics']}%  logic={result['accuracy_logic']}%"
          f"  ({elapsed_min:.1f}min)")
    return result


# ── Main loop ─────────────────────────────────────────────────────────────────

all_results = []
for baseline in BASELINES:
    for strategy_name, system_prompt in STRATEGIES.items():
        result = run_baseline(baseline, strategy_name, system_prompt)
        all_results.append(result)

        # Save incrementally (safe against crashes)
        out = {
            "run_ts":       _RUN_TS,
            "smoke_test":   args.smoke_test,
            "retrieval":    args.with_retrieval,
            "n_val":        len(samples),
            "results":      all_results,
        }
        with open(LOG_OUT, "w") as f:
            json.dump(out, f, indent=2)
        with open(LOG_OUT_LATEST, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[{_ts()}]  Saved → {LOG_OUT_LATEST}")


# ── Summary table ─────────────────────────────────────────────────────────────

print(f"\n[{_ts()}] ╔{'═'*72}╗")
print(f"[{_ts()}] ║ FAIR BASELINE SUMMARY{' '*50}║")
print(f"[{_ts()}] ╠{'═'*72}╣")
print(f"[{_ts()}] ║ {'ID':<4} {'Strategy':<12} {'Overall':>8} {'Physics':>8} {'Logic':>8} {'Description':<26} ║")
print(f"[{_ts()}] ╠{'═'*72}╣")

# Also show original results for comparison
print(f"[{_ts()}] ║ --- ORIGINAL RESULTS (step3, XML-forcing) ---{' '*25}║")
original = {
    'B2': (28.57, 36.17, 14.47), 'B3': (12.90, 5.67, 26.32),
    'B4': (9.68, 6.38, 15.79),   'B5': (26.73, 33.33, 14.47),
    'B6': (11.98, 9.93, 15.79),
}
for bid, (ov, ph, lo) in original.items():
    print(f"[{_ts()}] ║ {bid:<4} {'xml_orig':<12} {ov:>7.2f}% {ph:>7.2f}% {lo:>7.2f}% {'':26} ║")

print(f"[{_ts()}] ╠{'═'*72}╣")
print(f"[{_ts()}] ║ --- NEW RESULTS ---{' '*52}║")
for r in all_results:
    marker = " ← primary" if r["strategy"] == next(
        b["primary_strategy"] for b in BASELINES if b["id"] == r["id"]) else ""
    print(f"[{_ts()}] ║ {r['id']:<4} {r['strategy']:<12} "
          f"{r['accuracy_overall']:>7.2f}% {r['accuracy_physics']:>7.2f}% "
          f"{r['accuracy_logic']:>7.2f}% {marker:<26} ║")

print(f"[{_ts()}] ╚{'═'*72}╝")
print(f"[{_ts()}] Saved: {LOG_OUT_LATEST}")
