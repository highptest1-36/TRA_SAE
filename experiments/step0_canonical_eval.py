"""
BƯỚC 0 (CANONICAL): Một lần eval DUY NHẤT cho cả 6 config 0-5
================================================================
Đây là nguồn-sự-thật-duy-nhất (single source of truth) cho TẤT CẢ bảng số
trong paper. Thay thế việc trộn 2 lần chạy (Eval-V1 cũ + v2 bugfix) vốn cho
ra số mâu thuẫn (cfg0 35.48% vs 45.16%, cfg2 vs cfg3 đảo ngôi).

Khác biệt so với run_phase4_v2_agent.py:
  - Chạy cả 6 config (0-5) trong CÙNG một lần, cùng prompt/decoding/verifier.
  - Đếm n_physics/n_logic từ NHÃN THẬT của dataset (fix bug 146→141).
  - Ghi per-sample JSONL cho TẤT CẢ 6 config (không chỉ cfg5) → phục vụ
    McNemar (step4) và error analysis (step4/step12).
  - Hỗ trợ --seed để chạy multi-seed (gộp luôn step2).

Output:
  logs/qwen35_ablation_canonical[_seed{S}].json        — accuracy per config
  logs/ablation_per_sample_canonical[_seed{S}].jsonl   — per-sample cho 6 config
  logs/qwen35_ablation_v2_latest.json                  — alias (downstream compat)
  logs/ablation_per_sample_latest.jsonl                — alias (downstream compat)

Chạy:
    python experiments/step0_canonical_eval.py                 # full, seed 42
    python experiments/step0_canonical_eval.py --smoke-test    # 10 mẫu/config
    python experiments/step0_canonical_eval.py --seed 1337     # multi-seed
    python experiments/step0_canonical_eval.py --config 2      # chỉ 1 config
"""
from __future__ import annotations
import sys, os, json, time, gc, argparse
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true")
parser.add_argument("--seed", type=int, default=42, help="generation seed")
parser.add_argument("--config", type=int, default=-1, help="run only this config id (-1=all)")
parser.add_argument("--retries", type=int, default=-1, help="override max retries (0 = single-pass)")
parser.add_argument("--out-tag", type=str, default="", help="isolated output tag; does NOT touch canonical/compat files")
args, _ = parser.parse_known_args()

from src.config import (
    MODEL_NAME, QWEN35_SFT_FINAL, QWEN35_SFT_LOGIC_FINAL, QWEN35_GRPO_FINAL,
    QWEN35_GRPO_PHYS_FINAL, QWEN35_GRPO_LOGIC_FINAL,
    VAL_DS, LOG_DIR, TRAIN_DS, MAX_NEW_TOKENS, AGENT_MAX_RETRIES,
    ROUTER_PATH, SELF_CONSISTENCY_N, SELF_CONSISTENCY_TEMP,
)
from src.utils import setup_logger, print_vram
from src.symbolic_verifier import verify_answer, extract_answer_from_text
import torch
from pathlib import Path
from collections import Counter
from datetime import datetime

logger = setup_logger("step0_canonical", LOG_DIR)

# Reproducibility
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

EVAL_BATCH_SIZE = 8
_MAX_RETRIES = args.retries if args.retries >= 0 else AGENT_MAX_RETRIES
_SMOKE_N = 10
_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

_seed_tag = "" if args.seed == 42 else f"_seed{args.seed}"
_otag = f"_{args.out_tag}" if args.out_tag else ""
LOG_OUT           = Path(LOG_DIR) / f"qwen35_ablation_canonical{_seed_tag}{_otag}_{_RUN_TS}.json"
LOG_OUT_CANON     = Path(LOG_DIR) / f"qwen35_ablation_canonical{_seed_tag}{_otag}.json"
LOG_OUT_COMPAT    = Path(LOG_DIR) / "qwen35_ablation_v2_latest.json"   # downstream compat
PER_SAMPLE_OUT    = Path(LOG_DIR) / f"ablation_per_sample_canonical{_seed_tag}{_otag}.jsonl"
PER_SAMPLE_COMPAT = Path(LOG_DIR) / "ablation_per_sample_latest.jsonl"  # downstream compat

SYSTEM_PROMPT = (
    "You are an expert in Logic and Physics. "
    "Think step by step and respond in the exact format:\n"
    "<reasoning>\n[Your step-by-step reasoning]\n</reasoning>\n"
    "<answer>\n[Final answer: letter / Yes/No/Unknown / number+unit]\n</answer>\n"
    "<explanation>\n[Concise explanation of why this answer is correct]\n</explanation>"
)

# ── 6-config ablation matrix (unified) ───────────────────────────────────────
CONFIGS = [
    {"id": 0, "name": "zero_shot",        "desc": "Qwen3.5-4B zero-shot (no LoRA)",
     "lora": None,                  "dual": False, "router": False, "retr": False, "sc": False},
    {"id": 1, "name": "sft_phase1",       "desc": "+ SFT Phase 1",
     "lora": QWEN35_SFT_FINAL,      "dual": False, "router": False, "retr": True,  "sc": False},
    {"id": 2, "name": "sft_logic",        "desc": "+ Logic SFT Phase 1.5",
     "lora": QWEN35_SFT_LOGIC_FINAL,"dual": False, "router": False, "retr": True,  "sc": False},
    {"id": 3, "name": "grpo_mixed",       "desc": "+ GRPO mixed Phase 2",
     "lora": QWEN35_GRPO_FINAL,     "dual": False, "router": False, "retr": True,  "sc": False},
    {"id": 4, "name": "dual_lora_router", "desc": "+ Dual LoRA specialists + Router",
     "lora": None,                  "dual": True,  "router": True,  "retr": True,  "sc": False},
    {"id": 5, "name": "full_v2",          "desc": "+ Self-consistency 5x (full v2 agent)",
     "lora": None,                  "dual": True,  "router": True,  "retr": True,  "sc": True},
    {"id": 6, "name": "zero_shot_retr",   "desc": "Qwen3.5-4B zero-shot + retrieval (no LoRA)",
     "lora": None,                  "dual": False, "router": False, "retr": True,  "sc": False},
]


def _generate_batch(model, tokenizer, questions, temperature, device, subjects, few_shots):
    texts = []
    for q, fs, subj in zip(questions, few_shots, subjects):
        if subj and hasattr(model, "set_adapter"):
            try:
                model.set_adapter(subj)
            except Exception:
                pass
        user_content = (fs + q) if fs else q
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
        texts.append(text)

    tokenizer.padding_side = "left"
    enc = tokenizer(texts, return_tensors="pt", truncation=True,
                    max_length=MAX_NEW_TOKENS * 2, padding=True).to(device)
    do_sample = temperature > 0.05
    gen_kwargs = dict(
        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
        max_new_tokens=MAX_NEW_TOKENS, do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
    input_len = enc["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**gen_kwargs)
    return [tokenizer.decode(out[i][input_len:], skip_special_tokens=True)
            for i in range(len(questions))]


def _self_consistency_one(model, tokenizer, question, device, subject):
    """N-sample majority vote (batched)."""
    cands = _generate_batch(
        model, tokenizer, [question] * SELF_CONSISTENCY_N,
        SELF_CONSISTENCY_TEMP, device, [subject] * SELF_CONSISTENCY_N,
        [""] * SELF_CONSISTENCY_N)
    extracted = [extract_answer_from_text(c) for c in cands]
    best_ans, _ = Counter(extracted).most_common(1)[0]
    for raw in cands:
        if extract_answer_from_text(raw) == best_ans:
            return raw
    return cands[0]


def _fmt_few_shot(examples):
    if not examples:
        return ""
    lines = ["\n--- Reference examples ---"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"\nExample {i}:\n{ex['question'][:300]}\n<answer>\n{ex['answer']}\n</answer>")
    lines.append("\n--- Now answer the following ---\n")
    return "\n".join(lines)


# ── Load data ────────────────────────────────────────────────────────────────
print(f"[1] Loading val set | seed={args.seed} | ts={_RUN_TS}")
from datasets import load_from_disk
val_ds = load_from_disk(VAL_DS)
N = _SMOKE_N if args.smoke_test else len(val_ds)
eval_ds = val_ds.select(range(min(N, len(val_ds))))
N = len(eval_ds)
subj_counts = Counter(eval_ds["type"])
print(f"    Eval samples: {N}  composition={dict(subj_counts)}")
assert subj_counts.get("physics", 0) + subj_counts.get("logic", 0) == N, "unknown subject in val"

print("[2] Loading base model")
from src.model_loader import load_base_model, load_multi_adapter_model
from peft import PeftModel
base_model, tokenizer = load_base_model(
    model_name=MODEL_NAME, dtype=torch.bfloat16, drop_vision=True, device_map="auto")
device = "cuda" if torch.cuda.is_available() else "cpu"
print_vram("base model loaded")

print("[3] Loading retriever")
from src.retriever import Retriever
retriever = Retriever(TRAIN_DS)
retriever.build()

print("[4] Loading router")
router = None
if Path(ROUTER_PATH).exists():
    from src.router import SubjectRouter
    router = SubjectRouter.load(ROUTER_PATH)
    print(f"    Router loaded from {ROUTER_PATH}")
else:
    try:
        from src.router import get_router
        router = get_router(ROUTER_PATH, TRAIN_DS)
        print("    Router trained and cached")
    except Exception as e:
        print(f"    Router unavailable ({e}); using dataset labels")

# Pre-gather questions/gts/TRUE subjects once (shared across configs)
samples_list = list(eval_ds)
base_questions, base_gts, true_subjects = [], [], []
for s in samples_list:
    q = s["prompt"][-1]["content"] if s.get("prompt") else s.get("question", "")
    base_questions.append(q)
    base_gts.append(str(s.get("answer", "")))
    true_subjects.append(str(s.get("type", "")))   # TRUE label — used for denominators

def _save(partial: bool):
    ablation_results.sort(key=lambda x: x["config_id"])
    payload = {"ablation": ablation_results,
               "version": "canonical" + ("_partial" if partial else ""),
               "seed": args.seed, "run_timestamp": _RUN_TS, "smoke_test": args.smoke_test}
    targets = [LOG_OUT, LOG_OUT_CANON]
    alias = (args.config < 0 and not args.smoke_test and args.seed == 42 and not args.out_tag)
    if alias:
        targets.append(LOG_OUT_COMPAT)
    for p in targets:
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)
    ps_targets = [PER_SAMPLE_OUT] + ([PER_SAMPLE_COMPAT] if alias else [])
    for p in ps_targets:
        with open(p, "w") as f:
            for row in all_per_sample_rows:
                f.write(json.dumps(row) + "\n")


# Auto-resume: skip configs already present in this seed's canonical file
ablation_results = []
all_per_sample_rows = []
_done_ids = set()
if LOG_OUT_CANON.exists() and not args.smoke_test:
    try:
        prev = json.load(open(LOG_OUT_CANON))
        ablation_results = prev.get("ablation", [])
        _done_ids = {r["config_id"] for r in ablation_results}
        if PER_SAMPLE_OUT.exists():
            all_per_sample_rows = [json.loads(l) for l in open(PER_SAMPLE_OUT) if l.strip()]
        if _done_ids:
            print(f"    RESUME: configs already done = {sorted(_done_ids)}")
    except Exception as e:
        print(f"    (could not resume: {e})")

configs_to_run = [c for c in CONFIGS
                  if (args.config < 0 or c["id"] == args.config) and c["id"] not in _done_ids]

for cfg in configs_to_run:
    cfg_id, cfg_name = cfg["id"], cfg["name"]
    print(f"\n{'='*64}\nConfig {cfg_id}: {cfg['desc']}\n{'='*64}")

    # ── Load model variant ────────────────────────────────────────────────────
    if cfg["dual"]:
        adapters = {}
        for nm, ck in [("physics", QWEN35_GRPO_PHYS_FINAL), ("logic", QWEN35_GRPO_LOGIC_FINAL)]:
            if Path(ck).exists():
                adapters[nm] = ck
            elif Path(QWEN35_GRPO_FINAL).exists():
                adapters[nm] = QWEN35_GRPO_FINAL
                print(f"    WARN: {nm} specialist missing → mixed-GRPO fallback")
        model = load_multi_adapter_model(MODEL_NAME, adapters, dtype=torch.bfloat16,
                                         drop_vision=True)[0] if adapters else base_model
    elif cfg["lora"] and Path(cfg["lora"]).exists():
        model = PeftModel.from_pretrained(base_model, cfg["lora"], is_trainable=False)
    else:
        model = base_model
    model.eval()
    print_vram(f"cfg{cfg_id} model loaded")

    # ── Adapter-selection subject (router) + few-shots ────────────────────────
    adapter_subjects, few_shots = [], []
    for i in range(N):
        subj = true_subjects[i]
        if cfg["router"] and router is not None and subj not in ("physics", "logic"):
            subj, _ = router.predict(base_questions[i])
        adapter_subjects.append(subj)
        fs = ""
        if cfg["retr"]:
            try:
                ex = retriever.retrieve(base_questions[i], top_k=3, subject=subj or None)
                fs = _fmt_few_shot(ex)
            except Exception:
                pass
        few_shots.append(fs)

    # ── Eval loop (attempt 0 greedy → retries; cfg5 uses SC on retries) ───────
    t0 = time.time()
    correct_arr = [False] * N
    raw_outputs = [None] * N
    retry_counts = [0] * N
    resolved = [False] * N

    for attempt in range(_MAX_RETRIES + 1):
        pending = list(range(N)) if attempt == 0 else [i for i in range(N) if not resolved[i]]
        if not pending:
            break
        print(f"  attempt {attempt}: {len(pending)} pending", flush=True)

        if attempt >= 1 and cfg["sc"]:
            for i in pending:
                raw = _self_consistency_one(model, tokenizer, base_questions[i],
                                            device, adapter_subjects[i])
                ok = verify_answer(extract_answer_from_text(raw), base_gts[i],
                                   subject=true_subjects[i],
                                   question_text=base_questions[i], use_z3=True)
                raw_outputs[i], correct_arr[i], retry_counts[i] = raw, ok, attempt
                if ok or attempt == _MAX_RETRIES:
                    resolved[i] = True
        else:
            temperature = 0.1 if attempt == 0 else 0.7
            fs_list = few_shots if attempt == 0 else [""] * N
            raw_list = []
            for bs in range(0, len(pending), EVAL_BATCH_SIZE):
                bidx = pending[bs: bs + EVAL_BATCH_SIZE]
                raw_list.extend(_generate_batch(
                    model, tokenizer,
                    [base_questions[i] for i in bidx], temperature, device,
                    [adapter_subjects[i] for i in bidx], [fs_list[i] for i in bidx]))
            for pos, i in enumerate(pending):
                raw = raw_list[pos]
                ok = verify_answer(extract_answer_from_text(raw), base_gts[i],
                                   subject=true_subjects[i],
                                   question_text=base_questions[i], use_z3=True)
                raw_outputs[i], correct_arr[i], retry_counts[i] = raw, ok, attempt
                if ok or attempt == _MAX_RETRIES:
                    resolved[i] = True
        print(f"  attempt {attempt}: correct={sum(correct_arr)}/{N}", flush=True)

    # ── Metrics — denominators from TRUE labels (fixes 146→141 bug) ──────────
    correct_all = correct_phys = correct_logic = 0
    n_phys = n_logic = 0
    per_retry = {0: 0, 1: 0, 2: 0}
    for i in range(N):
        if correct_arr[i]:
            correct_all += 1
        if true_subjects[i] == "physics":
            n_phys += 1
            correct_phys += int(correct_arr[i])
        elif true_subjects[i] == "logic":
            n_logic += 1
            correct_logic += int(correct_arr[i])
        per_retry[min(retry_counts[i], 2)] = per_retry.get(min(retry_counts[i], 2), 0) + 1
    assert n_phys + n_logic == N, f"BUG: {n_phys}+{n_logic}!={N}"

    elapsed = round((time.time() - t0) / 60, 1)
    result = {
        "config_id": cfg_id, "config_name": cfg_name, "description": cfg["desc"],
        "accuracy_overall": round(correct_all / N * 100, 2),
        "accuracy_physics": round(correct_phys / n_phys * 100, 2) if n_phys else 0.0,
        "accuracy_logic": round(correct_logic / n_logic * 100, 2) if n_logic else 0.0,
        "n_total": N, "n_physics": n_phys, "n_logic": n_logic, "n_correct": correct_all,
        "per_retry": per_retry, "elapsed_min": elapsed,
        "seed": args.seed, "smoke_test": args.smoke_test, "canonical": True,
    }
    ablation_results.append(result)
    for i in range(N):
        all_per_sample_rows.append({
            "config_id": cfg_id, "config_name": cfg_name,
            "correct": bool(correct_arr[i]), "subject": true_subjects[i],
            "question": base_questions[i][:200],
            "prediction": extract_answer_from_text(raw_outputs[i]) if raw_outputs[i] else "",
            "raw_output": (raw_outputs[i] or "")[:1500],   # for manual error annotation
            "ground_truth": base_gts[i], "retry_count": retry_counts[i], "seed": args.seed,
        })
    print(f"  cfg{cfg_id} DONE: overall={result['accuracy_overall']}%  "
          f"phys={result['accuracy_physics']}%  logic={result['accuracy_logic']}%  "
          f"[{n_phys}+{n_logic}={N}]  {elapsed}min")
    _save(partial=True)   # checkpoint after each config (Colab-disconnect safe)
    print(f"  [checkpoint saved → {LOG_OUT_CANON.name}]")

    if model is not base_model:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ── Finalize (canonical + downstream-compat aliases) ─────────────────────────
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
_save(partial=False)

print(f"\nSaved: {LOG_OUT_CANON}")
print(f"Saved per-sample: {PER_SAMPLE_OUT}")
print("\nSANITY CHECK (n_phys + n_logic == n_total):")
for r in ablation_results:
    tot = r["n_physics"] + r["n_logic"]
    print(f"  cfg{r['config_id']} {r['config_name']:<18} acc={r['accuracy_overall']:>6.2f}%  "
          f"{r['n_physics']}+{r['n_logic']}={tot}  {'OK' if tot == r['n_total'] else 'BUG'}")
print("\nNOTE: rebuild ALL paper tables from this file. Do not mix with old qwen35_ablation.json.")
