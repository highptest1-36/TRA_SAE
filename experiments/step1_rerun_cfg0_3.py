"""
BƯỚC 1: Rerun config 0-3 để fix bug đếm mẫu
=============================================
Bug gốc: config 0-3 báo n_physics=146 thay vì 141.
         Nguyên nhân: biến đếm được khởi tạo từ tổng train, không từ val.
Fix: đếm n_physics/n_logic trực tiếp từ all_subjects sau khi resolve.

Chạy:
    python experiments/step1_rerun_cfg0_3.py
    python experiments/step1_rerun_cfg0_3.py --smoke-test   # kiểm tra nhanh
"""
from __future__ import annotations
import sys, os, json, time, gc, argparse
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true")
args, _ = parser.parse_known_args()

from src.config import (
    MODEL_NAME, MAX_SEQ_LEN, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS,
    QWEN35_SFT_FINAL, QWEN35_SFT_LOGIC_FINAL, QWEN35_GRPO_FINAL,
    VAL_DS, LOG_DIR, CKPT_DIR, MAX_NEW_TOKENS, AGENT_MAX_RETRIES, TRAIN_DS,
)
from src.utils import setup_logger, print_vram
from src.symbolic_verifier import verify_answer, extract_answer_from_text
import torch
from pathlib import Path
from collections import Counter

logger = setup_logger("step1_rerun", LOG_DIR)

# Mirror stdout → log file (append mode)
import sys as _sys
class _Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()
    def reconfigure(self, **kw): pass  # no-op shim

EVAL_BATCH_SIZE = 8
_MAX_RETRIES    = AGENT_MAX_RETRIES
_SMOKE_N        = 10

# Timestamp → mỗi lần chạy tạo file riêng, KHÔNG đè kết quả cũ
from datetime import datetime
_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_OUT            = Path(LOG_DIR) / f"qwen35_ablation_v2_{_RUN_TS}.json"
LOG_OUT_LATEST     = Path(LOG_DIR) / "qwen35_ablation_v2_latest.json"
LOG_STDOUT         = Path(LOG_DIR) / f"step1_rerun_{_RUN_TS}.log"
PER_SAMPLE_OUT     = Path(LOG_DIR) / f"ablation_per_sample_{_RUN_TS}.jsonl"
PER_SAMPLE_LATEST  = Path(LOG_DIR) / "ablation_per_sample_latest.jsonl"

SYSTEM_PROMPT = (
    "You are an expert in Logic and Physics. "
    "Think step by step and respond in the exact format:\n"
    "<reasoning>\n[Your step-by-step reasoning]\n</reasoning>\n"
    "<answer>\n[Final answer: letter / Yes/No/Unknown / number+unit]\n</answer>\n"
    "<explanation>\n[Concise explanation of why this answer is correct]\n</explanation>"
)

# Only configs 0-3 need to be rerun
CONFIGS_TO_RERUN = [
    {
        "id": 0, "name": "zero_shot", "desc": "Qwen3.5-4B zero-shot (no LoRA)",
        "lora_ckpt": None, "expected_acc": "25-30%",
    },
    {
        "id": 1, "name": "sft_phase1", "desc": "+ SFT Phase 1",
        "lora_ckpt": QWEN35_SFT_FINAL, "expected_acc": "40-45%",
    },
    {
        "id": 2, "name": "sft_logic", "desc": "+ Logic SFT Phase 1.5",
        "lora_ckpt": QWEN35_SFT_LOGIC_FINAL, "expected_acc": "45-50%",
    },
    {
        "id": 3, "name": "grpo_mixed", "desc": "+ GRPO mixed Phase 2",
        "lora_ckpt": QWEN35_GRPO_FINAL, "expected_acc": "50-55%",
    },
]

# ── Load existing cfg 4-5 from original ablation file ────────────────────────
existing_cfg45 = []
original_abl = Path(LOG_DIR) / "qwen35_ablation.json"
if original_abl.exists():
    with open(original_abl) as f:
        orig = json.load(f)
    existing_cfg45 = [r for r in orig.get("ablation", []) if r["config_id"] >= 4]
    print(f"Loaded cfg 4-5 from original: {len(existing_cfg45)} entries")


def _generate_batch(model, tokenizer, questions, temperatures, device, subjects, few_shots):
    texts = []
    for q, fs in zip(questions, few_shots):
        user_content = fs + q if fs else q
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
    temperature = temperatures[0] if temperatures else 0.1
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


print("[1] Loading evaluation dataset")
# Open log file and tee stdout
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
_logfile = open(LOG_STDOUT, "w", buffering=1)
_sys.stdout = _Tee(_sys.__stdout__, _logfile)
print(f"Run timestamp : {_RUN_TS}")
print(f"Result file   : {LOG_OUT}")
print(f"Stdout log    : {LOG_STDOUT}")
from datasets import load_from_disk
val_ds = load_from_disk(VAL_DS)
N = _SMOKE_N if args.smoke_test else len(val_ds)
eval_ds = val_ds.select(range(min(N, len(val_ds))))
print(f"    Eval samples: {len(eval_ds)}")

# Verify split composition
from collections import Counter
subj_counts = Counter(eval_ds["type"])
print(f"    Composition: {dict(subj_counts)}")
assert subj_counts.get("physics", 0) + subj_counts.get("logic", 0) == len(eval_ds), \
    "ERROR: unknown subject type in dataset"

print("[2] Loading base model")
from src.model_loader import load_base_model
from peft import PeftModel
base_model, tokenizer = load_base_model(
    model_name=MODEL_NAME, dtype=torch.bfloat16, drop_vision=True, device_map="auto")
device = "cuda" if torch.cuda.is_available() else "cpu"
print_vram("base model loaded")

print("[3] Loading retriever")
from src.retriever import Retriever
retriever = Retriever(TRAIN_DS)
retriever.build()

ablation_results = []
all_per_sample_rows: list[dict] = []  # for step4 McNemar

for cfg in CONFIGS_TO_RERUN:
    cfg_id   = cfg["id"]
    cfg_name = cfg["name"]
    print(f"\n{'='*60}")
    print(f"Config {cfg_id}: {cfg['desc']}")
    print(f"{'='*60}")

    if cfg["lora_ckpt"] and Path(cfg["lora_ckpt"]).exists():
        model = PeftModel.from_pretrained(base_model, cfg["lora_ckpt"], is_trainable=False)
    else:
        model = base_model
    model.eval()
    print_vram(f"Config {cfg_id} model loaded")

    # Pre-gather samples
    samples_list = list(eval_ds)
    all_questions, all_gts, all_subjects, all_few_shots = [], [], [], []
    for sample in samples_list:
        q    = sample["prompt"][-1]["content"] if sample.get("prompt") else sample.get("question", "")
        gt   = str(sample.get("answer", ""))
        subj = str(sample.get("type", ""))
        fs   = ""
        try:
            examples = retriever.retrieve(q, top_k=3, subject=subj or None)
            if examples:
                lines = ["\n--- Reference examples ---"]
                for i, ex in enumerate(examples, 1):
                    lines.append(f"\nExample {i}:\n{ex['question'][:300]}\n<answer>\n{ex['answer']}\n</answer>")
                lines.append("\n--- Now answer the following ---\n")
                fs = "\n".join(lines)
        except Exception:
            pass
        all_questions.append(q)
        all_gts.append(gt)
        all_subjects.append(subj)
        all_few_shots.append(fs)

    t_start = time.time()
    correct_arr  = [False] * N
    raw_outputs  = [None]  * N
    retry_counts = [0]     * N
    resolved     = [False] * N

    for attempt in range(_MAX_RETRIES + 1):
        pending = list(range(N)) if attempt == 0 else [i for i in range(N) if not resolved[i]]
        if not pending:
            break
        print(f"  Attempt {attempt}: {len(pending)} pending", flush=True)
        temperature = 0.1 if attempt == 0 else 0.7
        fs_list = all_few_shots if attempt == 0 else [""] * N
        raw_list = []
        for bs in range(0, len(pending), EVAL_BATCH_SIZE):
            batch_i  = pending[bs: bs + EVAL_BATCH_SIZE]
            raws = _generate_batch(
                model, tokenizer,
                [all_questions[i] for i in batch_i],
                [temperature] * len(batch_i), device,
                [all_subjects[i] for i in batch_i],
                [fs_list[i] for i in batch_i],
            )
            raw_list.extend(raws)
        for pos, i in enumerate(pending):
            raw  = raw_list[pos]
            pred = extract_answer_from_text(raw)
            ok   = verify_answer(pred, all_gts[i], subject=all_subjects[i],
                                 question_text=all_questions[i], use_z3=True)
            raw_outputs[i]  = raw
            correct_arr[i]  = ok
            retry_counts[i] = attempt
            if ok or attempt == _MAX_RETRIES:
                resolved[i] = True
        n_c = sum(correct_arr)
        print(f"  Attempt {attempt}: correct={n_c}/{N} acc={n_c/N*100:.2f}%", flush=True)

    # ── Build metrics — FIX: count from actual subjects ──────────────────────
    correct_all = correct_phys = correct_logic = 0
    n_phys = n_logic = 0
    per_retry = {0: 0, 1: 0, 2: 0}

    for i in range(N):
        subj = all_subjects[i]  # FIX: use actual subject from sample
        if correct_arr[i]:
            correct_all += 1
        if subj == "physics":
            n_phys += 1
            if correct_arr[i]:
                correct_phys += 1
        elif subj == "logic":
            n_logic += 1
            if correct_arr[i]:
                correct_logic += 1
        per_retry[min(retry_counts[i], 2)] = per_retry.get(min(retry_counts[i], 2), 0) + 1

    # Sanity check
    assert n_phys + n_logic == N, f"BUG: n_phys({n_phys})+n_logic({n_logic}) != N({N})"

    elapsed = round((time.time() - t_start) / 60, 1)
    acc_all = correct_all / N * 100
    acc_phy = correct_phys / n_phys * 100 if n_phys else 0
    acc_log = correct_logic / n_logic * 100 if n_logic else 0

    result = {
        "config_id":        cfg_id,
        "config_name":      cfg_name,
        "description":      cfg["desc"],
        "expected_acc":     cfg["expected_acc"],
        "accuracy_overall": round(acc_all, 2),
        "accuracy_physics": round(acc_phy, 2),
        "accuracy_logic":   round(acc_log, 2),
        "n_total":          N,
        "n_physics":        n_phys,    # FIX: actual count
        "n_logic":          n_logic,   # FIX: actual count
        "n_correct":        correct_all,
        "per_retry":        per_retry,
        "elapsed_min":      elapsed,
        "smoke_test":       args.smoke_test,
        "bug_fix_v2":       True,      # marker for v2
    }
    ablation_results.append(result)

    # ── Per-sample rows for step4 McNemar test ────────────────────────────────
    for i in range(N):
        all_per_sample_rows.append({
            "config_id":    cfg_id,
            "config_name":  cfg_name,
            "correct":      bool(correct_arr[i]),
            "subject":      all_subjects[i],
            "question":     all_questions[i][:200],
            "prediction":   extract_answer_from_text(raw_outputs[i]) if raw_outputs[i] else "",
            "ground_truth": all_gts[i],
            "retry_count":  retry_counts[i],
        })

    print(f"\n  Config {cfg_id} DONE: overall={acc_all:.2f}%  phys={acc_phy:.2f}%  logic={acc_log:.2f}%")
    print(f"  n_phys={n_phys}, n_logic={n_logic}, n_total={N}  [sum check: {n_phys+n_logic}=={N}]")

    # ── Mid-run checkpoint: save after each config in case of crash ───────────
    _checkpoint = {
        "ablation": ablation_results + existing_cfg45,
        "version": "v2_bugfix_partial",
        "run_timestamp": _RUN_TS,
        "completed_configs": [r["config_id"] for r in ablation_results],
    }
    _ckpt_file = Path(LOG_DIR) / f"step1_checkpoint_{_RUN_TS}.json"
    with open(_ckpt_file, "w") as _f:
        json.dump(_checkpoint, _f, indent=2)
    print(f"  [checkpoint saved: {_ckpt_file.name}]", flush=True)

    if cfg_id < len(CONFIGS_TO_RERUN) - 1:
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Merge với cfg 4-5
all_results = ablation_results + existing_cfg45
all_results.sort(key=lambda x: x["config_id"])

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
payload = {"ablation": all_results, "version": "v2_bugfix", "run_timestamp": _RUN_TS}

# 1) Timestamped file — never overwritten
with open(LOG_OUT, "w") as f:
    json.dump(payload, f, indent=2)

# 2) Latest symlink (overwrite OK — always points to newest run)
with open(LOG_OUT_LATEST, "w") as f:
    json.dump(payload, f, indent=2)

# 3) Per-sample JSONL for McNemar test (step4)
with open(PER_SAMPLE_OUT, "w") as f:
    for row in all_per_sample_rows:
        f.write(json.dumps(row) + "\n")
with open(PER_SAMPLE_LATEST, "w") as f:
    for row in all_per_sample_rows:
        f.write(json.dumps(row) + "\n")

print(f"\nSaved (timestamped) : {LOG_OUT}")
print(f"Saved (latest copy) : {LOG_OUT_LATEST}")
print(f"Saved per-sample    : {PER_SAMPLE_OUT}")
print(f"Saved per-sample lat: {PER_SAMPLE_LATEST}")
print("SANITY CHECK:")
for r in all_results:
    total = r["n_physics"] + r["n_logic"]
    ok = "OK" if total == r["n_total"] else f"BUG (sum={total}!={r['n_total']})"
    print(f"  cfg{r['config_id']} {r['config_name']}: n_phys={r['n_physics']}, "
          f"n_logic={r['n_logic']}, n_total={r['n_total']} → {ok}")
