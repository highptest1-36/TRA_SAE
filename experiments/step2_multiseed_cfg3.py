"""
BƯỚC 2: Multi-seed evaluation cho cfg 3 (grpo_mixed)
=====================================================
Chạy cfg 3 với 3 seeds (42, 1337, 2024) để có mean ± std.
Seed ảnh hưởng đến: thứ tự sample, generation temperature sampling.

Chạy:
    python experiments/step2_multiseed_cfg3.py
    python experiments/step2_multiseed_cfg3.py --smoke-test
"""
from __future__ import annotations
import sys, os, json, time, gc, argparse
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true")
args, _ = parser.parse_known_args()

from src.config import (
    MODEL_NAME, QWEN35_GRPO_FINAL, VAL_DS, LOG_DIR, TRAIN_DS,
    MAX_NEW_TOKENS, AGENT_MAX_RETRIES,
)
from src.utils import setup_logger, print_vram
from src.symbolic_verifier import verify_answer, extract_answer_from_text
import torch, numpy as np
from pathlib import Path

from datetime import datetime
_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

SEEDS           = [42, 1337, 2024]
EVAL_BATCH_SIZE = 8
_MAX_RETRIES    = AGENT_MAX_RETRIES
_SMOKE_N        = 10
LOG_OUT         = Path(LOG_DIR) / f"cfg3_multiseed_{_RUN_TS}.json"
LOG_OUT_LATEST  = Path(LOG_DIR) / "cfg3_multiseed_results_latest.json"

SYSTEM_PROMPT = (
    "You are an expert in Logic and Physics. "
    "Think step by step and respond in the exact format:\n"
    "<reasoning>\n[Your step-by-step reasoning]\n</reasoning>\n"
    "<answer>\n[Final answer: letter / Yes/No/Unknown / number+unit]\n</answer>\n"
    "<explanation>\n[Concise explanation of why this answer is correct]\n</explanation>"
)


def _generate_batch(model, tokenizer, questions, temperatures, device, few_shots):
    texts = []
    for q, fs in zip(questions, few_shots):
        user_content = fs + q if fs else q
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
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
    temperature = temperatures[0]
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


print("[1] Loading base + LoRA model")
from src.model_loader import load_base_model
from peft import PeftModel
base_model, tokenizer = load_base_model(
    model_name=MODEL_NAME, dtype=torch.bfloat16, drop_vision=True, device_map="auto")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PeftModel.from_pretrained(base_model, QWEN35_GRPO_FINAL, is_trainable=False)
model.eval()
print_vram("grpo_mixed loaded")

print("[2] Loading dataset")
from datasets import load_from_disk
val_ds = load_from_disk(VAL_DS)
N_full = len(val_ds)
N      = _SMOKE_N if args.smoke_test else N_full

print("[3] Loading retriever")
from src.retriever import Retriever
retriever = Retriever(TRAIN_DS)
retriever.build()

seed_results = []

for seed in SEEDS:
    print(f"\n{'='*55}")
    print(f"Seed {seed}")
    print(f"{'='*55}")

    # Set seed for reproducibility of sampling
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    eval_ds = val_ds.select(range(N))
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
                for i_ex, ex in enumerate(examples, 1):
                    lines.append(f"\nExample {i_ex}:\n{ex['question'][:300]}\n<answer>\n{ex['answer']}\n</answer>")
                lines.append("\n--- Now answer the following ---\n")
                fs = "\n".join(lines)
        except Exception:
            pass
        all_questions.append(q)
        all_gts.append(gt)
        all_subjects.append(subj)
        all_few_shots.append(fs)

    t_start      = time.time()
    correct_arr  = [False] * N
    raw_outputs  = [None]  * N
    retry_counts = [0]     * N
    resolved     = [False] * N

    for attempt in range(_MAX_RETRIES + 1):
        pending = list(range(N)) if attempt == 0 else [i for i in range(N) if not resolved[i]]
        if not pending:
            break
        temperature = 0.1 if attempt == 0 else 0.7
        fs_list = all_few_shots if attempt == 0 else [""] * N
        raw_list = []
        n_batches = (len(pending) + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE
        for b_idx, bs in enumerate(range(0, len(pending), EVAL_BATCH_SIZE)):
            batch_i = pending[bs: bs + EVAL_BATCH_SIZE]
            raws = _generate_batch(
                model, tokenizer,
                [all_questions[i] for i in batch_i],
                [temperature] * len(batch_i), device,
                [fs_list[i] for i in batch_i],
            )
            raw_list.extend(raws)
            done = min(bs + EVAL_BATCH_SIZE, len(pending))
            print(f"  [seed={seed} attempt={attempt}] batch {b_idx+1}/{n_batches}  ({done}/{len(pending)} samples)", flush=True)
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
        print(f"  Attempt {attempt}: acc={n_c/N*100:.2f}%", flush=True)

    correct_all = correct_phys = correct_logic = 0
    n_phys = n_logic = 0
    for i in range(N):
        subj = all_subjects[i]
        if correct_arr[i]:
            correct_all += 1
        if subj == "physics":
            n_phys += 1
            if correct_arr[i]: correct_phys += 1
        elif subj == "logic":
            n_logic += 1
            if correct_arr[i]: correct_logic += 1

    elapsed = round((time.time() - t_start) / 60, 1)
    r = {
        "seed":             seed,
        "accuracy_overall": round(correct_all / N * 100, 2),
        "accuracy_physics": round(correct_phys / n_phys * 100, 2) if n_phys else 0,
        "accuracy_logic":   round(correct_logic / n_logic * 100, 2) if n_logic else 0,
        "n_total":          N,
        "n_physics":        n_phys,
        "n_logic":          n_logic,
        "n_correct":        correct_all,
        "elapsed_min":      elapsed,
    }
    seed_results.append(r)
    print(f"  Seed {seed}: overall={r['accuracy_overall']:.2f}%  phys={r['accuracy_physics']:.2f}%  logic={r['accuracy_logic']:.2f}%")

# Compute mean ± std
accs_all  = [r["accuracy_overall"] for r in seed_results]
accs_phys = [r["accuracy_physics"] for r in seed_results]
accs_log  = [r["accuracy_logic"]   for r in seed_results]

summary = {
    "config":               "grpo_mixed (cfg3)",
    "seeds":                SEEDS,
    "per_seed":             seed_results,
    "mean_overall":         round(float(np.mean(accs_all)), 2),
    "std_overall":          round(float(np.std(accs_all)), 2),
    "mean_physics":         round(float(np.mean(accs_phys)), 2),
    "std_physics":          round(float(np.std(accs_phys)), 2),
    "mean_logic":           round(float(np.mean(accs_log)), 2),
    "std_logic":            round(float(np.std(accs_log)), 2),
}

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
with open(LOG_OUT, "w") as f:
    json.dump(summary, f, indent=2)
with open(LOG_OUT_LATEST, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*55}")
print("MULTI-SEED SUMMARY (cfg3 grpo_mixed)")
print(f"{'='*55}")
print(f"  Overall : {summary['mean_overall']:.2f} ± {summary['std_overall']:.2f}%")
print(f"  Physics : {summary['mean_physics']:.2f} ± {summary['std_physics']:.2f}%")
print(f"  Logic   : {summary['mean_logic']:.2f} ± {summary['std_logic']:.2f}%")
print(f"\nSaved (timestamped): {LOG_OUT}")
print(f"Saved (latest)     : {LOG_OUT_LATEST}")
