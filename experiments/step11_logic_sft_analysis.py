"""
BƯỚC 11: Logic SFT Regression Analysis
========================================
Phân tích tại sao cfg2 (Logic SFT) giảm 0.92pp so với cfg1 (SFT Ph1).

Part A (không cần GPU):
  - Phân tích per-sample: samples nào cfg1 đúng nhưng cfg2 sai?
  - Pattern: physics regression (catastrophic forgetting) hay logic không cải thiện?
  - Output: analysis JSON

Part B (cần GPU, ~$1):
  - Re-run Logic SFT với LR=5e-5 (nhẹ hơn, ít catastrophic forgetting hơn)
  - Eval ngay sau training
  - So sánh với cfg2 gốc (LR=1e-4)

Chạy:
    python experiments/step11_logic_sft_analysis.py          # Part A only (no GPU)
    python experiments/step11_logic_sft_analysis.py --run-b  # Part A + B (needs GPU)
    python experiments/step11_logic_sft_analysis.py --smoke  # smoke test
"""
from __future__ import annotations
import sys, os, json, argparse
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--run-b", action="store_true", help="Run Part B (GPU training)")
parser.add_argument("--smoke", action="store_true")
args, _ = parser.parse_known_args()

from datetime import datetime
_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
def _ts(): return datetime.now().strftime("%H:%M:%S")

from src.config import LOG_DIR, CKPT_DIR, QWEN35_SFT_FINAL, VAL_DS, TRAIN_DS
from pathlib import Path

LOG_OUT        = Path(LOG_DIR) / f"logic_sft_analysis_{_RUN_TS}.json"
LOG_OUT_LATEST = Path(LOG_DIR) / "logic_sft_analysis_latest.json"


# ══════════════════════════════════════════════════════════════════════════════
# PART A: Per-sample regression analysis (no GPU)
# ══════════════════════════════════════════════════════════════════════════════

print(f"[{_ts()}] ═══ PART A: Per-sample regression analysis ═══")

# Load ablation per-sample data (already have from previous runs)
PER_SAMPLE_FILE = Path(LOG_DIR) / "ablation_per_sample_latest.jsonl"

if not PER_SAMPLE_FILE.exists():
    print(f"[{_ts()}] ERROR: {PER_SAMPLE_FILE} not found")
    print(f"[{_ts()}] Need to run step1_rerun_cfg0_3.py first to generate per-sample data")
    sys.exit(1)

with open(PER_SAMPLE_FILE) as f:
    per_sample = [json.loads(l) for l in f if l.strip()]

print(f"[{_ts()}] Loaded {len(per_sample)} per-sample predictions")
print(f"[{_ts()}] Configs: {set(s['config_id'] for s in per_sample)}")

# Group by config
by_config = {}
for s in per_sample:
    cid = s["config_id"]
    if cid not in by_config:
        by_config[cid] = {}
    by_config[cid][s.get("question_id", s.get("question", "")[:50])] = s

cfg1 = by_config.get(1, {})  # SFT Phase 1
cfg2 = by_config.get(2, {})  # + Logic SFT

if not cfg1 or not cfg2:
    print(f"[{_ts()}] ERROR: config 1 or 2 not found in per-sample data")
    print(f"[{_ts()}] Available configs: {list(by_config.keys())}")
    sys.exit(1)

# Find regression cases: cfg1 correct, cfg2 wrong
common_keys = set(cfg1.keys()) & set(cfg2.keys())
print(f"[{_ts()}] Common samples: {len(common_keys)}")

regressions  = []   # cfg1 correct, cfg2 wrong → regression
improvements = []   # cfg1 wrong, cfg2 correct → improvement
both_correct  = []
both_wrong    = []

for key in common_keys:
    s1 = cfg1[key]
    s2 = cfg2[key]
    c1 = s1.get("correct", False)
    c2 = s2.get("correct", False)
    subj = s1.get("subject", "unknown")
    entry = {
        "question":   s1.get("question", "")[:100],
        "subject":    subj,
        "pred_cfg1":  s1.get("prediction", "")[:80],
        "pred_cfg2":  s2.get("prediction", "")[:80],
        "gt":         s1.get("ground_truth", ""),
        "cfg1_ok":    c1,
        "cfg2_ok":    c2,
    }
    if c1 and not c2:
        regressions.append(entry)
    elif not c1 and c2:
        improvements.append(entry)
    elif c1 and c2:
        both_correct.append(entry)
    else:
        both_wrong.append(entry)

print(f"\n[{_ts()}] cfg1 → cfg2 per-sample analysis ({len(common_keys)} samples):")
print(f"[{_ts()}]   Both correct:  {len(both_correct)}")
print(f"[{_ts()}]   Both wrong:    {len(both_wrong)}")
print(f"[{_ts()}]   Regression:    {len(regressions)}  (cfg1 correct → cfg2 wrong)")
print(f"[{_ts()}]   Improvement:   {len(improvements)}  (cfg1 wrong  → cfg2 correct)")
print(f"[{_ts()}]   Net change:    {len(improvements) - len(regressions):+d}")

# Subject breakdown of regressions
reg_by_subj = {}
for r in regressions:
    subj = r["subject"]
    reg_by_subj[subj] = reg_by_subj.get(subj, 0) + 1

imp_by_subj = {}
for r in improvements:
    subj = r["subject"]
    imp_by_subj[subj] = imp_by_subj.get(subj, 0) + 1

print(f"\n[{_ts()}] Regressions by subject: {reg_by_subj}")
print(f"[{_ts()}] Improvements by subject: {imp_by_subj}")

# Show top regression examples
print(f"\n[{_ts()}] === Top regression examples (cfg1 correct, cfg2 wrong) ===")
for r in regressions[:8]:
    print(f"  subj={r['subject']:8} | gt='{r['gt'][:20]}'")
    print(f"    cfg1 pred: '{r['pred_cfg1'][:60]}'")
    print(f"    cfg2 pred: '{r['pred_cfg2'][:60]}'")
    print()

# Interpretation
print(f"[{_ts()}] === INTERPRETATION ===")
phys_reg  = reg_by_subj.get("physics", 0)
logic_reg = reg_by_subj.get("logic",   0)
phys_imp  = imp_by_subj.get("physics", 0)
logic_imp = imp_by_subj.get("logic",   0)

if phys_reg > logic_reg:
    print(f"[{_ts()}] FINDING: Logic SFT causes PHYSICS REGRESSION ({phys_reg} physics vs {logic_reg} logic regressions)")
    print(f"[{_ts()}] → Catastrophic forgetting: focused logic training hurts physics capability")
    finding = "catastrophic_forgetting_physics"
elif logic_imp > 0 and logic_reg < logic_imp:
    print(f"[{_ts()}] FINDING: Logic SFT HELPS logic (+{logic_imp} improvements, -{logic_reg} regressions)")
    print(f"[{_ts()}] → Net logic improvement, but overall accuracy drops due to physics regression")
    finding = "helps_logic_hurts_physics"
else:
    print(f"[{_ts()}] FINDING: Mixed results — needs deeper inspection")
    finding = "mixed"

analysis_a = {
    "n_common":        len(common_keys),
    "regressions":     len(regressions),
    "improvements":    len(improvements),
    "both_correct":    len(both_correct),
    "both_wrong":      len(both_wrong),
    "reg_by_subject":  reg_by_subj,
    "imp_by_subject":  imp_by_subj,
    "finding":         finding,
    "regression_examples": regressions[:10],
    "improvement_examples": improvements[:10],
}


# ══════════════════════════════════════════════════════════════════════════════
# PART B: Re-run Logic SFT with lower LR (optional, needs GPU)
# ══════════════════════════════════════════════════════════════════════════════

analysis_b = None

if args.run_b:
    print(f"\n[{_ts()}] ═══ PART B: Logic SFT with lower LR ═══")
    import time, gc
    import torch
    from datasets import load_from_disk
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, PeftModel
    from trl import SFTTrainer
    from src.config import (
        MODEL_NAME, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS,
        MAX_SEQ_LEN, MAX_NEW_TOKENS
    )
    from src.symbolic_verifier import verify_answer, extract_answer_from_text

    SYSTEM_PROMPT = (
        "You are an expert in Logic and Physics. "
        "Think step by step and respond in the exact format:\n"
        "<reasoning>\n[Your step-by-step reasoning]\n</reasoning>\n"
        "<answer>\n[Final answer: letter / Yes/No/Unknown / number+unit]\n</answer>\n"
        "<explanation>\n[Concise explanation]\n</explanation>"
    )

    # Load train data (logic only) and val data
    train_ds_full = load_from_disk(TRAIN_DS)
    val_ds        = load_from_disk(VAL_DS)
    logic_train   = [s for s in train_ds_full if s.get("subject") == "logic"]
    val_samples   = list(val_ds)
    if args.smoke:
        logic_train = logic_train[:20]
        val_samples = val_samples[:16]

    print(f"[{_ts()}] Logic train: {len(logic_train)} | Val: {len(val_samples)}")

    def _format_sample(sample):
        q = sample["question"]
        a = sample["answer"]
        return {
            "text": (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\nQuestion: {q}<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"<reasoning>\nLet me analyze this step by step.\n</reasoning>\n"
                f"<answer>\n{a}\n</answer>\n"
                f"<explanation>\nThis follows from the given premises.\n</explanation>"
                f"<|im_end|>"
            )
        }

    from datasets import Dataset
    logic_dataset = Dataset.from_list(
        [_format_sample(s) for s in logic_train])

    # Run with lower LR = 5e-5 (original was 1e-4)
    results_b = []
    for lr_tag, lr_val in [("lr_1e4_original", 1e-4), ("lr_5e5_gentle", 5e-5)]:
        print(f"\n[{_ts()}] Training Logic SFT with LR={lr_val} ({lr_tag})")
        t0 = time.time()

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, padding_side="right", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)

        # Start from SFT Phase 1 checkpoint
        model = PeftModel.from_pretrained(base, QWEN35_SFT_FINAL)
        model = model.merge_and_unload()

        from peft import get_peft_model
        lora_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_cfg)

        epochs = 1 if not args.smoke else 1
        sft_args = TrainingArguments(
            output_dir=str(Path(CKPT_DIR) / f"logic_sft_{lr_tag}"),
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=lr_val,
            bf16=True,
            save_strategy="no",
            logging_steps=20,
            lr_scheduler_type="cosine",
            report_to="none",
            max_steps=5 if args.smoke else -1,
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=logic_dataset,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
        )
        trainer.train()
        train_loss = trainer.state.log_history[-1].get("train_loss", None)
        elapsed_min = (time.time() - t0) / 60

        # Evaluate
        model.eval()
        tokenizer2 = AutoTokenizer.from_pretrained(
            MODEL_NAME, padding_side="left", trust_remote_code=True)
        if tokenizer2.pad_token is None:
            tokenizer2.pad_token = tokenizer2.eos_token

        n_correct = n_phys = n_logic = n_phys_c = n_logic_c = 0
        BATCH = 8
        for i in range(0, len(val_samples), BATCH):
            batch = val_samples[i:i+BATCH]
            prompts = []
            for s in batch:
                msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Question: {s['question']}"},
                ]
                try:
                    p = tokenizer2.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                except Exception:
                    p = f"{SYSTEM_PROMPT}\n\nQuestion: {s['question']}\n\nResponse:"
                prompts.append(p)
            inputs = tokenizer2(prompts, return_tensors="pt", padding=True,
                                truncation=True, max_length=1024).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False,
                                     pad_token_id=tokenizer2.pad_token_id)
            for j, (s, inp_ids) in enumerate(zip(batch, inputs["input_ids"])):
                gen_ids = out[j][len(inp_ids):]
                raw  = tokenizer2.decode(gen_ids, skip_special_tokens=True)
                pred = extract_answer_from_text(raw)
                ok   = verify_answer(pred, s["answer"], subject=s.get("subject",""),
                                     question_text=s["question"])
                n_correct += int(ok)
                subj = s.get("subject","")
                if subj == "physics":
                    n_phys += 1; n_phys_c += int(ok)
                elif subj == "logic":
                    n_logic += 1; n_logic_c += int(ok)

        acc_overall = round(n_correct / len(val_samples) * 100, 2)
        acc_physics = round(n_phys_c / max(n_phys,1) * 100, 2)
        acc_logic   = round(n_logic_c / max(n_logic,1) * 100, 2)

        results_b.append({
            "lr_tag":       lr_tag,
            "lr":           lr_val,
            "train_loss":   train_loss,
            "elapsed_min":  round(elapsed_min, 1),
            "acc_overall":  acc_overall,
            "acc_physics":  acc_physics,
            "acc_logic":    acc_logic,
        })
        print(f"[{_ts()}] {lr_tag}: overall={acc_overall}%  "
              f"phys={acc_physics}%  logic={acc_logic}%  loss={train_loss}")

        del model, base, trainer
        gc.collect(); torch.cuda.empty_cache()

    analysis_b = {
        "results": results_b,
        "finding": (
            "gentle_lr_better"
            if results_b[1]["acc_overall"] > results_b[0]["acc_overall"]
            else "original_lr_comparable"
        )
    }

    print(f"\n[{_ts()}] === PART B SUMMARY ===")
    print(f"[{_ts()}] Original LR=1e-4: overall={results_b[0]['acc_overall']}%  phys={results_b[0]['acc_physics']}%  logic={results_b[0]['acc_logic']}%")
    print(f"[{_ts()}] Gentle  LR=5e-5: overall={results_b[1]['acc_overall']}%  phys={results_b[1]['acc_physics']}%  logic={results_b[1]['acc_logic']}%")
    delta = results_b[1]["acc_overall"] - results_b[0]["acc_overall"]
    print(f"[{_ts()}] Delta: {delta:+.2f}pp  → {'LR=5e-5 is better (less forgetting)' if delta > 0 else 'LR=1e-4 comparable'}")


# ── Save results ──────────────────────────────────────────────────────────────

out = {
    "run_ts":     _RUN_TS,
    "part_a":     analysis_a,
    "part_b":     analysis_b,
    "cfg1_acc":   52.53,  # from existing results
    "cfg2_acc":   51.61,  # from existing results
    "delta":      -0.92,
}
with open(LOG_OUT, "w") as f: json.dump(out, f, indent=2)
with open(LOG_OUT_LATEST, "w") as f: json.dump(out, f, indent=2)
print(f"[{_ts()}] Saved → {LOG_OUT_LATEST}")

print(f"\n[{_ts()}] ═══ PAPER NOTES ═══")
print(f"[{_ts()}] Use Part A findings to explain cfg2 regression in Section 4.3:")
phys_reg  = analysis_a["reg_by_subject"].get("physics", 0)
logic_imp = analysis_a["imp_by_subject"].get("logic",   0)
print(f"[{_ts()}]   Physics regressions: {phys_reg}")
print(f"[{_ts()}]   Logic improvements:  {logic_imp}")
print(f"[{_ts()}]   Finding: '{analysis_a['finding']}'")
