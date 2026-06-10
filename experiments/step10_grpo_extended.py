"""
BƯỚC 10: Extended GRPO Training — Convergence Curve
====================================================
Chạy R1 (full reward) và R4 (correctness-only) lên 500 steps.
Evaluate tại: 50, 100, 150, 200, 250, 300, 400, 500 steps.
Vẽ convergence curve để trả lời câu hỏi:
  "Tại budget dài hơn, full reward (R1) có vượt correctness-only (R4) không?"

Tại sao cần:
  - Reviewer: "150 steps quá ngắn để kết luận về reward shaping"
  - Current finding: R4 > R1 tại 150 steps
  - Extended finding: tìm crossover point (hoặc confirm R4 luôn tốt hơn)

Chạy:
    python experiments/step10_grpo_extended.py
    python experiments/step10_grpo_extended.py --smoke-steps 5 --smoke-eval 16
    python experiments/step10_grpo_extended.py --only-variants R4   # chạy từng variant
    python experiments/step10_grpo_extended.py --resume             # tiếp tục nếu bị ngắt
"""
from __future__ import annotations
import sys, os, json, time, gc, argparse
from datetime import datetime
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-steps", type=int, default=0)
parser.add_argument("--smoke-eval",  type=int, default=0)
parser.add_argument("--only-variants", nargs="*", default=None,
                    help="e.g. R1 R4")
parser.add_argument("--resume",      action="store_true",
                    help="Resume from existing partial results")
args, _ = parser.parse_known_args()

from src.config import (
    MODEL_NAME, QWEN35_SFT_FINAL, VAL_DS, TRAIN_DS, LOG_DIR, CKPT_DIR,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS,
    MAX_SEQ_LEN, MAX_NEW_TOKENS, GRPO_LR, GRPO_BETA, GRPO_NUM_GEN, GRPO_MAX_COMP,
)
from src.symbolic_verifier import verify_answer, extract_answer_from_text
import torch
from pathlib import Path

_RUN_TS        = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_OUT        = Path(LOG_DIR) / f"grpo_extended_{_RUN_TS}.json"
LOG_OUT_LATEST = Path(LOG_DIR) / "grpo_extended_results_latest.json"
def _ts(): return datetime.now().strftime("%H:%M:%S")
def _vram():
    try:
        if torch.cuda.is_available():
            used = torch.cuda.memory_reserved() / 1024**3
            tot  = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"VRAM {used:.1f}/{tot:.0f}GB"
    except Exception: pass
    return ""

TOTAL_STEPS  = 10 if args.smoke_steps else 500
EVAL_AT      = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500] if not args.smoke_steps \
               else [5, 10]
EVAL_N       = args.smoke_eval if args.smoke_eval else 217
SMOKE        = bool(args.smoke_steps or args.smoke_eval)

SYSTEM_PROMPT = (
    "You are an expert in Logic and Physics. "
    "Think step by step and respond in the exact format:\n"
    "<reasoning>\n[Your step-by-step reasoning]\n</reasoning>\n"
    "<answer>\n[Final answer: letter / Yes/No/Unknown / number+unit]\n</answer>\n"
    "<explanation>\n[Concise explanation]\n</explanation>"
)

# ── Reward variants ───────────────────────────────────────────────────────────

REWARD_VARIANTS = [
    {
        "id":         "R1",
        "name":       "full_reward",
        "desc":       "Full: format(0.30) + correct(0.60) + unit(0.10) + len_pen(-0.10)",
        "format_w":   0.30, "correct_w": 0.60, "unit_w": 0.10, "len_pen": -0.10,
    },
    {
        "id":         "R4",
        "name":       "correct_only",
        "desc":       "Correctness only: correct(1.0)",
        "format_w":   0.00, "correct_w": 1.00, "unit_w": 0.00, "len_pen":  0.00,
    },
]

if args.only_variants:
    REWARD_VARIANTS = [r for r in REWARD_VARIANTS if r["id"] in args.only_variants]

print(f"[{_ts()}] Extended GRPO: {TOTAL_STEPS} steps per variant")
print(f"[{_ts()}] Variants: {[r['id'] for r in REWARD_VARIANTS]}")
print(f"[{_ts()}] Evaluate at steps: {EVAL_AT}")

# ── Load existing results (resume) ────────────────────────────────────────────

existing_results = {}
if args.resume and LOG_OUT_LATEST.exists():
    with open(LOG_OUT_LATEST) as f:
        saved = json.load(f)
    for v in saved.get("variants", []):
        if len(v.get("checkpoints", [])) > 0:
            existing_results[v["id"]] = v
            print(f"[{_ts()}] Resuming {v['id']}: {len(v['checkpoints'])} checkpoints found")

# ── Load data ─────────────────────────────────────────────────────────────────

from datasets import load_from_disk
train_ds = load_from_disk(TRAIN_DS)
val_ds   = load_from_disk(VAL_DS)
train_samples = list(train_ds)
val_samples   = list(val_ds)[:EVAL_N]

print(f"[{_ts()}] Train: {len(train_samples)} | Val: {len(val_samples)}")


# ── Reward function ───────────────────────────────────────────────────────────

def make_reward_fn(format_w: float, correct_w: float, unit_w: float, len_pen: float):
    """Build reward function for GRPO training."""
    from src.symbolic_verifier import verify_answer, same_unit_scale

    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            if isinstance(completion, list):
                completion = completion[0] if completion else ""
            y = completion if isinstance(completion, str) else ""

            # Extract answer & metadata from prompt
            meta = kwargs.get("meta", {})
            gt      = meta.get("answer", "")
            subject = meta.get("subject", "")
            q_text  = meta.get("question", "")

            # Format reward
            r_fmt = 0.0
            if format_w > 0:
                tags = ["<reasoning>", "<answer>", "<explanation>"]
                r_fmt = format_w * sum(t in y for t in tags) / len(tags)

            # Correctness reward
            pred = extract_answer_from_text(y) if y else ""
            correct = verify_answer(pred, gt, subject=subject,
                                    question_text=q_text) if pred and gt else False
            r_correct = correct_w * float(correct)

            # Unit reward
            r_unit = 0.0
            if unit_w > 0 and subject == "physics" and gt:
                try:
                    r_unit = unit_w * float(same_unit_scale(pred, gt))
                except Exception:
                    pass

            # Length penalty
            r_len = 0.0
            if len_pen < 0:
                try:
                    from transformers import AutoTokenizer
                    n_tok = len(y.split())  # rough token count
                    if n_tok > 800:
                        r_len = len_pen
                except Exception:
                    pass

            rewards.append(r_fmt + r_correct + r_unit + r_len)
        return rewards

    return reward_fn


# ── Evaluate at a checkpoint ──────────────────────────────────────────────────

def evaluate(model, tokenizer, step: int) -> dict:
    """Run evaluation on val_samples. Returns accuracy dict."""
    model.eval()
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
                p = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                p = f"{SYSTEM_PROMPT}\n\nQuestion: {s['question']}\n\nResponse:"
            prompts.append(p)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, temperature=1.0,
                                 pad_token_id=tokenizer.pad_token_id)

        for j, (s, inp_ids) in enumerate(zip(batch, inputs["input_ids"])):
            gen_ids = out[j][len(inp_ids):]
            raw  = tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred = extract_answer_from_text(raw)
            ok   = verify_answer(pred, s["answer"], subject=s.get("subject",""),
                                 question_text=s["question"])
            n_correct += int(ok)
            subj = s.get("subject","")
            if subj == "physics":
                n_phys += 1; n_phys_c += int(ok)
            elif subj == "logic":
                n_logic += 1; n_logic_c += int(ok)

    model.train()
    return {
        "step":             step,
        "acc_overall":      round(n_correct / len(val_samples) * 100, 2),
        "acc_physics":      round(n_phys_c  / max(n_phys,  1) * 100, 2),
        "acc_logic":        round(n_logic_c / max(n_logic, 1) * 100, 2),
        "n_correct":        n_correct,
        "n_total":          len(val_samples),
    }


# ── Training loop ─────────────────────────────────────────────────────────────

all_variant_results = []

for variant in REWARD_VARIANTS:
    vid = variant["id"]

    if vid in existing_results:
        print(f"[{_ts()}] {vid}: skipping (resume mode, already have results)")
        all_variant_results.append(existing_results[vid])
        continue

    print(f"\n[{_ts()}] {'='*60}")
    print(f"[{_ts()}] VARIANT {vid}: {variant['desc']}")
    print(f"[{_ts()}] {'='*60}")

    # ── Load base model (SFT Phase 1 checkpoint) ────────────────────────────
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel

    print(f"[{_ts()}] Loading SFT Phase 1 checkpoint: {QWEN35_SFT_FINAL}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)

    model = PeftModel.from_pretrained(base, QWEN35_SFT_FINAL)
    model = model.merge_and_unload()

    # Add new LoRA for GRPO
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    print(f"[{_ts()}] {_vram()}")

    # ── GRPO setup ──────────────────────────────────────────────────────────
    from trl import GRPOConfig, GRPOTrainer

    reward_fn = make_reward_fn(
        variant["format_w"], variant["correct_w"],
        variant["unit_w"],   variant["len_pen"],
    )

    def _format_prompt(sample):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Question: {sample['question']}"},
        ]
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"{SYSTEM_PROMPT}\n\nQuestion: {sample['question']}\n\nResponse:"

    # Prepare dataset with prompt field
    from datasets import Dataset
    def _add_prompt(sample):
        sample["prompt"] = _format_prompt(sample)
        return sample
    train_dataset = Dataset.from_list(train_samples).map(_add_prompt)

    grpo_cfg = GRPOConfig(
        output_dir=str(Path(CKPT_DIR) / f"grpo_extended_{vid}"),
        max_steps=TOTAL_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=GRPO_LR,
        bf16=True,
        save_strategy="no",          # We save manually at eval checkpoints
        logging_steps=10,
        num_generations=GRPO_NUM_GEN,
        max_completion_length=GRPO_MAX_COMP,
        temperature=1.0,
        beta=GRPO_BETA,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=train_dataset,
        reward_funcs=reward_fn,
    )

    # ── Train with intermediate evals ───────────────────────────────────────
    checkpoints = []
    prev_step = 0
    t_start = time.time()

    for eval_step in EVAL_AT:
        steps_to_run = eval_step - prev_step
        if steps_to_run <= 0:
            continue

        print(f"[{_ts()}] Training {vid}: steps {prev_step+1}→{eval_step} ...")
        trainer.args.max_steps = eval_step
        trainer.train(resume_from_checkpoint=(prev_step > 0))

        # Evaluate
        print(f"[{_ts()}] Evaluating at step {eval_step} ...")
        acc = evaluate(model, tokenizer, eval_step)
        elapsed = (time.time() - t_start) / 60
        acc["elapsed_min"] = round(elapsed, 1)
        checkpoints.append(acc)

        print(f"[{_ts()}]  Step {eval_step:>4} | "
              f"overall={acc['acc_overall']}% "
              f"phys={acc['acc_physics']}% "
              f"logic={acc['acc_logic']}%  "
              f"({elapsed:.1f}min)")
        prev_step = eval_step

        # Save incrementally
        variant_result = {
            "id":          vid,
            "name":        variant["name"],
            "desc":        variant["desc"],
            "checkpoints": checkpoints,
        }
        out = {
            "run_ts":   _RUN_TS,
            "smoke":    SMOKE,
            "variants": all_variant_results + [variant_result],
        }
        with open(LOG_OUT, "w") as f: json.dump(out, f, indent=2)
        with open(LOG_OUT_LATEST, "w") as f: json.dump(out, f, indent=2)

    all_variant_results.append({
        "id":          vid,
        "name":        variant["name"],
        "desc":        variant["desc"],
        "checkpoints": checkpoints,
    })

    # Cleanup
    del model, base, trainer
    gc.collect(); torch.cuda.empty_cache()
    print(f"[{_ts()}] {vid} done. GPU cleared.")


# ── Final summary ─────────────────────────────────────────────────────────────

print(f"\n[{_ts()}] ╔{'═'*70}╗")
print(f"[{_ts()}] ║ EXTENDED GRPO CONVERGENCE SUMMARY{' '*34}║")
print(f"[{_ts()}] ╠{'═'*70}╣")
print(f"[{_ts()}] ║ {'Variant':<8} {'Steps':>6} {'Overall':>8} {'Physics':>8} {'Logic':>8}{' '*29}║")
print(f"[{_ts()}] ╠{'═'*70}╣")

for v in all_variant_results:
    for ck in v["checkpoints"]:
        print(f"[{_ts()}] ║ {v['id']:<8} {ck['step']:>6} "
              f"{ck['acc_overall']:>7.2f}% {ck['acc_physics']:>7.2f}% "
              f"{ck['acc_logic']:>7.2f}%{' '*29}║")
    print(f"[{_ts()}] ╠{'═'*70}╣")

print(f"[{_ts()}] ╚{'═'*70}╝")

# Crossover analysis
if len(all_variant_results) == 2:
    r1 = {ck["step"]: ck["acc_overall"] for ck in all_variant_results[0]["checkpoints"]}
    r4 = {ck["step"]: ck["acc_overall"] for ck in all_variant_results[1]["checkpoints"]}
    crossover = None
    for step in sorted(r1.keys()):
        if step in r4 and r1[step] > r4[step]:
            crossover = step
            break
    if crossover:
        print(f"[{_ts()}] Crossover found: R1 overtakes R4 at step {crossover}")
        print(f"[{_ts()}] → Paper claim: 'full reward becomes beneficial after ~{crossover} steps'")
    else:
        print(f"[{_ts()}] No crossover: R4 (correctness-only) remains ≥ R1 through 500 steps")
        print(f"[{_ts()}] → Paper claim: 'correctness-only reward consistently better within 500-step budget'")

print(f"[{_ts()}] Saved: {LOG_OUT_LATEST}")
