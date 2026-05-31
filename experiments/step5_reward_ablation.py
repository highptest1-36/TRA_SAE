"""
BƯỚC 5: Reward component ablation
===================================
Rerun GRPO Phase 2 với 4 biến thể reward để đo đóng góp của từng component.

Variants:
    R1: FULL  — format=0.30, correct=0.60, unit=0.10, len_pen=-0.10  (default)
    R2: NO_UNIT — format=0.30, correct=0.70, no unit, no len_pen
    R3: NO_FORMAT — correct=0.80, unit=0.10, no format, no len_pen
    R4: CORRECT_ONLY — correct=1.0 (ablate everything else)

Chạy:
    python experiments/step5_reward_ablation.py
    python experiments/step5_reward_ablation.py --smoke-steps 10 --smoke-eval 8
    python experiments/step5_reward_ablation.py --only-variants R2 R3
"""
from __future__ import annotations
import sys, os, json, time, gc, argparse
from datetime import datetime
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-steps", type=int, default=0,
                    help="small number of GRPO steps for smoke testing (0=full)")
parser.add_argument("--smoke-eval",  type=int, default=0,
                    help="small number of eval samples (0=full)")
parser.add_argument("--only-variants", nargs="*",
                    help="subset e.g. R1 R2 R3 R4")
parser.add_argument("--ablation-steps", type=int, default=150,
                    help="GRPO steps for ablation run (default 150, overrides config)")
args, _ = parser.parse_known_args()

from src.config import (
    MODEL_NAME, QWEN35_SFT_FINAL, VAL_DS, TRAIN_DS, LOG_DIR, CKPT_DIR,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS, MAX_SEQ_LEN,
    MAX_NEW_TOKENS, GRPO_MAX_STEPS, GRPO_LR, GRPO_BETA, GRPO_NUM_GEN, GRPO_MAX_COMP,
)
from src.utils import setup_logger, print_vram
from src.symbolic_verifier import verify_answer, extract_answer_from_text
import torch
from pathlib import Path

_RUN_TS         = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_OUT         = Path(LOG_DIR) / f"reward_ablation_{_RUN_TS}.json"
LOG_OUT_LATEST  = Path(LOG_DIR) / "reward_ablation_results_latest.json"
EVAL_BATCH_SIZE = 8
if args.smoke_steps > 0:
    GRPO_STEPS_RUN = args.smoke_steps
else:
    GRPO_STEPS_RUN = args.ablation_steps   # default 150; pass --ablation-steps N to override
EVAL_N          = args.smoke_eval  if args.smoke_eval  > 0 else 217
SMOKE           = args.smoke_steps > 0 or args.smoke_eval > 0

SYSTEM_PROMPT = (
    "You are an expert in Logic and Physics. "
    "Think step by step and respond in the exact format:\n"
    "<reasoning>\n[Your step-by-step reasoning]\n</reasoning>\n"
    "<answer>\n[Final answer: letter / Yes/No/Unknown / number+unit]\n</answer>\n"
    "<explanation>\n[Concise explanation of why this answer is correct]\n</explanation>"
)

REWARD_VARIANTS = [
    {
        "id": "R1", "name": "full_reward",
        "desc": "Full: format(0.30) + correct(0.60) + unit(0.10) + len_pen(-0.10)",
        "format_w": 0.30, "correct_w": 0.60, "unit_w": 0.10, "len_pen": -0.10,
    },
    {
        "id": "R2", "name": "no_unit",
        "desc": "No unit reward: format(0.30) + correct(0.70), no unit, no len_pen",
        "format_w": 0.30, "correct_w": 0.70, "unit_w": 0.00, "len_pen": 0.0,
    },
    {
        "id": "R3", "name": "no_format",
        "desc": "No format reward: correct(0.80) + unit(0.10), no format, no len_pen",
        "format_w": 0.00, "correct_w": 0.80, "unit_w": 0.10, "len_pen": 0.0,
    },
    {
        "id": "R4", "name": "correct_only",
        "desc": "Correctness only: correct(1.0)",
        "format_w": 0.00, "correct_w": 1.00, "unit_w": 0.00, "len_pen": 0.0,
    },
]

if args.only_variants:
    REWARD_VARIANTS = [r for r in REWARD_VARIANTS if r["id"] in args.only_variants]


def _make_reward_fn(fmt_w: float, corr_w: float, unit_w: float, len_pen: float):
    """Factory: build a reward function with specific weights."""
    from src.reward import format_reward, correctness_reward, unit_reward, _extract_text

    def reward_fn(completions, prompts, answer=None, type=None, **kwargs):
        rewards = []
        for i, comp in enumerate(completions):
            # TRL ≥ 1.4 may pass completions as list[dict]; normalise to str
            text    = _extract_text(comp)
            gt      = answer[i] if answer else ""
            subject = type[i] if type else ""
            q_text  = prompts[i][-1]["content"] if prompts else ""
            r = 0.0
            if fmt_w > 0:
                r += format_reward(text) * fmt_w
            if corr_w > 0:
                r += correctness_reward(text, gt, subject, q_text) * corr_w
            if unit_w > 0:
                r += unit_reward(text, gt) * unit_w
            if len_pen < 0:
                import re
                reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>",
                                            text, re.DOTALL)
                if reasoning_match:
                    n_tok = len(reasoning_match.group(1).split())
                    if n_tok > 800:
                        r += len_pen
            rewards.append(r)
        return rewards

    return reward_fn


def _run_grpo_variant(variant: dict) -> dict:
    """Run full GRPO training + eval for one reward variant."""
    vid   = variant["id"]
    vname = variant["name"]
    print(f"\n{'='*60}")
    print(f"Reward variant {vid}: {variant['desc']}")
    print(f"GRPO steps: {GRPO_STEPS_RUN}")
    print(f"{'='*60}")

    ckpt_out = Path(CKPT_DIR) / f"grpo_reward_ablation_{vid}"

    # ── Load base SFT ────────────────────────────────────────────────────────
    from src.model_loader import load_base_model
    from peft import LoraConfig, get_peft_model, PeftModel
    model, tokenizer = load_base_model(
        model_name=MODEL_NAME, dtype=torch.bfloat16, drop_vision=True, device_map="auto")
    model = PeftModel.from_pretrained(model, QWEN35_SFT_FINAL, is_trainable=True)
    model.train()
    print_vram(f"{vid} SFT loaded")

    # ── Load train dataset ───────────────────────────────────────────────────
    from datasets import load_from_disk
    train_ds = load_from_disk(TRAIN_DS)
    # Filter physics + logic only
    train_ds = train_ds.filter(lambda x: x.get("type") in ["physics", "logic"])
    # Shuffle
    train_ds = train_ds.shuffle(seed=42)

    # ── Configure GRPO ───────────────────────────────────────────────────────
    from trl import GRPOConfig, GRPOTrainer

    grpo_cfg = GRPOConfig(
        max_completion_length=GRPO_MAX_COMP,
        learning_rate=GRPO_LR,
        beta=GRPO_BETA,
        num_generations=GRPO_NUM_GEN,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=GRPO_STEPS_RUN,
        output_dir=str(ckpt_out),
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,  # keep only latest checkpoint to save Drive space
        report_to="none",
        bf16=True,
        remove_unused_columns=False,
    )

    reward_fn = _make_reward_fn(
        variant["format_w"], variant["correct_w"],
        variant["unit_w"],   variant["len_pen"],
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=train_ds,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
    )

    # ── Within-variant resume: detect latest checkpoint from a previous run ────
    import shutil
    ckpt_out.mkdir(parents=True, exist_ok=True)
    resume_ckpt = None
    step_dirs = sorted(
        [d for d in ckpt_out.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1])
    )
    if step_dirs:
        resume_ckpt = str(step_dirs[-1])
        print(f"  [resume] Found checkpoint: {step_dirs[-1].name} — resuming from step {step_dirs[-1].name.split('-')[-1]}")
    else:
        print(f"  [resume] No checkpoint found — training from scratch")

    t_train = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    train_elapsed = round((time.time() - t_train) / 60, 1)
    final_loss = round(train_result.training_loss, 6)
    print(f"  Training done: loss={final_loss}, elapsed={train_elapsed}min")

    # Save final LoRA adapter
    model.save_pretrained(str(ckpt_out))
    tokenizer.save_pretrained(str(ckpt_out))

    # Cleanup intermediate HuggingFace checkpoint-* dirs (keep final adapter only)
    for d in ckpt_out.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            shutil.rmtree(d)
            print(f"  [cleanup] Removed intermediate checkpoint: {d.name}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    model.eval()
    val_ds = load_from_disk(VAL_DS)
    N = min(EVAL_N, len(val_ds))
    eval_ds = val_ds.select(range(N))
    samples = list(eval_ds)

    questions = [s.get("question", "") or (s["prompt"][-1]["content"] if s.get("prompt") else "")
                 for s in samples]
    gts      = [str(s.get("answer", "")) for s in samples]
    subjects = [str(s.get("type", ""))   for s in samples]
    correct_arr = [False] * N
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for bs in range(0, N, EVAL_BATCH_SIZE):
        batch_idx = list(range(bs, min(bs + EVAL_BATCH_SIZE, N)))
        texts = []
        for i in batch_idx:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": questions[i]}]
            try:
                text = tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            except TypeError:
                text = tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False)
            texts.append(text)

        tokenizer.padding_side = "left"
        enc = tokenizer(texts, return_tensors="pt", truncation=True,
                        max_length=MAX_NEW_TOKENS * 2, padding=True).to(device)
        input_len = enc["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for pos, i in enumerate(batch_idx):
            raw  = tokenizer.decode(out[pos][input_len:], skip_special_tokens=True)
            pred = extract_answer_from_text(raw)
            correct_arr[i] = verify_answer(
                pred, gts[i], subject=subjects[i],
                question_text=questions[i], use_z3=True)

    correct_all = correct_phys = correct_logic = 0
    n_phys = n_logic = 0
    for i in range(N):
        subj = subjects[i]
        if correct_arr[i]: correct_all += 1
        if subj == "physics":
            n_phys += 1
            if correct_arr[i]: correct_phys += 1
        elif subj == "logic":
            n_logic += 1
            if correct_arr[i]: correct_logic += 1

    result = {
        "variant_id":          vid,
        "variant_name":        vname,
        "description":         variant["desc"],
        "reward_weights":      {
            "format_w": variant["format_w"],
            "correct_w": variant["correct_w"],
            "unit_w": variant["unit_w"],
            "len_pen": variant["len_pen"],
        },
        "accuracy_overall":    round(correct_all / N * 100, 2),
        "accuracy_physics":    round(correct_phys / n_phys * 100, 2) if n_phys else 0,
        "accuracy_logic":      round(correct_logic / n_logic * 100, 2) if n_logic else 0,
        "n_total":             N,
        "training_loss":       final_loss,
        "grpo_steps":          GRPO_STEPS_RUN,
        "train_elapsed_min":   train_elapsed,
        "smoke":               SMOKE,
    }

    print(f"  {vid}: overall={result['accuracy_overall']:.2f}%  "
          f"phys={result['accuracy_physics']:.2f}%  logic={result['accuracy_logic']:.2f}%")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


all_results = []
_resume_file = LOG_OUT_LATEST if LOG_OUT_LATEST.exists() else LOG_OUT
if _resume_file.exists():
    with open(_resume_file) as f:
        done = json.load(f).get("reward_ablation", [])
    done_ids = {r["variant_id"] for r in done}
    all_results.extend(done)
    print(f"Resuming from {_resume_file} — already done: {done_ids}")
else:
    done_ids = set()

for variant in REWARD_VARIANTS:
    if variant["id"] in done_ids:
        print(f"\nSkipping {variant['id']} (already done)")
        continue
    result = _run_grpo_variant(variant)
    all_results.append(result)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    payload = {"reward_ablation": all_results, "run_ts": _RUN_TS}
    with open(LOG_OUT, "w") as f:
        json.dump(payload, f, indent=2)
    with open(LOG_OUT_LATEST, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {LOG_OUT}")
    print(f"Saved: {LOG_OUT_LATEST}")

print(f"\n{'='*60}")
print("REWARD ABLATION SUMMARY")
print(f"{'='*60}")
print(f"{'Variant':<12} {'Overall':>8} {'Physics':>8} {'Logic':>8}  Description")
print("-" * 65)
for r in sorted(all_results, key=lambda x: x.get("accuracy_overall", 0), reverse=True):
    print(f"{r['variant_id']:<12} {r.get('accuracy_overall',0):>7.2f}% "
          f"{r.get('accuracy_physics',0):>7.2f}% "
          f"{r.get('accuracy_logic',0):>7.2f}%  {r.get('variant_name','')}")
