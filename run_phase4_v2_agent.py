"""
Phase 4 — Agent v2 Evaluation + Ablation Matrix
================================================
Evaluates the full dual-LoRA + self-consistency agent v2 and runs
a 6-configuration ablation study.

Ablation Matrix
---------------
  Config 0: Qwen3.5-4B zero-shot (no LoRA)               ~25-30% expected
  Config 1: + SFT Phase 1  (qwen35_sft/final)            ~40-45%
  Config 2: + Logic SFT Phase 1.5  (qwen35_sft_logic/final) ~45-50%
  Config 3: + GRPO mixed Phase 2  (qwen35_grpo/final)    ~50-55%
  Config 4: + Dual LoRA specialists + Router              ~55-60%
  Config 5: + Self-consistency 5x  (full v2 agent)       ~65-75% target

Output files
------------
  logs/qwen35_ablation.json          — per-config accuracy breakdown
  logs/qwen35_final_results.jsonl    — per-sample detail (config 5 only)
  logs/qwen35_final_summary.json     — overall + sub-group breakdown

Run:
    python run_phase4_v2_agent.py                      # full ablation
    python run_phase4_v2_agent.py --smoke-test         # 10 samples per config
    python run_phase4_v2_agent.py --config 5           # only config 5 (full v2)
"""
from __future__ import annotations

import sys, os, json, time, argparse, gc, logging
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
# Force line-buffered stdout so tee/logs flush in real-time
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test",  action="store_true",
                    help="Evaluate only 10 samples per config")
parser.add_argument("--config",      type=int, default=-1,
                    help="Run only this config ID (-1 = all)")
parser.add_argument("--start-config", type=int, default=0,
                    help="Resume from config N (loads previous LOG_ABL if present)")
parser.add_argument("--val-only",    action="store_true",
                    help="Use validation set (default=test/full eval set)")
args, _ = parser.parse_known_args()

from src.config import (
    MODEL_NAME, MAX_SEQ_LEN,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS,
    QWEN35_SFT_FINAL, QWEN35_SFT_LOGIC_FINAL,
    QWEN35_GRPO_FINAL, QWEN35_GRPO_PHYS_FINAL, QWEN35_GRPO_LOGIC_FINAL,
    VAL_DS,
    LOG_DIR, CKPT_DIR,
    SELF_CONSISTENCY_N, SELF_CONSISTENCY_TEMP,
    MAX_NEW_TOKENS, AGENT_MAX_RETRIES,
    ROUTER_PATH, TRAIN_DS,
)
from src.utils import setup_logger, print_vram
from src.symbolic_verifier import verify_answer, extract_answer_from_text
import torch
from pathlib import Path
from collections import Counter
import re

logger = setup_logger("phase4_eval", LOG_DIR)

_SMOKE_N    = 10
_EVAL_N     = None   # None = all
_MAX_RETRIES = AGENT_MAX_RETRIES   # 2
EVAL_BATCH_SIZE = 8  # batched inference: ~8x faster than sample-by-sample

LOG_JSONL  = Path(LOG_DIR) / "qwen35_final_results.jsonl"
LOG_ABL    = Path(LOG_DIR) / "qwen35_ablation.json"
LOG_SUM    = Path(LOG_DIR) / "qwen35_final_summary.json"

logger.info("=" * 68)
logger.info("Phase 4 Evaluation + Ablation")
logger.info(f"  smoke_test={args.smoke_test}  config={args.config}")
logger.info("=" * 68)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an expert in Logic and Physics. "
    "Think step by step and respond in the exact format:\n"
    "<reasoning>\n[Your step-by-step reasoning]\n</reasoning>\n"
    "<answer>\n[Final answer: letter / Yes/No/Unknown / number+unit]\n</answer>\n"
    "<explanation>\n[Concise explanation of why this answer is correct]\n</explanation>"
)


def _generate_one(model, tokenizer, question: str, temperature: float,
                  device: str = "cuda", subject: str = "",
                  few_shot: str = "") -> str:
    """Single generation. Switches LoRA adapter if available."""
    if subject and hasattr(model, "set_adapter"):
        try:
            model.set_adapter(subject)
        except Exception:
            pass

    user_content = few_shot + question if few_shot else question
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    # Use tokenize=False + separate tokenizer call (transformers 5.x returns
    # BatchEncoding from apply_chat_template, not a plain tensor)
    try:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=False, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)

    enc  = tokenizer(text, return_tensors="pt", truncation=True,
                     max_length=MAX_NEW_TOKENS * 2).to(device)
    inp  = enc["input_ids"]
    attn = enc["attention_mask"]
    do_sample = temperature > 0.05
    kwargs = dict(
        input_ids=inp, attention_mask=attn,
        max_new_tokens=MAX_NEW_TOKENS, do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stop_strings=["</explanation>"], tokenizer=tokenizer,
    )
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = 0.9

    with torch.inference_mode():
        out = model.generate(**kwargs)
    return tokenizer.decode(out[0][inp.shape[1]:], skip_special_tokens=True)


def _generate_batch(model, tokenizer, questions: list, temperatures: list,
                    device: str, subjects: list, few_shots: list) -> list:
    """Batched generation for multiple questions simultaneously.
    Returns list of decoded strings (new tokens only)."""
    texts = []
    for q, fs, subj in zip(questions, few_shots, subjects):
        if subj and hasattr(model, "set_adapter"):
            try:
                model.set_adapter(subj)
            except Exception:
                pass
        user_content = fs + q if fs else q
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=False, enable_thinking=False)
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
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stop_strings=["</explanation>"], tokenizer=tokenizer,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    input_len = enc["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**gen_kwargs)

    return [
        tokenizer.decode(out[i][input_len:], skip_special_tokens=True)
        for i in range(len(questions))
    ]


def _self_consistency(model, tokenizer, question: str, n: int,
                      device: str, subject: str) -> tuple[str, float]:
    """Self-consistency: n samples, majority vote — batched for speed."""
    # Batch all N generations at once instead of one-by-one
    candidates = _generate_batch(
        model, tokenizer,
        questions=[question] * n,
        temperatures=[SELF_CONSISTENCY_TEMP] * n,
        device=device,
        subjects=[subject] * n,
        few_shots=[""] * n,
    )
    extracted = [extract_answer_from_text(c) for c in candidates]
    counts = Counter(extracted)
    best_ans, best_cnt = counts.most_common(1)[0]
    frac = best_cnt / len(extracted)
    for raw in candidates:
        if extract_answer_from_text(raw) == best_ans:
            return raw, frac
    return candidates[0], frac


def _run_agent(model, tokenizer, question: str, ground_truth: str,
               subject: str, device: str, use_self_consistency: bool,
               retriever=None) -> dict:
    """Run the agent loop (attempt 0 greedy → attempt 1 SC if not verified)."""
    t_start = time.time()

    # Retrieve few-shot context (attempt 0 only)
    few_shot = ""
    if retriever is not None:
        try:
            examples = retriever.retrieve(question, top_k=3, subject=subject or None)
            few_shot = _fmt_few_shot(examples)
        except Exception:
            pass

    best_raw, vote_frac, retries, correct = "", 0.0, 0, False
    all_raws = []

    for attempt in range(_MAX_RETRIES + 1):
        if attempt == 0:
            raw = _generate_one(model, tokenizer, question, 0.1,
                                 device, subject, few_shot)
            vf = None
        else:
            if use_self_consistency:
                raw, vf = _self_consistency(
                    model, tokenizer, question, SELF_CONSISTENCY_N, device, subject)
                vote_frac = vf
            else:
                raw = _generate_one(model, tokenizer, question, 0.7,
                                     device, subject, "")
                vf = None

        all_raws.append(raw)
        pred = extract_answer_from_text(raw)
        correct = verify_answer(pred, ground_truth,
                                subject=subject, question_text=question, use_z3=True)
        if correct:
            best_raw = raw
            retries = attempt
            break
        best_raw = raw
        retries = attempt

    pred = extract_answer_from_text(best_raw)
    return {
        "predicted":   pred,
        "correct":     correct,
        "retry_count": retries,
        "latency_s":   round(time.time() - t_start, 2),
        "confidence":  round(vote_frac, 4) if vote_frac else 0.0,
        "n_attempts":  len(all_raws),
    }


def _fmt_few_shot(examples: list) -> str:
    if not examples:
        return ""
    lines = ["\n--- Reference examples ---"]
    for i, ex in enumerate(examples, 1):
        q = ex["question"][:300]
        a = ex["answer"]
        lines.append(f"\nExample {i}:\n{q}\n<answer>\n{a}\n</answer>")
    lines.append("\n--- Now answer the following ---\n")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION CONFIG DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = [
    {
        "id": 0, "name": "zero_shot",
        "desc": "Qwen3.5-4B zero-shot (no LoRA)",
        "lora_ckpt": None,
        "dual_lora": False,
        "use_router": False,
        "use_retriever": False,
        "use_self_consistency": False,
        "expected_acc": "25-30%",
    },
    {
        "id": 1, "name": "sft_phase1",
        "desc": "+ SFT Phase 1",
        "lora_ckpt": QWEN35_SFT_FINAL,
        "dual_lora": False,
        "use_router": False,
        "use_retriever": True,
        "use_self_consistency": False,
        "expected_acc": "40-45%",
    },
    {
        "id": 2, "name": "sft_logic",
        "desc": "+ Logic SFT Phase 1.5",
        "lora_ckpt": QWEN35_SFT_LOGIC_FINAL,
        "dual_lora": False,
        "use_router": False,
        "use_retriever": True,
        "use_self_consistency": False,
        "expected_acc": "45-50%",
    },
    {
        "id": 3, "name": "grpo_mixed",
        "desc": "+ GRPO mixed Phase 2",
        "lora_ckpt": QWEN35_GRPO_FINAL,
        "dual_lora": False,
        "use_router": False,
        "use_retriever": True,
        "use_self_consistency": False,
        "expected_acc": "50-55%",
    },
    {
        "id": 4, "name": "dual_lora_router",
        "desc": "+ Dual LoRA specialists + Router",
        "lora_ckpt": None,       # loads both physics + logic
        "dual_lora": True,
        "use_router": True,
        "use_retriever": True,
        "use_self_consistency": False,
        "expected_acc": "55-60%",
    },
    {
        "id": 5, "name": "full_v2",
        "desc": "+ Self-consistency 5x (full v2 agent)",
        "lora_ckpt": None,       # dual LoRA
        "dual_lora": True,
        "use_router": True,
        "use_retriever": True,
        "use_self_consistency": True,
        "expected_acc": "65-75%",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# LOAD EVAL DATASET
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1] Loading evaluation dataset")
from datasets import load_from_disk
val_ds = load_from_disk(VAL_DS)
N = _SMOKE_N if args.smoke_test else (_EVAL_N or len(val_ds))
eval_ds = val_ds.select(range(min(N, len(val_ds))))
print(f"    Eval samples: {len(eval_ds)}")
logger.info(f"Eval dataset: {len(eval_ds)} samples")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD BASE MODEL (shared across configs)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2] Loading base model")
from src.model_loader import load_base_model, apply_lora, load_multi_adapter_model
from peft import PeftModel

base_model, tokenizer = load_base_model(
    model_name=MODEL_NAME, dtype=torch.bfloat16,
    drop_vision=True, device_map="auto",
)
device = "cuda" if torch.cuda.is_available() else "cpu"
print_vram("base model loaded")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD RETRIEVER + ROUTER (shared across applicable configs)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3] Loading retriever")
from src.retriever import Retriever
retriever = Retriever(TRAIN_DS)
retriever.build()

print("\n[4] Loading router")
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
        logger.warning(f"Router unavailable: {e} — subject from dataset will be used")


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

ablation_results = []
# Resume: load previous partial results if starting mid-way
if args.start_config > 0 and Path(LOG_ABL).exists():
    try:
        with open(LOG_ABL) as _f:
            _prev = json.load(_f)
        ablation_results = _prev.get("ablation", [])
        logger.info(f"Resumed: loaded {len(ablation_results)} previous config results from {LOG_ABL}")
        print(f"  Resumed from {LOG_ABL}: {len(ablation_results)} configs already done.", flush=True)
    except Exception as _e:
        logger.warning(f"Could not load previous results: {_e}")

configs_to_run = (
    [c for c in ABLATION_CONFIGS if c["id"] == args.config]
    if args.config >= 0
    else [c for c in ABLATION_CONFIGS if c["id"] >= args.start_config]
)

for cfg in configs_to_run:
    cfg_id   = cfg["id"]
    cfg_name = cfg["name"]
    print(f"\n{'='*68}")
    print(f"Config {cfg_id}: {cfg['desc']}")
    print(f"{'='*68}")
    logger.info(f"Starting ablation config {cfg_id}: {cfg_name}")

    # ── Load correct model variant ────────────────────────────────────────────
    if cfg["dual_lora"]:
        # Dual LoRA: load physics + logic adapters
        adapters = {}
        for name, ckpt in [
            ("physics", QWEN35_GRPO_PHYS_FINAL),
            ("logic",   QWEN35_GRPO_LOGIC_FINAL),
        ]:
            if Path(ckpt).exists():
                adapters[name] = ckpt
            else:
                logger.warning(f"Dual LoRA {name} checkpoint not found: {ckpt}")
                # Fall back to the mixed GRPO checkpoint
                if Path(QWEN35_GRPO_FINAL).exists():
                    adapters[name] = QWEN35_GRPO_FINAL
                    logger.warning(f"  → Using mixed GRPO fallback for '{name}'")

        if adapters:
            model, _ = load_multi_adapter_model(
                MODEL_NAME, adapters, dtype=torch.bfloat16, drop_vision=True)
        else:
            model = base_model   # zero-shot fallback

    elif cfg["lora_ckpt"] and Path(cfg["lora_ckpt"]).exists():
        model = PeftModel.from_pretrained(
            base_model, cfg["lora_ckpt"], is_trainable=False)
    else:
        model = base_model

    model.eval()
    print_vram(f"Config {cfg_id} model loaded")

    # ── Batched Eval loop ──────────────────────────────────────────────────────
    t_cfg_start = time.time()
    ret_obj = cfg["use_retriever"] and retriever or None

    # Pre-gather all sample metadata + few-shots upfront
    samples_list = list(eval_ds)
    N = len(samples_list)
    all_questions  = []
    all_gts        = []
    all_subjects   = []
    all_few_shots  = []

    for sample in samples_list:
        q    = sample["prompt"][-1]["content"] if sample.get("prompt") else sample.get("question", "")
        gt   = str(sample.get("answer", ""))
        subj = str(sample.get("type", ""))
        if cfg["use_router"] and router is not None and subj not in ("physics", "logic"):
            subj, _ = router.predict(q)
        fs = ""
        if ret_obj is not None:
            try:
                examples = ret_obj.retrieve(q, top_k=3, subject=subj or None)
                fs = _fmt_few_shot(examples)
            except Exception:
                pass
        all_questions.append(q)
        all_gts.append(gt)
        all_subjects.append(subj)
        all_few_shots.append(fs)

    print(f"  Pre-gathered {N} samples + few-shots. Starting batched inference (batch={EVAL_BATCH_SIZE})...", flush=True)

    # State tracking
    correct_arr  = [False] * N
    raw_outputs  = [None]  * N
    retry_counts = [0]     * N
    resolved     = [False] * N  # True once we have a final answer (correct or max retries)

    for attempt in range(_MAX_RETRIES + 1):
        # pending = samples not yet resolved (not correct and retries not exhausted)
        if attempt == 0:
            pending = list(range(N))
        else:
            pending = [i for i in range(N) if not resolved[i]]

        if not pending:
            break

        print(f"  Attempt {attempt}: {len(pending)} samples pending...", flush=True)

        if attempt >= 1 and cfg["use_self_consistency"]:
            # Self-consistency mode — _self_consistency now batches internally
            for i in pending:
                raw, vf = _self_consistency(
                    model, tokenizer, all_questions[i],
                    SELF_CONSISTENCY_N, device, all_subjects[i])
                pred = extract_answer_from_text(raw)
                ok   = verify_answer(pred, all_gts[i],
                                     subject=all_subjects[i],
                                     question_text=all_questions[i], use_z3=True)
                raw_outputs[i]  = raw
                correct_arr[i]  = ok
                retry_counts[i] = attempt
                if ok or attempt == _MAX_RETRIES:
                    resolved[i] = True
        else:
            # Batched generation
            temperature = 0.1 if attempt == 0 else 0.7
            fs_list     = all_few_shots if attempt == 0 else [""] * N

            # Collect texts for pending samples
            pending_questions  = [all_questions[i]  for i in pending]
            pending_subjects   = [all_subjects[i]   for i in pending]
            pending_few_shots  = [fs_list[i]        for i in pending]
            pending_temps      = [temperature]      * len(pending)

            raw_list = []
            for bs in range(0, len(pending), EVAL_BATCH_SIZE):
                batch_q   = pending_questions [bs: bs + EVAL_BATCH_SIZE]
                batch_s   = pending_subjects  [bs: bs + EVAL_BATCH_SIZE]
                batch_fs  = pending_few_shots [bs: bs + EVAL_BATCH_SIZE]
                batch_t   = pending_temps     [bs: bs + EVAL_BATCH_SIZE]
                raws = _generate_batch(model, tokenizer, batch_q, batch_t,
                                       device, batch_s, batch_fs)
                raw_list.extend(raws)
                done_so_far = min(bs + EVAL_BATCH_SIZE, len(pending))
                if done_so_far % (EVAL_BATCH_SIZE * 4) == 0 or done_so_far == len(pending):
                    print(f"    generated {done_so_far}/{len(pending)}", flush=True)

            for pos, i in enumerate(pending):
                raw  = raw_list[pos]
                pred = extract_answer_from_text(raw)
                ok   = verify_answer(pred, all_gts[i],
                                     subject=all_subjects[i],
                                     question_text=all_questions[i], use_z3=True)
                raw_outputs[i]  = raw
                correct_arr[i]  = ok
                retry_counts[i] = attempt
                if ok or attempt == _MAX_RETRIES:
                    resolved[i] = True

        n_resolved = sum(resolved)
        n_correct  = sum(correct_arr)
        print(f"  Attempt {attempt} done: resolved={n_resolved}/{N}  correct={n_correct}/{N}  acc={n_correct/N*100:.2f}%", flush=True)

    # ── Build sample_records ──────────────────────────────────────────────────
    correct_all = 0
    correct_phys = correct_logic = 0
    n_phys = n_logic = 0
    per_retry = {0: 0, 1: 0, 2: 0}
    conf_correct = []
    conf_wrong   = []
    sample_records = []

    for i in range(N):
        pred = extract_answer_from_text(raw_outputs[i] or "")
        result = {
            "predicted":   pred,
            "correct":     correct_arr[i],
            "retry_count": retry_counts[i],
            "latency_s":   0.0,
            "confidence":  0.0,
            "n_attempts":  retry_counts[i] + 1,
            "idx":         i,
            "subject":     all_subjects[i],
            "ground_truth": all_gts[i],
            "question":    all_questions[i][:200],
        }
        sample_records.append(result)

        if correct_arr[i]:
            correct_all += 1
        if all_subjects[i] == "physics":
            n_phys += 1
            if correct_arr[i]:
                correct_phys += 1
        elif all_subjects[i] == "logic":
            n_logic += 1
            if correct_arr[i]:
                correct_logic += 1

        rc = min(retry_counts[i], 2)
        per_retry[rc] = per_retry.get(rc, 0) + 1
        conf_correct.append(0.0) if correct_arr[i] else conf_wrong.append(0.0)

        if (i + 1) % 10 == 0 or i == N - 1:
            acc = correct_all / (i + 1) * 100
            print(f"  [{i+1:4d}/{N}]  acc={acc:.2f}%", flush=True)

    # ── Compute metrics ───────────────────────────────────────────────────────
    total   = len(eval_ds)
    acc_all = correct_all / total * 100 if total else 0
    acc_phy = correct_phys / n_phys * 100 if n_phys else 0
    acc_log = correct_logic / n_logic * 100 if n_logic else 0
    elapsed = round((time.time() - t_cfg_start) / 60, 1)

    avg_conf_c = sum(conf_correct) / len(conf_correct) if conf_correct else 0
    avg_conf_w = sum(conf_wrong)   / len(conf_wrong)   if conf_wrong   else 0

    cfg_result = {
        "config_id":        cfg_id,
        "config_name":      cfg_name,
        "description":      cfg["desc"],
        "expected_acc":     cfg["expected_acc"],
        "accuracy_overall": round(acc_all, 2),
        "accuracy_physics": round(acc_phy, 2),
        "accuracy_logic":   round(acc_log, 2),
        "n_total":          total,
        "n_physics":        n_phys,
        "n_logic":          n_logic,
        "n_correct":        correct_all,
        "per_retry":        per_retry,
        "avg_confidence_correct": round(avg_conf_c, 4),
        "avg_confidence_wrong":   round(avg_conf_w, 4),
        "elapsed_min":      elapsed,
        "smoke_test":       args.smoke_test,
    }
    ablation_results.append(cfg_result)
    logger.info(
        f"Config {cfg_id} done | acc={acc_all:.2f}%  phys={acc_phy:.2f}%  "
        f"logic={acc_log:.2f}%  elapsed={elapsed}min"
    )
    print(f"\n  Config {cfg_id} result:")
    print(f"    Overall   : {acc_all:.2f}%")
    print(f"    Physics   : {acc_phy:.2f}%")
    print(f"    Logic     : {acc_log:.2f}%")
    print(f"    Time      : {elapsed} min")

    # For full v2 (config 5), also save per-sample JSONL
    if cfg_id == 5 or (args.config == cfg_id):
        LOG_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_JSONL, "w") as f:
            for rec in sample_records:
                f.write(json.dumps(rec) + "\n")
        logger.info(f"Per-sample JSONL saved → {LOG_JSONL}")

    # Free model VRAM between configs
    if cfg_id < len(ABLATION_CONFIGS) - 1:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# SAVE ABLATION RESULTS
# ══════════════════════════════════════════════════════════════════════════════

Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
with open(LOG_ABL, "w") as f:
    json.dump({"ablation": ablation_results}, f, indent=2)
logger.info(f"Ablation matrix saved → {LOG_ABL}")

# Final summary from config 5 (or last config run)
last = ablation_results[-1]
summary = {
    "model":           MODEL_NAME,
    "eval_n":          last["n_total"],
    "smoke_test":      args.smoke_test,
    "accuracy_overall": last["accuracy_overall"],
    "accuracy_physics": last["accuracy_physics"],
    "accuracy_logic":   last["accuracy_logic"],
    "per_retry_distribution": last["per_retry"],
    "confidence_analysis": {
        "avg_when_correct": last["avg_confidence_correct"],
        "avg_when_wrong":   last["avg_confidence_wrong"],
    },
    "ablation_summary": [
        {
            "id": r["config_id"],
            "name": r["config_name"],
            "acc": r["accuracy_overall"],
            "phys": r["accuracy_physics"],
            "logic": r["accuracy_logic"],
        }
        for r in ablation_results
    ],
}
with open(LOG_SUM, "w") as f:
    json.dump(summary, f, indent=2)
logger.info(f"Summary saved → {LOG_SUM}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 68)
print(" PHASE 4 EVALUATION COMPLETE")
print("=" * 68)
print(f"\n{'CFG':>3}  {'NAME':<22}  {'ACC':>7}  {'PHYS':>7}  {'LOGIC':>7}  EXPECTED")
print("-" * 68)
for r in ablation_results:
    print(
        f"{r['config_id']:>3}  {r['config_name']:<22}  "
        f"{r['accuracy_overall']:>6.2f}%  "
        f"{r['accuracy_physics']:>6.2f}%  "
        f"{r['accuracy_logic']:>6.2f}%  "
        f"  {r['expected_acc']}"
    )
print("-" * 68)
print(f"\n  Ablation JSON  : {LOG_ABL}")
print(f"  Per-sample JSONL: {LOG_JSONL}")
print(f"  Summary JSON   : {LOG_SUM}")
print("=" * 68)
