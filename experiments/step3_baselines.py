"""
BƯỚC 3: Baseline inference — 6 models zero-shot
=================================================
Chạy 5 open-source models + GPT-4o-mini trên cùng 217 val samples.

Chạy:
    python experiments/step3_baselines.py
    python experiments/step3_baselines.py --smoke-test --only-ids B2 B3
    HF_TOKEN=hf_... python experiments/step3_baselines.py

Models:
    B1  = Qwen3.5-4B zero-shot   (đã có trong ablation v2, có thể skip)
    B2  = Qwen/Qwen2.5-Math-7B-Instruct
    B3  = EleutherAI/llemma_7b
    B4  = mistralai/Mistral-7B-Instruct-v0.3
    B5  = Qwen/Qwen2-Math-7B-Instruct
    B6  = deepseek-ai/deepseek-math-7b-instruct  (thay thế GPT-4o-mini)
"""
from __future__ import annotations
import sys, os, json, time, gc, argparse, re
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true")
parser.add_argument("--only-ids",   nargs="*",
                    help="subset of B2 B3 B4 B5 B6 to run")
args, _ = parser.parse_known_args()

# ── Helpers ──────────────────────────────────────────────────────────────────
from datetime import datetime

def _ts() -> str:
    """Current time string HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")

def _bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / total) if total else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"

def _eta(elapsed: float, done: int, total: int) -> str:
    if done == 0:
        return "ETA: --:--"
    remaining = elapsed / done * (total - done)
    m, s = divmod(int(remaining), 60)
    return f"ETA: {m:02d}:{s:02d}"

def _vram_str() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"VRAM {used:.1f}/{total:.0f}GB"
    except Exception:
        pass
    return "VRAM N/A"

# ── HuggingFace login ────────────────────────────────────────────────────────
print(f"[{_ts()}] ── STEP 3: Baseline Evaluation ──────────────────────────────")
_hf_token = os.environ.get("HF_TOKEN", "").strip()
if _hf_token:
    try:
        from huggingface_hub import login as hf_login, whoami
        hf_login(token=_hf_token, add_to_git_credential=False)
        user = whoami(token=_hf_token)["name"]
        print(f"[{_ts()}] HF login OK  →  user={user}  token=hf_...{_hf_token[-4:]}")
    except Exception as _e:
        print(f"[{_ts()}] HF login FAILED: {_e}")
else:
    print(f"[{_ts()}] WARNING: HF_TOKEN not set — gated models (B5) will fail.")
    print(f"[{_ts()}]   Run:  HF_TOKEN=hf_... python experiments/step3_baselines.py")


from src.config    import VAL_DS, LOG_DIR, MAX_NEW_TOKENS
from src.utils     import setup_logger, print_vram
from src.symbolic_verifier import verify_answer, extract_answer_from_text
import torch
from pathlib import Path

from datetime import datetime
_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Compat shim: InternLM2 custom code uses DynamicCache.from_legacy_cache()
# which was removed in transformers ≥ 4.46 / 5.x ────────────────────────────
try:
    from transformers import DynamicCache as _DC
    if not hasattr(_DC, "from_legacy_cache"):
        @classmethod
        def _from_legacy(cls, past_key_values=None):
            if past_key_values is None:
                return cls()
            cache = cls()
            for layer_idx, layer_past in enumerate(past_key_values):
                cache.update(layer_past[0], layer_past[1], layer_idx)
            return cache
        _DC.from_legacy_cache = _from_legacy
        print(f"[{_ts()}]  DynamicCache.from_legacy_cache shim applied (transformers compat)")
except Exception:
    pass

EVAL_BATCH_SIZE = 4    # smaller for 7B models
_SMOKE_N        = 8
LOG_OUT         = Path(LOG_DIR) / f"baselines_{_RUN_TS}.json"
LOG_OUT_LATEST  = Path(LOG_DIR) / "baselines_results_latest.json"

SYSTEM_PROMPT = (
    "You are an expert in Logic and Physics. "
    "Think step by step and respond in the exact format:\n"
    "<reasoning>\n[Your step-by-step reasoning]\n</reasoning>\n"
    "<answer>\n[Final answer: letter / Yes/No/Unknown / number+unit]\n</answer>\n"
    "<explanation>\n[Concise explanation of why this answer is correct]\n</explanation>"
)

BASELINES = [
    {
        "id": "B2", "model_id": "Qwen/Qwen2.5-Math-7B-Instruct",
        "type": "hf", "chat_template": True,
        "desc": "Qwen2.5-Math-7B-Instruct (zero-shot)",
    },
    {
        "id": "B3", "model_id": "EleutherAI/llemma_7b",
        "type": "hf", "chat_template": False,
        "desc": "Llemma-7B (zero-shot)",
    },
    {
        "id": "B4", "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "type": "hf", "chat_template": True,
        "desc": "Mistral-7B-Instruct-v0.3 (zero-shot)",
    },
    {
        "id": "B5", "model_id": "Qwen/Qwen2-Math-7B-Instruct",
        "type": "hf", "chat_template": True,
        "desc": "Qwen2-Math-7B-Instruct (zero-shot)",
    },
    {
        "id": "B6", "model_id": "deepseek-ai/deepseek-math-7b-instruct",
        "type": "hf", "chat_template": True,
        "desc": "DeepSeek-Math-7B-Instruct (zero-shot)",
    },
]

# Filter by --only-ids
if args.only_ids:
    BASELINES = [b for b in BASELINES if b["id"] in args.only_ids]
    print(f"Only running: {[b['id'] for b in BASELINES]}")


def _format_llemma(question: str) -> str:
    """Llemma has no chat template — use plain text."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {question}\n\n"
        "Response:"
    )


def _run_hf_baseline(baseline: dict, samples: list) -> dict:
    """Run a HuggingFace model in zero-shot mode with detailed progress."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_id = baseline["model_id"]
    bid      = baseline["id"]

    print(f"\n[{_ts()}] {'='*58}", flush=True)
    print(f"[{_ts()}]  {bid}: {baseline['desc']}", flush=True)
    print(f"[{_ts()}]  Model: {model_id}", flush=True)
    print(f"[{_ts()}] {'='*58}", flush=True)

    load_kwargs = dict(trust_remote_code=baseline.get("trust_remote_code", True))
    if _hf_token:
        load_kwargs["token"] = _hf_token
    print(f"[{_ts()}]       trust_remote_code={load_kwargs['trust_remote_code']}", flush=True)

    # ── [1/4] Tokenizer ──────────────────────────────────────────────────────
    print(f"[{_ts()}]  [1/4] Loading tokenizer ...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, padding_side="left", **load_kwargs)
    except Exception as e:
        err = str(e)
        print(f"[{_ts()}]  ERROR loading tokenizer: {err[:200]}", flush=True)
        if any(k in err for k in ["401", "403", "gated", "auth", "login"]):
            print(f"[{_ts()}]  → Auth error. Set HF_TOKEN and re-run.", flush=True)
            return {"id": bid, "model_id": model_id, "description": baseline["desc"],
                    "error": f"auth_error: {err[:120]}",
                    "accuracy_overall": 0.0, "accuracy_physics": 0.0,
                    "accuracy_logic": 0.0, "n_total": 0, "n_physics": 0,
                    "n_logic": 0, "n_correct": 0, "elapsed_min": 0.0}
        raise
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[{_ts()}]       vocab_size={tokenizer.vocab_size:,}  "
          f"pad='{tokenizer.pad_token}'", flush=True)

    # ── [2/4] Model ───────────────────────────────────────────────────────────
    print(f"[{_ts()}]  [2/4] Loading model weights (BF16) ...", flush=True)
    t_load = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            **load_kwargs)
    except Exception as e:
        err = str(e)
        print(f"[{_ts()}]  ERROR loading model: {err[:200]}", flush=True)
        if any(k in err for k in ["401", "403", "gated", "auth", "login"]):
            print(f"[{_ts()}]  → Auth error. Set HF_TOKEN and re-run.", flush=True)
            return {"id": bid, "model_id": model_id, "description": baseline["desc"],
                    "error": f"auth_error: {err[:120]}",
                    "accuracy_overall": 0.0, "accuracy_physics": 0.0,
                    "accuracy_logic": 0.0, "n_total": 0, "n_physics": 0,
                    "n_logic": 0, "n_correct": 0, "elapsed_min": 0.0}
        raise
    model.eval()
    load_elapsed = time.time() - t_load
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[{_ts()}]       Loaded in {load_elapsed:.1f}s  "
          f"params={n_params:.2f}B  {_vram_str()}", flush=True)

    # ── [3/4] Build prompts ───────────────────────────────────────────────────
    N        = len(samples)
    questions = [s.get("question", "")
                 or (s["prompt"][-1]["content"] if s.get("prompt") else "")
                 for s in samples]
    gts      = [str(s.get("answer", "")) for s in samples]
    subjects = [str(s.get("type",   "")) for s in samples]
    n_phys_total  = subjects.count("physics")
    n_logic_total = subjects.count("logic")
    print(f"[{_ts()}]  [3/4] Prompts ready: "
          f"total={N}  physics={n_phys_total}  logic={n_logic_total}", flush=True)

    # ── [4/4] Inference ───────────────────────────────────────────────────────
    n_batches = (N + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE
    print(f"[{_ts()}]  [4/4] Inference  "
          f"batch_size={EVAL_BATCH_SIZE}  n_batches={n_batches}  "
          f"max_new={MAX_NEW_TOKENS}", flush=True)

    correct_arr = [False] * N
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    t_infer     = time.time()

    for b_num, bs in enumerate(range(0, N, EVAL_BATCH_SIZE)):
        t_batch   = time.time()
        batch_idx = list(range(bs, min(bs + EVAL_BATCH_SIZE, N)))
        texts = []
        for i in batch_idx:
            if baseline["chat_template"]:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": questions[i]},
                ]
                try:
                    text = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False)
                except Exception:
                    text = _format_llemma(questions[i])
            else:
                text = _format_llemma(questions[i])
            texts.append(text)

        enc = tokenizer(texts, return_tensors="pt", truncation=True,
                        max_length=MAX_NEW_TOKENS * 2, padding=True).to(device)
        input_len = enc["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        batch_marks = []
        for pos, i in enumerate(batch_idx):
            raw  = tokenizer.decode(out[pos][input_len:], skip_special_tokens=True)
            pred = extract_answer_from_text(raw)
            ok   = verify_answer(pred, gts[i], subject=subjects[i],
                                 question_text=questions[i], use_z3=True)
            correct_arr[i] = ok
            batch_marks.append("✓" if ok else "✗")

        done    = min(bs + EVAL_BATCH_SIZE, N)
        elapsed = time.time() - t_infer
        # running domain accuracy
        n_p = sum(1 for j in range(done) if subjects[j] == "physics")
        n_l = sum(1 for j in range(done) if subjects[j] == "logic")
        c_p = sum(correct_arr[j] for j in range(done) if subjects[j] == "physics")
        c_l = sum(correct_arr[j] for j in range(done) if subjects[j] == "logic")
        phys_str  = f"{c_p/n_p*100:.0f}%" if n_p else "---"
        logic_str = f"{c_l/n_l*100:.0f}%" if n_l else "---"
        acc_now   = sum(correct_arr[:done]) / done * 100
        batch_sec = time.time() - t_batch
        print(
            f"[{_ts()}]  batch {b_num+1:>3}/{n_batches}  "
            f"{_bar(done, N)}  {done:>3}/{N}  "
            f"acc={acc_now:5.1f}%  "
            f"(phy={phys_str} lgc={logic_str})  "
            f"{_eta(elapsed, done, N)}  "
            f"{_vram_str()}  "
            f"[{''.join(batch_marks)}]  {batch_sec:.1f}s/batch",
            flush=True,
        )

    total_elapsed = time.time() - t_infer
    final_acc = sum(correct_arr) / N * 100
    print(f"[{_ts()}]  ── Inference done in {total_elapsed/60:.1f} min  "
          f"final acc={final_acc:.2f}% ──", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[{_ts()}]  Model unloaded  {_vram_str()}", flush=True)

    return _build_result(baseline, N, correct_arr, subjects,
                         load_elapsed + total_elapsed)


def _run_openai_baseline(baseline: dict, samples: list) -> dict:
    """Run GPT-4o-mini via openai API."""
    try:
        from openai import OpenAI
    except ImportError:
        print("  openai package not found. pip install openai")
        return {"id": baseline["id"], "error": "openai not installed"}

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("  WARNING: OPENAI_API_KEY not set — skipping B6")
        return {"id": baseline["id"], "error": "no OPENAI_API_KEY"}

    client = OpenAI(api_key=api_key)
    N = len(samples)
    questions = [s.get("question", "")
                 or (s["prompt"][-1]["content"] if s.get("prompt") else "")
                 for s in samples]
    gts      = [str(s.get("answer", "")) for s in samples]
    subjects = [str(s.get("type", ""))   for s in samples]
    correct_arr = [False] * N
    t_start = time.time()

    for i, (q, gt, subj) in enumerate(zip(questions, gts, subjects)):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": q},
                ],
                max_tokens=1024,
                temperature=0.0,
            )
            raw  = resp.choices[0].message.content or ""
            pred = extract_answer_from_text(raw)
            correct_arr[i] = verify_answer(
                pred, gt, subject=subj, question_text=q, use_z3=True)
        except Exception as exc:
            print(f"  API error at sample {i}: {exc}")
        if i % 20 == 0:
            done = i + 1
            print(f"    {done}/{N}  acc={sum(correct_arr[:done])/done*100:.1f}%", flush=True)
        time.sleep(0.2)  # rate-limit guard

    return _build_result(baseline, N, correct_arr, subjects, time.time() - t_start)


def _build_result(baseline, N, correct_arr, subjects, elapsed_sec):
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
    return {
        "id":               baseline["id"],
        "model_id":         baseline["model_id"],
        "description":      baseline["desc"],
        "accuracy_overall": round(correct_all / N * 100, 2),
        "accuracy_physics": round(correct_phys / n_phys * 100, 2) if n_phys else 0.0,
        "accuracy_logic":   round(correct_logic / n_logic * 100, 2) if n_logic else 0.0,
        "n_total":          N,
        "n_physics":        n_phys,
        "n_logic":          n_logic,
        "n_correct":        correct_all,
        "elapsed_min":      round(elapsed_sec / 60, 1),
    }


print(f"\n[{_ts()}] [1] Loading val dataset")
from datasets import load_from_disk
val_ds   = load_from_disk(VAL_DS)
N        = _SMOKE_N if args.smoke_test else len(val_ds)
eval_ds  = val_ds.select(range(N))
samples  = list(eval_ds)
print(f"[{_ts()}]     Eval samples: {N}  "
      f"(smoke={args.smoke_test})")

# Load existing results to allow resuming
all_results = []
if LOG_OUT_LATEST.exists():
    with open(LOG_OUT_LATEST) as f:
        existing = json.load(f).get("baselines", [])
    done_ids = {r["id"] for r in existing if "error" not in r}
    all_results.extend(existing)
    print(f"[{_ts()}]     Resuming — already done (no error): {done_ids}")
else:
    done_ids = set()

print(f"[{_ts()}]     Models to run: {[b['id'] for b in BASELINES if b['id'] not in done_ids]}")
t_total = time.time()

for baseline in BASELINES:
    if baseline["id"] in done_ids:
        print(f"\n[{_ts()}] Skipping {baseline['id']} (already done OK)")
        continue

    try:
        if baseline["type"] == "hf":
            result = _run_hf_baseline(baseline, samples)
        elif baseline["type"] == "openai":
            result = _run_openai_baseline(baseline, samples)
        else:
            print(f"[{_ts()}]   Unknown type: {baseline['type']}")
            continue
    except Exception as exc:
        import traceback
        print(f"[{_ts()}]   ERROR running {baseline['id']}: {exc}")
        traceback.print_exc()
        result = {
            "id": baseline["id"],
            "model_id": baseline.get("model_id", ""),
            "description": baseline.get("desc", ""),
            "error": str(exc),
            "accuracy_overall": 0.0,
            "accuracy_physics": 0.0,
            "accuracy_logic":   0.0,
            "n_total": 0, "n_physics": 0, "n_logic": 0, "n_correct": 0,
            "elapsed_min": 0.0,
        }
        import gc, torch
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    all_results.append(result)

    # ── Save after each model ────────────────────────────────────────────────
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    payload = {"baselines": all_results, "smoke_test": args.smoke_test}
    with open(LOG_OUT, "w") as f:
        json.dump(payload, f, indent=2)
    with open(LOG_OUT_LATEST, "w") as f:
        json.dump(payload, f, indent=2)

    acc   = result.get('accuracy_overall', 'ERR')
    phys  = result.get('accuracy_physics', '---')
    logic = result.get('accuracy_logic',  '---')
    elap  = result.get('elapsed_min', 0)
    err   = result.get('error', None)
    status = f"FAILED ({err[:60]}...)" if err else f"overall={acc}%  phys={phys}%  logic={logic}%"
    print(f"[{_ts()}]  ✔ {baseline['id']} saved → {status}  ({elap} min)")

total_wall = (time.time() - t_total) / 60
print(f"\n[{_ts()}] {'='*60}")
print(f"[{_ts()}]  ALL BASELINES DONE  ({total_wall:.1f} min total)")
print(f"[{_ts()}] {'='*60}")
print(f"[{_ts()}]  {'ID':<4}  {'Model':<42}  {'Overall':>8}  {'Phys':>7}  {'Logic':>7}  {'Status'}")
print(f"[{_ts()}]  {'-'*80}")
for r in sorted(all_results, key=lambda x: x.get("accuracy_overall", 0), reverse=True):
    err    = r.get("error")
    status = f"ERROR: {err[:30]}" if err else "OK"
    print(
        f"[{_ts()}]  {r['id']:<4}  {r.get('model_id','N/A'):<42}  "
        f"{r.get('accuracy_overall',0):>7.2f}%  "
        f"{r.get('accuracy_physics',0):>6.2f}%  "
        f"{r.get('accuracy_logic',0):>6.2f}%  "
        f"{status}",
        flush=True,
    )
print(f"[{_ts()}]  {'='*80}")
print(f"[{_ts()}]  Saved → {LOG_OUT}", flush=True)
