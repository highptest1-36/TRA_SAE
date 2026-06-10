"""
BƯỚC 13: Tool-Augmented Baselines (reviewer "missing #2")
=========================================================
So sánh LLM-direct vs LLM-tool-assisted trên CÙNG model (best checkpoint),
để trả lời câu hỏi reviewer: "có baseline tool-based (LLM+calculator, LLM+Z3)?"

  Arm A — direct      : model tự trả lời (giống pipeline chính).
  Arm B — tool-assist :
      * Physics: model xuất biểu thức Python trong <calc>...</calc>; ta tính lại
                 bằng AST sandbox (PAL-style) → sửa lỗi số học.
      * Logic  : dùng Z3 (src.z3_engine) suy diễn verdict Yes/No từ premises của
                 dataset, KHÔNG nhìn ground-truth (probe cả 'Yes' và 'No').
                 Nếu Z3 abstain (Unknown/parse fail) → giữ câu trả lời của LLM.

Không leak đáp án: Z3 verdict suy ra từ premises; chỉ so GT lúc chấm điểm.

Output: logs/tool_baselines_results_latest.json
        (overall/physics/logic cho cả 2 arm + delta + #samples Z3/calc kích hoạt)

Chạy:
    python experiments/step13_tool_baselines.py
    python experiments/step13_tool_baselines.py --smoke-test
    python experiments/step13_tool_baselines.py --ckpt /path/to/lora/final
"""
from __future__ import annotations
import sys, os, json, time, gc, ast, math, operator, argparse, re
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true")
parser.add_argument("--ckpt", default=None, help="LoRA checkpoint (default: best GRPO)")
args, _ = parser.parse_known_args()

from src.config import (MODEL_NAME, QWEN35_GRPO_FINAL, VAL_DS, LOG_DIR, MAX_NEW_TOKENS)
from src.symbolic_verifier import verify_answer, extract_answer_from_text, parse_numerical
import torch
from pathlib import Path
from datetime import datetime
from collections import Counter

_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_OUT        = Path(LOG_DIR) / f"tool_baselines_{_RUN_TS}.json"
LOG_OUT_LATEST = Path(LOG_DIR) / "tool_baselines_results_latest.json"
CKPT = args.ckpt or QWEN35_GRPO_FINAL
EVAL_BATCH_SIZE = 8

DIRECT_PROMPT = (
    "You are an expert in Logic and Physics. "
    "Think step by step and respond in the exact format:\n"
    "<reasoning>\n[Your step-by-step reasoning]\n</reasoning>\n"
    "<answer>\n[Final answer: letter / Yes/No/Unknown / number+unit]\n</answer>\n"
    "<explanation>\n[Concise explanation]\n</explanation>"
)

TOOL_PROMPT = (
    "You are an expert in Logic and Physics with access to tools.\n"
    "Think step by step inside <reasoning>...</reasoning>.\n"
    "For PHYSICS numerical problems: put the final numeric computation as a pure "
    "Python arithmetic expression inside <calc>...</calc> (e.g. <calc>0.5*2*3**2</calc>), "
    "then give <answer>number+unit</answer>.\n"
    "For LOGIC problems: give <answer>Yes/No/Unknown or the letter</answer>.\n"
    "Always end with:\n"
    "<answer>\n[final answer]\n</answer>"
)

# ── Sandboxed arithmetic evaluator (PAL-style calculator tool) ───────────────
_ALLOWED_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                   ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
                   ast.FloorDiv: operator.floordiv}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
                  "log": math.log, "log10": math.log10, "exp": math.exp, "pi": math.pi,
                  "abs": abs, "e": math.e}


def _safe_eval(expr: str):
    """Evaluate a pure arithmetic expression safely. Returns float or None."""
    expr = expr.strip().replace("^", "**").replace("×", "*").replace("·", "*")
    try:
        node = ast.parse(expr, mode="eval").body
        return _ev(node)
    except Exception:
        return None


def _ev(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_ev(node.left), _ev(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_ev(node.operand))
    if isinstance(node, ast.Name) and node.id in _ALLOWED_FUNCS:
        return _ALLOWED_FUNCS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _ALLOWED_FUNCS:
        return _ALLOWED_FUNCS[node.func.id](*[_ev(a) for a in node.args])
    raise ValueError("disallowed expression")


def _z3_verdict(question_text: str):
    """Derive Z3 yes/no verdict from premises WITHOUT seeing ground truth.
    Probe both hypotheses; verify(q,'Yes')==True iff z3 entails → 'Yes', etc."""
    try:
        from src.z3_engine import Z3Engine
        if Z3Engine.verify(question_text, "Yes") is True:
            return "Yes"
        if Z3Engine.verify(question_text, "No") is True:
            return "No"
    except Exception:
        pass
    return None


def _apply_tool(raw: str, subject: str, question: str):
    """Post-process a tool-arm generation. Returns (final_answer, tool_used)."""
    llm_ans = extract_answer_from_text(raw)
    if subject == "logic":
        v = _z3_verdict(question)
        if v is not None:
            return v, "z3"
        return llm_ans, "none"
    # physics: recompute <calc> with sandbox
    m = re.search(r"<calc>\s*(.*?)\s*</calc>", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        val = _safe_eval(m.group(1))
        if val is not None:
            _, unit = parse_numerical(llm_ans)   # keep model's unit
            num = f"{val:.6g}"
            return (f"{num} {unit}".strip() if unit else num), "calc"
    return llm_ans, "none"


def _generate_batch(model, tokenizer, questions, system_prompt, device):
    texts = []
    for q in questions:
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": q}]
        try:
            t = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                              tokenize=False, enable_thinking=False)
        except TypeError:
            t = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        texts.append(t)
    tokenizer.padding_side = "left"
    enc = tokenizer(texts, return_tensors="pt", truncation=True,
                    max_length=MAX_NEW_TOKENS * 2, padding=True).to(device)
    ilen = enc["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                             max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id,
                             eos_token_id=tokenizer.eos_token_id)
    return [tokenizer.decode(out[i][ilen:], skip_special_tokens=True) for i in range(len(questions))]


def _score(correct, subjects):
    n = len(correct)
    cp = cl = npz = nl = 0
    for ok, s in zip(correct, subjects):
        if s == "physics":
            npz += 1; cp += int(ok)
        elif s == "logic":
            nl += 1; cl += int(ok)
    return {"accuracy_overall": round(sum(correct) / n * 100, 2),
            "accuracy_physics": round(cp / npz * 100, 2) if npz else 0.0,
            "accuracy_logic": round(cl / nl * 100, 2) if nl else 0.0,
            "n_total": n, "n_physics": npz, "n_logic": nl, "n_correct": sum(correct)}


# ── Load ──────────────────────────────────────────────────────────────────────
print(f"[1] Loading val + model (ckpt={CKPT})")
from datasets import load_from_disk
val_ds = load_from_disk(VAL_DS)
N = 10 if args.smoke_test else len(val_ds)
samples = list(val_ds.select(range(min(N, len(val_ds)))))
questions = [s["prompt"][-1]["content"] if s.get("prompt") else s.get("question", "") for s in samples]
gts = [str(s.get("answer", "")) for s in samples]
subjects = [str(s.get("type", "")) for s in samples]

from src.model_loader import load_base_model
from peft import PeftModel
base, tokenizer = load_base_model(model_name=MODEL_NAME, dtype=torch.bfloat16,
                                  drop_vision=True, device_map="auto")
model = PeftModel.from_pretrained(base, CKPT, is_trainable=False) if Path(CKPT).exists() else base
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Arm A: direct ─────────────────────────────────────────────────────────────
print("[2] Arm A — LLM direct")
direct_correct = [False] * len(samples)
t0 = time.time()
for bs in range(0, len(samples), EVAL_BATCH_SIZE):
    idx = list(range(bs, min(bs + EVAL_BATCH_SIZE, len(samples))))
    raws = _generate_batch(model, tokenizer, [questions[i] for i in idx], DIRECT_PROMPT, device)
    for pos, i in enumerate(idx):
        direct_correct[i] = verify_answer(extract_answer_from_text(raws[pos]), gts[i],
                                          subject=subjects[i], question_text=questions[i], use_z3=True)
    print(f"    {min(bs+EVAL_BATCH_SIZE,len(samples))}/{len(samples)}", flush=True)
direct_min = round((time.time() - t0) / 60, 1)

# ── Arm B: tool-assisted ──────────────────────────────────────────────────────
print("[3] Arm B — LLM + tools (Z3 logic / Python calc physics)")
tool_correct = [False] * len(samples)
tool_used = Counter()
t0 = time.time()
for bs in range(0, len(samples), EVAL_BATCH_SIZE):
    idx = list(range(bs, min(bs + EVAL_BATCH_SIZE, len(samples))))
    raws = _generate_batch(model, tokenizer, [questions[i] for i in idx], TOOL_PROMPT, device)
    for pos, i in enumerate(idx):
        final, used = _apply_tool(raws[pos], subjects[i], questions[i])
        tool_used[used] += 1
        tool_correct[i] = verify_answer(final, gts[i], subject=subjects[i],
                                        question_text=questions[i], use_z3=True)
    print(f"    {min(bs+EVAL_BATCH_SIZE,len(samples))}/{len(samples)}  tools={dict(tool_used)}", flush=True)
tool_min = round((time.time() - t0) / 60, 1)

# ── Report ────────────────────────────────────────────────────────────────────
arm_a = _score(direct_correct, subjects)
arm_b = _score(tool_correct, subjects)
out = {
    "run_ts": _RUN_TS, "checkpoint": CKPT, "smoke_test": args.smoke_test,
    "arm_a_direct": {**arm_a, "elapsed_min": direct_min},
    "arm_b_tool": {**arm_b, "elapsed_min": tool_min, "tool_activations": dict(tool_used)},
    "delta_overall": round(arm_b["accuracy_overall"] - arm_a["accuracy_overall"], 2),
}
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
for p in (LOG_OUT, LOG_OUT_LATEST):
    with open(p, "w") as f:
        json.dump(out, f, indent=2)

print(f"\n{'='*60}\nTOOL-AUGMENTED BASELINE\n{'='*60}")
print(f"{'Arm':<18}{'Overall':>9}{'Physics':>9}{'Logic':>9}")
print(f"{'A: LLM direct':<18}{arm_a['accuracy_overall']:>8.2f}%{arm_a['accuracy_physics']:>8.2f}%{arm_a['accuracy_logic']:>8.2f}%")
print(f"{'B: LLM + tools':<18}{arm_b['accuracy_overall']:>8.2f}%{arm_b['accuracy_physics']:>8.2f}%{arm_b['accuracy_logic']:>8.2f}%")
print(f"Δ overall = {out['delta_overall']:+.2f} pp   tool activations: {dict(tool_used)}")
print(f"Saved → {LOG_OUT_LATEST}")
