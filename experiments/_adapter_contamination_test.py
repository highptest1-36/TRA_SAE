"""
One-off diagnostic: does step0's shared-base_model reuse contaminate cfg2/cfg3?

step0 wraps a SINGLE shared base_model with cfg1's LoRA, does `del model`, then
wraps the SAME base_model with cfg2's LoRA. PeftModel injects adapters IN PLACE,
so cfg1's layers may persist. This test compares:

  CLEAN  : fresh base + cfg2 adapter            (correct)
  DIRTY  : base + cfg1, del, + cfg2 (step0 way) (suspect)
  CLEAN1 : fresh base + cfg1 adapter            (reference)

If CLEAN != DIRTY  -> contamination is real, step0 must reload base per config.
If DIRTY == CLEAN1 -> cfg2 is actually running cfg1 (worst case).
"""
import sys, torch
sys.path.insert(0, "/content/drive/MyDrive/TRA-SAE")
from src.config import MODEL_NAME, QWEN35_SFT_FINAL, QWEN35_SFT_LOGIC_FINAL
from src.model_loader import load_base_model
from peft import PeftModel

PROMPT = ("You are an expert in Logic and Physics. Answer concisely.\n"
          "Question: A 2 kg object accelerates at 3 m/s^2. What net force acts on it?")

def gen(model, tok):
    msgs = [{"role": "user", "content": PROMPT}]
    try:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=120, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

def active_adapter_info(m, tag):
    # report which adapters are attached + active
    try:
        names = list(m.peft_config.keys())
    except Exception:
        names = "?"
    act = getattr(m, "active_adapters", getattr(m, "active_adapter", "?"))
    print(f"  [{tag}] adapters={names} active={act}")

print("=== CLEAN1: fresh base + cfg1 (SFT) ===")
b1, tok = load_base_model(model_name=MODEL_NAME, dtype=torch.bfloat16, drop_vision=True, device_map="auto")
m1 = PeftModel.from_pretrained(b1, QWEN35_SFT_FINAL, is_trainable=False); m1.eval()
active_adapter_info(m1, "clean1")
clean1 = gen(m1, tok)
del m1, b1; import gc; gc.collect(); torch.cuda.empty_cache()

print("=== CLEAN2: fresh base + cfg2 (SFT_LOGIC) ===")
b2, tok = load_base_model(model_name=MODEL_NAME, dtype=torch.bfloat16, drop_vision=True, device_map="auto")
m2 = PeftModel.from_pretrained(b2, QWEN35_SFT_LOGIC_FINAL, is_trainable=False); m2.eval()
active_adapter_info(m2, "clean2")
clean2 = gen(m2, tok)
del m2, b2; gc.collect(); torch.cuda.empty_cache()

print("=== DIRTY: base + cfg1, del, + cfg2 (replicates step0 reuse) ===")
bd, tok = load_base_model(model_name=MODEL_NAME, dtype=torch.bfloat16, drop_vision=True, device_map="auto")
md1 = PeftModel.from_pretrained(bd, QWEN35_SFT_FINAL, is_trainable=False); md1.eval()
_ = gen(md1, tok)            # use it like step0 does
del md1; gc.collect(); torch.cuda.empty_cache()   # step0's cleanup (model is not base_model)
md2 = PeftModel.from_pretrained(bd, QWEN35_SFT_LOGIC_FINAL, is_trainable=False); md2.eval()
active_adapter_info(md2, "dirty2")
dirty2 = gen(md2, tok)

print("\n" + "="*70)
print("CLEAN2 (correct cfg2):\n ", repr(clean2[:200]))
print("DIRTY2 (step0 cfg2):\n ", repr(dirty2[:200]))
print("CLEAN1 (cfg1 ref):\n ", repr(clean1[:200]))
print("="*70)
print("cfg2 clean == cfg2 dirty ?  ->", clean2.strip() == dirty2.strip())
print("cfg2 dirty == cfg1 clean ?  ->", dirty2.strip() == clean1.strip())
if clean2.strip() == dirty2.strip():
    print("\nVERDICT: NO contamination — step0 reuse is SAFE (peft overwrites 'default').")
else:
    print("\nVERDICT: CONTAMINATION CONFIRMED — step0 must reload fresh base per LoRA config.")
