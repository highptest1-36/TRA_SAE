✅ KẾ HOẠCH THỰC HIỆM HOÀN CHỈNH & CHI TIẾT NHẤT – TRA-SAE (v2.0)
Nguyên tắc cốt lõi (đúng như bạn yêu cầu):

100% PyTorch + Unsloth cho toàn bộ training pipeline (Phase 1–3) → nhanh nhất, ổn định nhất, tiết kiệm VRAM nhất trên Colab Pro+ A100 40GB, và tối ưu P1/P3 cho cuộc thi EXACT.
TensorFlow/Keras chỉ dùng ở 1 file duy nhất: reward_evaluator_keras.py (Self-Evaluator cho GRPO reward function). Đây là cách nhỏ nhất, không ảnh hưởng đến tốc độ training chính.

Bạn chỉ cần copy-paste từng phần, chạy tuần tự trên Colab Pro+.
1. Tổng quan Pipeline (5 Phases)






















































PhaseThời gian (A100 40GB)Mục tiêuCông cụ chínhTensorFlow?01–2 giờData prepHuggingFace DatasetsKhông14–8 giờWarm-up SFT / DistillationUnsloth LoRAKhông26–12 giờSASFT (Qwen-Scope SAE)Unsloth + Qwen-Scope PyTorch SAEKhông312–24 giờ (chia 2 session)GRPOUnsloth GRPO + Keras evaluatorChỉ ở reward44–8 giờTRA-SAE AgentLangGraph + Z3 + SymPy + SAE hookKhông51 ngàyTest & SubmitGradio / HF SpaceKhông
Tổng thời gian thực tế: 4–6 ngày (có thể overnight).
Model chính: Qwen/Qwen3-8B-Instruct + Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50 (PyTorch).
2. Phase 0: Data Preparation (1–2 giờ)
Tạo dataset HuggingFace với format Unsloth/GRPO yêu cầu.
Python# Notebook: data_prep.ipynb
from datasets import load_dataset
import json

SYSTEM_PROMPT = """Bạn là học sinh xuất sắc môn Logic & Vật lý. 
Hãy suy nghĩ từng bước, giải thích rõ ràng bằng tiếng Việt, và trả lời đúng format:
<reasoning>...</reasoning>
<answer>...</answer>
<explanation>...</explanation>"""

def preprocess(example):
    # ... (giống code tôi đã gửi trước, đã có trong tin nhắn cũ)
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
        "answer": example["answer"],
        "cot": example.get("cot", ""),
        "explanation": example.get("explanation", ""),
        "type": "logic" if "premises-NL" in example else "physics"
    }

dataset = load_dataset("json", data_files="your_2000_samples.jsonl")
train_data = dataset["train"].map(preprocess).train_test_split(test_size=0.1, seed=42)
train_data["train"].push_to_hub("yourusername/exact-2000-train")
train_data["test"].push_to_hub("yourusername/exact-2000-val")
3. Phase 1: Warm-up SFT (4–8 giờ)
Dùng Unsloth LoRA (không cần teacher offline).
Notebook chính thức: Unsloth_SFT_Qwen3-8B.ipynb
Python# Cài Unsloth
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("Qwen/Qwen3-8B-Instruct", dtype="float16", load_in_4bit=True)

model = FastLanguageModel.get_peft_model(model, r=64, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    dataset_text_field="prompt",  # Unsloth tự apply chat template
    max_seq_length=2048,
    ...
)
trainer.train()
model.save_pretrained("tra-sae-base")
4. Phase 2: SASFT (Qwen-Scope SAE-guided SFT) (6–12 giờ)
Sử dụng SAE PyTorch chính thức.
Python# Load SAE (PyTorch)
from qwen_scope import load_sae   # hoặc code từ HF collection
sae = load_sae("Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50", device="cuda")

# Trong training loop (Unsloth hỗ trợ custom loss)
# Thêm auxiliary loss: encourage features physics/logic
def sae_aux_loss(hidden_states, sae):
    features = sae.encode(hidden_states)
    # Ví dụ: reward feature "parallel_resistors", "regulation_13"
    return -features[:, physics_feature_ids].mean()   # custom theo concept

# Kết hợp vào SFTTrainer custom loss
Checkpoint: tra-sae-sasft
5. Phase 3: GRPO + TensorFlow Self-Evaluator (12–24 giờ)
Đây là phần duy nhất dùng TensorFlow (chỉ 1 file).
Tạo file reward_evaluator_keras.py (copy-paste ngay):
Python# reward_evaluator_keras.py
import tensorflow as tf
from tensorflow import keras
import numpy as np

class ExplanationEvaluator(keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = keras.layers.Embedding(32000, 256)
        self.lstm = keras.layers.Bidirectional(keras.layers.LSTM(128))
        self.dense = keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        return self.dense(x)

# Load model nhỏ (chỉ ~5M params)
evaluator = ExplanationEvaluator()
evaluator.build((None, 512))  # max length
evaluator.load_weights("explanation_evaluator.keras")  # bạn train nhanh 1 lần trên 500 samples

def get_explanation_score(text: str) -> float:
    """Trả reward P2 (0.0 - 1.0)"""
    tokens = tokenizer_keras(text)[:512]   # tokenizer nhỏ (có thể dùng SentencePiece)
    score = evaluator.predict(np.array([tokens]), verbose=0)[0][0]
    return float(score)
Reward function GRPO (PyTorch + gọi Keras):
Pythondef grpo_reward(completions, prompts, answers, **kwargs):
    rewards = []
    for comp, ans in zip(completions, answers):
        # 1. Correctness (symbolic verifier)
        correct = 1.0 if symbolic_verify(comp["answer"], ans) else 0.0
        
        # 2. Explanation quality (chỉ phần TensorFlow)
        expl_score = get_explanation_score(comp["explanation"])   # gọi Keras
        
        # 3. SAE feature alignment (PyTorch)
        sae_score = sae_feature_alignment(comp["hidden_states"])
        
        total = correct * 1.0 + expl_score * 0.5 + sae_score * 0.3
        rewards.append(total)
    return rewards
Dùng notebook Unsloth_GRPO_Qwen3-8B_FP8.ipynb và pass reward function trên.
Checkpoint cuối: tra-sae-final
6. Phase 4: Xây dựng TRA-SAE Agent (4–8 giờ)
Dùng LangGraph (PyTorch-only):

Orchestrator (Qwen3-8B fine-tuned)
Skills: Z3, SymPy, SAE Interpreter (PyTorch hook), Self-Evaluator (gọi file Keras)
Correction loop + SAE steering

7. Phase 5: Test & Deploy
Test trên val set → deploy FastAPI/Gradio → submit API endpoint.