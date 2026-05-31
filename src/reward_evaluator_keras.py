"""
TRA-SAE Reward Evaluator  ── TensorFlow/Keras
===============================================
Lightweight Bi-LSTM model for scoring explanation quality (P2 reward).

This is the ONLY TensorFlow file in the entire project.
  * ~5 M parameters, runs in < 10 ms per batch on CPU/GPU
  * Trained once on ~500 labelled explanation samples, then reused in GRPO

Usage:
    from src.reward_evaluator_keras import get_explanation_score

    score = get_explanation_score("Because premise 1 implies ...", weights_path="...")
    # Returns float in [0.0, 1.0]
"""
from __future__ import annotations

import os
import numpy as np

# ── Suppress TF CUDA init messages ───────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # silence TF C++ logs

# Lazy singletons — TF is imported only when first needed
_evaluator_instance = None


# ── Model Definition ─────────────────────────────────────────────────────────

class ExplanationEvaluator:
    """Wrapper around a Keras Bi-LSTM explanation-quality classifier."""

    def __init__(self, vocab_size: int = 32000, max_len: int = 256):
        import tensorflow as tf
        from tensorflow import keras

        # Force TF to CPU — even if set_visible_devices() fails (Colab pre-inits TF),
        # placing all ops under tf.device('/CPU:0') prevents GPU memory allocation.
        try:
            tf.config.set_visible_devices([], 'GPU')
        except RuntimeError:
            pass  # already initialized; rely on explicit CPU placement below

        self._tf = tf   # cache reference for use in predict()
        self.max_len = max_len
        with tf.device('/CPU:0'):
            self.model = keras.Sequential([
                keras.layers.Embedding(vocab_size, 128, input_length=max_len),
                keras.layers.Bidirectional(keras.layers.LSTM(64)),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(1,  activation="sigmoid"),
            ], name="explanation_evaluator")
            self.model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

    def build(self, weights_path: str | None = None) -> None:
        """Build graph and optionally load pre-trained weights."""
        with self._tf.device('/CPU:0'):
            self.model.build((None, self.max_len))
        if weights_path and os.path.exists(weights_path):
            try:
                self.model.load_weights(weights_path)
                print(f"[ExplanationEvaluator] Loaded weights: {weights_path}")
            except Exception as exc:
                print(f"[ExplanationEvaluator] Could not load weights: {exc}")
                print("  → Using random weights (scores will be noisy until trained)")
        else:
            print("[ExplanationEvaluator] No weights found — using random init.")
            print("  → Run train_evaluator() on labelled explanations first.")

    def predict(self, token_ids: np.ndarray) -> np.ndarray:
        # Direct __call__ (avoids tf.function retrace accumulation) placed on CPU
        # so TF never allocates VRAM that would compete with PyTorch's allocator.
        with self._tf.device('/CPU:0'):
            return self.model(token_ids, training=False).numpy()

    def fit(self, token_ids: np.ndarray, labels: np.ndarray,
            epochs: int = 15, batch_size: int = 32) -> None:
        self.model.fit(
            token_ids, labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
        )

    def save_weights(self, path: str) -> None:
        self.model.save_weights(path)
        print(f"[ExplanationEvaluator] Weights saved: {path}")


# ── Tokenizer Helper ─────────────────────────────────────────────────────────

def _simple_tokenize(text: str, vocab_size: int = 32000,
                     max_len: int = 256) -> list[int]:
    """Character-level fallback tokenizer.

    Replace with proper BPE tokenizer (e.g. Qwen2.5 tokenizer) for
    better quality.  This is only used when no HF tokenizer is passed.
    """
    tokens = [ord(c) % vocab_size for c in text[:max_len]]
    tokens += [0] * (max_len - len(tokens))   # pad
    return tokens[:max_len]


# ── Public API ───────────────────────────────────────────────────────────────

def get_evaluator(weights_path: str | None = None,
                  vocab_size: int = 32000,
                  max_len: int = 256) -> ExplanationEvaluator:
    """Return the singleton evaluator, creating it on first call."""
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = ExplanationEvaluator(
            vocab_size=vocab_size, max_len=max_len
        )
        _evaluator_instance.build(weights_path)
        # NOTE: warmup intentionally removed — calling predict() here would trigger
        # TF GPU init before training, pre-allocating VRAM and competing with PyTorch.
        # The first reward call bears a one-time ~1-2s tf.function compile cost instead.
    return _evaluator_instance


def get_explanation_score(
    text: str,
    weights_path: str | None = None,
    tokenizer=None,
) -> float:
    """Score an explanation text.  Returns float in [0.0, 1.0].

    Args:
        text:          The explanation string to score.
        weights_path:  Path to .keras weights file (optional).
        tokenizer:     HuggingFace tokenizer for proper BPE encoding.
                       Falls back to _simple_tokenize if None.
    """
    if not text or len(text.strip()) < 5:
        return 0.0

    evaluator = get_evaluator(weights_path)
    max_len    = evaluator.max_len

    if tokenizer is not None:
        enc    = tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        tokens = enc["input_ids"]
    else:
        tokens = np.array([_simple_tokenize(text, max_len=max_len)])

    score = evaluator.predict(tokens)[0][0]
    return float(np.clip(score, 0.0, 1.0))


def train_evaluator(
    explanations: list[str],
    labels: list[int],
    save_path: str,
    vocab_size: int = 32000,
    max_len: int = 256,
    tokenizer=None,
) -> None:
    """Quick-train the evaluator on labelled explanation strings.

    Args:
        explanations:  List of explanation strings.
        labels:        Parallel list of 0/1 labels (0 = poor, 1 = good).
        save_path:     Where to save the trained weights (.keras).
        tokenizer:     HuggingFace tokenizer (optional).
    """
    evaluator = get_evaluator(vocab_size=vocab_size, max_len=max_len)

    if tokenizer is not None:
        enc    = tokenizer(
            explanations,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        tokens = enc["input_ids"]
    else:
        tokens = np.array([
            _simple_tokenize(e, vocab_size=vocab_size, max_len=max_len)
            for e in explanations
        ])

    labels_arr = np.array(labels, dtype=np.float32)
    evaluator.fit(tokens, labels_arr, epochs=15, batch_size=32)
    evaluator.save_weights(save_path)
    print(f"[train_evaluator] Done — weights saved to {save_path}")
