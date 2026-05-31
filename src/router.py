"""
TRA-SAE Subject Router
=======================
Classifies questions into 'physics' or 'logic' before routing to the
appropriate LoRA adapter in Agent v2.

Architecture:
  - TF-IDF (max_features=5000, ngram=(1,2)) + Logistic Regression (C=1.0)
  - Trained on all 1945 training samples using the 'type' label field
  - If model confidence < CONFIDENCE_THRESH → fallback keyword heuristic
  - Saved/loaded as a single pickle: checkpoints/router.pkl

Usage:
    from src.router import SubjectRouter
    router = SubjectRouter()
    router.fit(train_dataset_path)             # trains and saves router
    subject, conf = router.predict("What is ...")  # 'physics' | 'logic', float

    # Or load pre-trained:
    router = SubjectRouter.load("checkpoints/router.pkl")
    subject, conf = router.predict(question_text)
"""
from __future__ import annotations

import os
import re
import pickle
import logging
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

logger = logging.getLogger("tra-sae.router")

# ── Keyword heuristic sets ────────────────────────────────────────────────────
_PHYSICS_KEYWORDS = frozenset([
    "voltage", "current", "resistance", "capacitor", "inductor",
    "capacitance", "inductance", "electric", "magnetic", "charge",
    "force", "energy", "power", "frequency", "wavelength", "circuit",
    "ohm", "ampere", "volt", "watt", "joule", "newton", "field",
    "wire", "coil", "solenoid", "transformer", "flux", "potential",
    "capacitance", "acceleration", "velocity", "momentum", "mass",
    "temperature", "heat", "pressure", "wave", "photon", "electron",
    "proton", "neutron", "nucleus", "atom", "resistor", "battery",
    "emf", "impedance", "reactance", "dielectric", "permittivity",
    "permeability", "resonance", "oscillation", "period", "hertz",
    "kilowatt", "milliamp", "microfarad", "ohm's", "kirchhoff",
    "faraday", "coulomb", "tesla", "weber", "lenz",
])

_LOGIC_KEYWORDS = frozenset([
    "premise", "premises", "conclusion", "follows", "implies",
    "therefore", "hence", "infer", "deduce", "valid",
    "forall", "exists", "∀", "∃", "¬", "→", "∧", "∨",
    "true", "false", "unknown", "statement", "argument",
    "contrapositive", "converse", "inverse", "syllogism",
    "modus", "ponens", "tollens", "logic", "proposition",
    "predicate", "quantifier", "first-order", "boolean",
    "if then", "if and only if", "iff", "biconditional",
    "does it follow", "is the following", "must be true",
    "can we conclude", "consistent", "inconsistent",
])


def _keyword_predict(text: str) -> tuple[str, float]:
    """Rule-based fallback: count physics vs logic keywords."""
    lower = text.lower()
    p_count = sum(1 for kw in _PHYSICS_KEYWORDS if kw in lower)
    l_count = sum(1 for kw in _LOGIC_KEYWORDS   if kw in lower)
    total = max(p_count + l_count, 1)
    if p_count >= l_count:
        return "physics", p_count / total
    return "logic", l_count / total


def _preprocess(text: str) -> str:
    """Minimal text cleaning for TF-IDF input."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# SubjectRouter
# ──────────────────────────────────────────────────────────────────────────────

class SubjectRouter:
    """TF-IDF + Logistic Regression subject classifier.

    Predicts whether a question belongs to 'physics' or 'logic'.
    Falls back to keyword heuristic when model confidence is low.
    """

    CONFIDENCE_THRESH = 0.80   # below this → use keyword fallback

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_path: str,
        save_path: str | None = None,
    ) -> dict:
        """Train on processed_data/exact_train, optionally evaluate on val.

        Args:
            train_path:  Path to HuggingFace dataset directory (exact_train).
            save_path:   If set, serialise the fitted router here.

        Returns:
            dict with 'train_accuracy' and optional 'val_accuracy'.
        """
        from datasets import load_from_disk

        logger.info(f"[router] Loading training data from {train_path}")
        ds = load_from_disk(train_path)

        texts  = []
        labels = []
        for sample in ds:
            user_content = ""
            for msg in sample["prompt"]:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_content = msg["content"]
                    break
            texts.append(_preprocess(user_content))
            labels.append(sample.get("type", "unknown"))

        logger.info(
            f"[router] Training samples: {len(texts)}  "
            f"({labels.count('physics')} physics, {labels.count('logic')} logic)"
        )

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
                analyzer="word",
            )),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="auto",
            )),
        ])
        self.pipeline.fit(texts, labels)

        train_preds = self.pipeline.predict(texts)
        train_acc   = accuracy_score(labels, train_preds)
        logger.info(f"[router] Train accuracy: {train_acc:.4f}")
        print(f"[Router] Train accuracy: {train_acc:.4f}")
        print(classification_report(labels, train_preds))

        if save_path:
            self.save(save_path)
            logger.info(f"[router] Saved → {save_path}")

        return {"train_accuracy": train_acc}

    def evaluate(self, val_path: str) -> dict:
        """Evaluate on a held-out HuggingFace dataset split."""
        if self.pipeline is None:
            raise RuntimeError("Call fit() or load() first.")

        from datasets import load_from_disk
        ds = load_from_disk(val_path)

        texts, labels = [], []
        for sample in ds:
            for msg in sample["prompt"]:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    texts.append(_preprocess(msg["content"]))
                    break
            labels.append(sample.get("type", "unknown"))

        preds = self.pipeline.predict(texts)
        acc   = accuracy_score(labels, preds)
        logger.info(f"[router] Val accuracy: {acc:.4f}")
        print(f"[Router] Val accuracy: {acc:.4f}")
        print(classification_report(labels, preds))

        return {
            "val_accuracy": acc,
            "report": classification_report(labels, preds, output_dict=True),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, question_text: str) -> tuple[str, float]:
        """Predict subject + confidence for a single question.

        Returns:
            (subject, confidence)  where subject ∈ {'physics', 'logic'}
            and confidence ∈ [0, 1].

        Falls back to keyword heuristic if model confidence is below
        CONFIDENCE_THRESH or if the router has not been fitted yet.
        """
        if self.pipeline is None:
            logger.warning("[router] Not fitted — using keyword fallback.")
            return _keyword_predict(question_text)

        text  = _preprocess(question_text)
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_

        best_idx  = int(np.argmax(proba))
        subject   = str(classes[best_idx])
        confidence = float(proba[best_idx])

        if confidence < self.CONFIDENCE_THRESH:
            kw_subject, kw_conf = _keyword_predict(question_text)
            logger.debug(
                f"[router] Low confidence ({confidence:.2f}) for '{subject}' "
                f"→ keyword fallback → '{kw_subject}' ({kw_conf:.2f})"
            )
            # Blend: trust model unless keyword is very certain
            if kw_conf > confidence:
                return kw_subject, kw_conf

        return subject, confidence

    def predict_batch(self, questions: list[str]) -> list[tuple[str, float]]:
        """Batch predict for efficiency."""
        if self.pipeline is None:
            return [_keyword_predict(q) for q in questions]

        texts = [_preprocess(q) for q in questions]
        probas  = self.pipeline.predict_proba(texts)
        classes = self.pipeline.classes_

        results = []
        for proba in probas:
            best_idx   = int(np.argmax(proba))
            subject    = str(classes[best_idx])
            confidence = float(proba[best_idx])

            if confidence < self.CONFIDENCE_THRESH:
                kw_subject, kw_conf = _keyword_predict(questions[len(results)])
                if kw_conf > confidence:
                    results.append((kw_subject, kw_conf))
                    continue

            results.append((subject, confidence))

        return results

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f, protocol=4)
        print(f"[Router] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> "SubjectRouter":
        """Load a previously trained router from disk."""
        router = cls()
        with open(path, "rb") as f:
            router.pipeline = pickle.load(f)
        logger.info(f"[router] Loaded from {path}")
        return router


# ── Module-level singleton (lazy init) ───────────────────────────────────────
_default_router: SubjectRouter | None = None


def get_router(
    router_path: str | None = None,
    train_path: str | None = None,
) -> SubjectRouter:
    """Return the module-level cached router.

    Loads from router_path if given, otherwise trains if train_path given,
    otherwise returns an unfitted router (keyword fallback only).
    """
    global _default_router
    if _default_router is None:
        router = SubjectRouter()
        if router_path and os.path.isfile(router_path):
            try:
                router = SubjectRouter.load(router_path)
            except Exception as e:
                logger.warning(f"[router] Failed to load from {router_path}: {e}")
        elif train_path:
            router.fit(train_path, save_path=router_path)
        _default_router = router
    return _default_router
