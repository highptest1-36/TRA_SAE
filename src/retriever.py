"""
TRA-SAE Retriever
==================
TF-IDF based similarity search over the 1945 training examples.
Used by Phase 4 agent to build few-shot context for each inference query.

Usage:
    from src.retriever import Retriever
    r = Retriever(train_path="processed_data/exact_train")
    r.build(cache_path="logs/retriever_cache.pkl")
    examples = r.retrieve("What is Newton's second law?", top_k=3)
"""
from __future__ import annotations

import os
import re
import pickle
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extract_user_content(prompt: list[dict]) -> str:
    """Return the 'user' turn content from a chat-formatted prompt list."""
    for msg in prompt:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg["content"]
    return ""


def _strip_premises(text: str) -> str:
    """Remove the 'Premises:' block and return only the question sentence.

    Physics questions usually have no premises; logic questions do.
    Stripping premises reduces noise during similarity search.
    """
    # Find 'Question:' line and return everything from there
    m = re.search(r"(Question:.*)", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

# Cache format version — bump this to invalidate any pre-v2 cache files
_CACHE_VERSION = "v2_subject"


class Retriever:
    """TF-IDF cosine-similarity retriever over training examples.

    v2 additions:
      - Stores the 'type' field as 'subjects' list.
      - retrieve() accepts an optional subject= filter so logic queries
        only see logic examples and physics queries only see physics examples.
    """

    def __init__(self, train_path: str) -> None:
        self.train_path = train_path
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix: Any = None
        self.questions: list[str] = []   # full user content for display
        self.queries: list[str] = []     # stripped text used for TF-IDF
        self.answers: list[str] = []
        self.explanations: list[str] = []
        self.types: list[str] = []
        self.subjects: list[str] = []    # 'logic' | 'physics' | 'unknown'

    # ------------------------------------------------------------------
    # Build / load index
    # ------------------------------------------------------------------

    def build(self, cache_path: str | None = None) -> None:
        """Fit TF-IDF on training data.  Loads from cache if available.

        Cache version key is checked on load; a stale cache (built with
        a prior code version) is automatically rebuilt.
        """
        if cache_path and os.path.exists(cache_path):
            self._load(cache_path)
            print(f"[Retriever] Loaded index from cache ({len(self.questions)} docs)")
            return

        from datasets import load_from_disk  # lazy import to avoid Colab issues
        ds = load_from_disk(self.train_path)

        self.questions    = [_extract_user_content(s["prompt"]) for s in ds]
        self.queries      = [_strip_premises(q) for q in self.questions]
        self.answers      = [s["answer"] for s in ds]
        self.explanations = [s.get("explanation", "") for s in ds]
        self.types        = [s.get("type", "unknown") for s in ds]
        self.subjects     = self.types   # alias — 'logic' | 'physics'

        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        self.matrix = self.vectorizer.fit_transform(self.queries)
        print(
            f"[Retriever] Built TF-IDF index: {len(self.questions)} docs, "
            f"{self.matrix.shape[1]} features"
        )
        by_subj = {s: self.subjects.count(s) for s in set(self.subjects)}
        print(f"[Retriever] Subject distribution: {by_subj}")

        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            self._save(cache_path)
            print(f"[Retriever] Saved cache → {cache_path}")

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        subject: str | None = None,
    ) -> list[dict]:
        """Return top-k similar training examples.

        Args:
            query:    The question text to search for.
            top_k:    Number of examples to return.
            subject:  If set ('logic' or 'physics'), restrict search to examples
                      of that subject.  None → search across all subjects.

        Returns:
            List of dicts: question, answer, explanation, type, score.
        """
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("Call Retriever.build() before retrieve().")

        stripped = _strip_premises(query)
        q_vec    = self.vectorizer.transform([stripped])
        scores   = cosine_similarity(q_vec, self.matrix).flatten()

        # Subject filtering — compute on a boolean mask
        if subject and self.subjects:
            valid_indices = [
                i for i, s in enumerate(self.subjects)
                if s == subject
            ]
            if valid_indices:
                # Zero out scores for non-matching subjects
                import numpy as np
                mask = np.zeros_like(scores, dtype=bool)
                mask[valid_indices] = True
                scores = scores * mask   # zero out other subjects
            else:
                # No examples of this subject → no filtering
                pass

        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0 and subject:
                break   # no valid filtered results
            results.append({
                "question":    self.questions[idx],
                "answer":      self.answers[idx],
                "explanation": self.explanations[idx],
                "type":        self.types[idx],
                "score":       float(scores[idx]),
            })
        return results

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "_version":     _CACHE_VERSION,
                "vectorizer":   self.vectorizer,
                "matrix":       self.matrix,
                "questions":    self.questions,
                "queries":      self.queries,
                "answers":      self.answers,
                "explanations": self.explanations,
                "types":        self.types,
                "subjects":     self.subjects,
            }, f, protocol=4)

    def _load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Version check — if stale, force rebuild on next build() call
        if data.get("_version") != _CACHE_VERSION:
            print(
                f"[Retriever] Cache version mismatch "
                f"(found '{data.get('_version')}', need '{_CACHE_VERSION}') "
                f"— cache will be rebuilt."
            )
            # Remove stale file so build() falls through to rebuild
            import os as _os
            try:
                _os.remove(path)
            except OSError:
                pass
            return
        self.vectorizer   = data["vectorizer"]
        self.matrix       = data["matrix"]
        self.questions    = data["questions"]
        self.queries      = data.get("queries", data["questions"])
        self.answers      = data["answers"]
        self.explanations = data["explanations"]
        self.types        = data["types"]
        self.subjects     = data.get("subjects", self.types)
