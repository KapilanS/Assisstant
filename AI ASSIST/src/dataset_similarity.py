"""
BFSI Call Center AI - Alpaca Dataset Similarity Check (Tier 1)
If strong similarity match is found, return stored response DIRECTLY without modification.
Uses embedding-based similarity for lightweight local execution.
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class DatasetSimilarityChecker:
    """
    Tier 1: Primary response layer.
    Checks user query against Alpaca BFSI dataset.
    Returns stored output EXACTLY when similarity >= threshold.
    """

    def __init__(self, config: dict, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent
        ds_rel = config.get("dataset_path", "data/alpaca_dataset.json")
        self.dataset_path = Path(ds_rel) if Path(ds_rel).is_absolute() else self.base_path / ds_rel
        self.threshold = float(config.get("threshold", 0.85))
        self.top_k = int(config.get("top_k", 3))
        self.embedding_model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self._model = None
        self._embeddings = None
        self._dataset = None

    def _load_dataset(self) -> list:
        """Load Alpaca-formatted dataset."""
        if self._dataset is not None:
            return self._dataset
        path = self.dataset_path
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            self._dataset = json.load(f)
        if not isinstance(self._dataset, list):
            self._dataset = [self._dataset]
        return self._dataset

    def _get_model(self):
        """Lazy load sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for similarity matching. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def _build_embeddings(self) -> np.ndarray:
        """Build embeddings for all query representations in dataset."""
        if self._embeddings is not None:
            return self._embeddings
        dataset = self._load_dataset()
        # Use instruction + input as searchable text (represents user intent)
        texts = []
        for item in dataset:
            inst = item.get("instruction", "")
            inp = item.get("input", "")
            text = f"{inst} {inp}".strip()
            if not text:
                text = inp or inst
            texts.append(text)
        model = self._get_model()
        self._embeddings = model.encode(texts)
        return self._embeddings

    def search(self, query: str) -> Tuple[Optional[str], float]:
        """
        Search dataset for strong similarity match.
        Returns (output_text, similarity_score) or (None, 0.0) if no strong match.
        When match >= threshold, return the stored output EXACTLY without modification.
        """
        dataset = self._load_dataset()
        if not dataset:
            return None, 0.0

        embeddings = self._build_embeddings()
        model = self._get_model()
        query_emb = model.encode([query])
        scores = np.dot(embeddings, query_emb.T).flatten() / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-9
        )
        top_indices = np.argsort(scores)[::-1][: self.top_k]

        best_idx = top_indices[0]
        best_score = float(scores[best_idx])

        if best_score >= self.threshold:
            # STRICT: Return stored response exactly, no modification
            output = dataset[best_idx].get("output", "")
            return output, best_score
        return None, best_score

    def get_best_match_info(self, query: str) -> Optional[dict]:
        """Return full best match info (for debugging/logging)."""
        dataset = self._load_dataset()
        embeddings = self._build_embeddings()
        model = self._get_model()
        query_emb = model.encode([query])
        scores = np.dot(embeddings, query_emb.T).flatten() / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-9
        )
        best_idx = int(np.argmax(scores))
        return {
            "index": best_idx,
            "score": float(scores[best_idx]),
            "instruction": dataset[best_idx].get("instruction"),
            "input": dataset[best_idx].get("input"),
            "output": dataset[best_idx].get("output"),
        }
