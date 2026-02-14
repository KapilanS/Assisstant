"""
BFSI Call Center AI - RAG Layer (Tier 3)
Handles complex financial or policy-related queries.
Retrieves from structured knowledge documents.
Use ONLY for: interest explanations, EMI breakdowns, penalties, policy rules.
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np


class RAGRetriever:
    """
    Tier 3: RAG retrieval for complex queries.
    Retrieves relevant chunks from structured knowledge documents.
    """

    def __init__(self, config: dict, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent
        kb_path = config.get("knowledge_base_path", "data/rag_knowledge")
        self.knowledge_path = Path(kb_path) if Path(kb_path).is_absolute() else self.base_path / kb_path
        self.similarity_threshold = float(config.get("similarity_threshold", 0.7))
        self.max_context_chunks = int(config.get("max_context_chunks", 4))
        self.embedding_model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self._model = None
        self._chunks = []
        self._embeddings = None

    def _load_knowledge(self) -> List[dict]:
        """Load and chunk knowledge documents."""
        if self._chunks:
            return self._chunks
        chunks = []
        if not self.knowledge_path.exists():
            return chunks
        for f in self.knowledge_path.glob("*.md"):
            with open(f, "r", encoding="utf-8") as fp:
                content = fp.read()
            # Chunk by ## or ### sections
            parts = []
            current = []
            for line in content.split("\n"):
                if line.startswith("## ") or line.startswith("### "):
                    if current:
                        text = "\n".join(current).strip()
                        if text:
                            parts.append(text)
                    current = [line.lstrip("# ").strip()]
                else:
                    current.append(line)
            if current:
                text = "\n".join(current).strip()
                if text:
                    parts.append(text)
            if parts:
                for part in parts:
                    if part:
                        chunks.append({"source": f.name, "title": "", "text": part})
            else:
                # Fallback: chunk by paragraph
                for para in content.split("\n\n"):
                    if para.strip():
                        chunks.append({"source": f.name, "title": "", "text": para.strip()})
        self._chunks = [c for c in chunks if c.get("text")]
        return self._chunks

    def _get_model(self):
        """Lazy load embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                raise ImportError("sentence-transformers required. pip install sentence-transformers")
        return self._model

    def _build_embeddings(self) -> np.ndarray:
        """Build embeddings for all chunks."""
        if self._embeddings is not None:
            return self._embeddings
        chunks = self._load_knowledge()
        if not chunks:
            self._embeddings = np.array([])
            return self._embeddings
        texts = [c["title"] + " " + c["text"] for c in chunks]
        model = self._get_model()
        self._embeddings = model.encode(texts)
        return self._embeddings

    def retrieve(self, query: str) -> List[dict]:
        """
        Retrieve relevant chunks for query.
        Returns list of chunks with text and source.
        """
        chunks = self._load_knowledge()
        if not chunks:
            return []
        embeddings = self._build_embeddings()
        if embeddings.size == 0:
            return []
        model = self._get_model()
        query_emb = model.encode([query])
        scores = np.dot(embeddings, query_emb.T).flatten() / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-9
        )
        top_indices = np.argsort(scores)[::-1][: self.max_context_chunks]
        results = []
        for i in top_indices:
            if scores[i] >= self.similarity_threshold:
                results.append({
                    **chunks[i],
                    "score": float(scores[i]),
                })
        return results

    def get_context(self, query: str) -> str:
        """Get concatenated context for RAG generation."""
        results = self.retrieve(query)
        if not results:
            return ""
        return "\n\n".join(r["text"] for r in results)
