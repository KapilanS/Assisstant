"""
BFSI Call Center AI - Response Orchestrator
Implements STRICT priority order (MUST NOT BE VIOLATED):
  Tier 1: Dataset Match → Return stored response exactly
  Tier 2: Fine-Tuned SLM → If no dataset match
  Tier 3: RAG Retrieval → For complex queries (interest, EMI, penalties, policy)
"""

import sys
from pathlib import Path
from typing import Optional

import yaml

# Ensure project root in path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Keywords indicating complex financial/policy queries requiring RAG
RAG_TRIGGER_KEYWORDS = [
    "interest rate", "interest calculation", "emi formula", "emi breakdown",
    "principal", "interest component", "penalty", "penalties", "late payment charge",
    "foreclosure charge", "prepayment charge", "processing fee", "policy",
    "regulatory", "compliant", "kyc", "grievance", "compound interest",
    "fixed vs floating", "repo rate", "lvt", "tax deduction",
]


class BFSIOrchestrator:
    """
    Main orchestrator implementing exact PRD response logic.
    """

    def __init__(self, config_path: Optional[str] = None):
        base = Path(__file__).parent.parent
        config_path = Path(config_path) if config_path else base / "config" / "settings.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.config = cfg
        self.base_path = base

        # Initialize components
        from src.guardrails import Guardrails
        from src.dataset_similarity import DatasetSimilarityChecker
        from src.slm_inference import SLMInference
        from src.rag_retrieval import RAGRetriever

        self.guardrails = Guardrails(cfg.get("guardrails", {}))
        self.dataset = DatasetSimilarityChecker(cfg.get("similarity", {}), str(base))
        self.slm = SLMInference(cfg.get("slm", {}), str(base))
        self.rag = RAGRetriever(cfg.get("rag", {}), str(base))

    def _is_complex_query(self, query: str) -> bool:
        """Determine if query requires RAG (complex financial/policy)."""
        q = query.lower()
        return any(kw in q for kw in RAG_TRIGGER_KEYWORDS)

    def _generate_rag_response(self, query: str) -> str:
        """Retrieve context and generate RAG-grounded response."""
        context = self.rag.get_context(query)
        if not context:
            # No RAG context - fallback to SLM with caveat
            return self.slm.generate(
                query + "\n[Note: No policy document match. Respond cautiously; do not invent numbers.]"
            )
        # Use SLM with RAG context for grounded generation
        augmented = f"""The following is verified policy/knowledge. Use it to answer. Do NOT invent numbers.

{context}

---

User query: {query}

Provide a factual, policy-aligned response based on the above. If the answer is not in the context, say so and direct to official channels."""
        return self.slm.generate(augmented)

    def process(self, query: str) -> dict:
        """
        Process user query following exact priority order.
        Returns dict with: response, source (dataset|slm|rag), metadata.
        """
        metadata = {"tier": None, "similarity_score": None}

        # Guardrails (absolute enforcement)
        allowed, reason = self.guardrails.check(query)
        if not allowed:
            return {
                "response": reason,
                "source": "guardrail_reject",
                "metadata": metadata,
            }

        # --- Tier 1: Dataset Similarity Check ---
        stored_response, score = self.dataset.search(query)
        metadata["similarity_score"] = score
        if stored_response is not None:
            metadata["tier"] = "dataset"
            return {
                "response": stored_response,  # EXACT, no modification
                "source": "dataset",
                "metadata": metadata,
            }

        # --- Tier 2 vs Tier 3: SLM vs RAG ---
        if self._is_complex_query(query):
            # Tier 3: RAG for complex queries
            metadata["tier"] = "rag"
            response = self._generate_rag_response(query)
            return {
                "response": response,
                "source": "rag",
                "metadata": metadata,
            }
        else:
            # Tier 2: SLM for non-complex queries
            metadata["tier"] = "slm"
            response = self.slm.generate(query)
            return {
                "response": response,
                "source": "slm",
                "metadata": metadata,
            }
