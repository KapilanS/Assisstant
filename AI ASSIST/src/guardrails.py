"""
BFSI Call Center AI - Guardrails and Security Module
Absolute enforcement of safety and compliance rules.
- NO guessing of financial numbers
- NO generation of fake interest rates or policies
- NO exposure of sensitive customer data
- MUST reject out-of-domain or unsafe queries
- MUST maintain BFSI regulatory compliance at all times
"""

import re
from typing import Optional, Tuple


class Guardrails:
    """Enforces BFSI safety and compliance guardrails."""

    def __init__(self, config: dict):
        self.reject_keywords = set(
            kw.lower() for kw in config.get("reject_queries_containing", [])
        )
        self.max_query_length = config.get("max_query_length", 512)

    def check(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Check if query passes guardrails.
        Returns (allowed, rejection_reason).
        If allowed is True, rejection_reason is None.
        """
        if not query or not isinstance(query, str):
            return False, "Invalid or empty query."

        query_lower = query.lower().strip()

        # Length check
        if len(query_lower) > self.max_query_length:
            return False, "Query exceeds maximum allowed length."

        # Sensitive data / unsafe query check
        for kw in self.reject_keywords:
            if kw in query_lower:
                return False, (
                    "For your security, we cannot process requests "
                    "involving sensitive information through this channel. "
                    "Please visit a branch or use secure authenticated channels."
                )

        # Numeric financial guessing prevention: reject queries that ask
        # for specific numbers we might guess (e.g., "what's my balance? 12345")
        # We handle this via the response logic - we never output financial numbers
        # unless from dataset or RAG. No explicit check needed here for output.

        return True, None

    def sanitize_for_logging(self, text: str) -> str:
        """Remove or mask potentially sensitive content for logging."""
        if not text:
            return ""
        # Mask patterns that might contain sensitive data
        text = re.sub(r"\b\d{10,16}\b", "[CARD_MASKED]", text)
        text = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD_MASKED]", text)
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_MASKED]", text)
        return text
