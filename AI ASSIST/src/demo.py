"""
BFSI Call Center AI - End-to-End Demo
Demonstrates the complete pipeline with strict priority logic.
"""

import sys
from pathlib import Path

# Add project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    print("=" * 60)
    print("BFSI Call Center AI Assistant - End-to-End Demo")
    print("=" * 60)
    print("\nPipeline: User Query → Dataset Similarity → SLM (if no match) → RAG (if complex)\n")

    try:
        from src.orchestrator import BFSIOrchestrator
        orchestrator = BFSIOrchestrator()
    except Exception as e:
        print(f"Initialization error: {e}")
        print("\nEnsure dependencies are installed: pip install -r requirements.txt")
        print("Dataset must exist at data/alpaca_dataset.json")
        return 1

    # Demo queries covering all tiers
    demo_queries = [
        "How do I check my loan eligibility?",
        "What is the status of my loan application?",
        "How is EMI calculated?",
        "What are the penalties for late EMI payment?",
        "What is the interest rate formula for loans?",
    ]

    for i, query in enumerate(demo_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        try:
            result = orchestrator.process(query)
            source = result.get("source", "unknown")
            response = result.get("response", "No response.")
            tier = result.get("metadata", {}).get("tier", "-")
            score = result.get("metadata", {}).get("similarity_score")
            print(f"Source: {source} | Tier: {tier}" + (f" | Similarity: {score:.3f}" if score is not None else ""))
            print(f"Response: {response[:300]}{'...' if len(response) > 300 else ''}")
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Demo complete. You can also run interactively.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
