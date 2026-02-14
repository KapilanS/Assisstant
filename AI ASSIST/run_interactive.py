#!/usr/bin/env python3
"""
BFSI Call Center AI Assistant - Interactive Demo
Type queries; Ctrl+C to exit.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("BFSI Call Center AI Assistant - Interactive Mode")
    print("Type your query (or 'quit' to exit):\n")
    try:
        from src.orchestrator import BFSIOrchestrator
        orch = BFSIOrchestrator()
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure: pip install -r requirements.txt")
        return 1
    while True:
        try:
            q = input("You: ").strip()
            if not q or q.lower() in ("quit", "exit", "q"):
                break
            r = orch.process(q)
            print(f"[{r['source']}] {r['response'][:500]}")
            if len(r["response"]) > 500:
                print("...")
            print()
        except KeyboardInterrupt:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
