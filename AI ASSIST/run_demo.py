#!/usr/bin/env python3
"""
BFSI Call Center AI Assistant - Run Demo
Run from project root: python run_demo.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from src.demo import main as demo_main
    return demo_main()


if __name__ == "__main__":
    sys.exit(main())
