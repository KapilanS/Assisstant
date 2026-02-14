# BFSI Call Center AI Assistant

A lightweight, compliant, and efficient AI assistant for Banking, Financial Services, and Insurance (BFSI) call center queries.

## Features

- **Dataset-first responses** (150+ Alpaca BFSI samples): Strong matches return stored responses exactly
- **Local Small Language Model (SLM)** for non-matching queries
- **RAG Layer** for complex financial/policy queries (interest, EMI, penalties)
- **Strict guardrails**: No guessing of financial numbers; rejects sensitive queries
- **BFSI regulatory compliance** by design

## Pipeline

```
User Query → Guardrails → Dataset Similarity → SLM (if no match) → RAG (if complex)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset (if not present)
python scripts/generate_dataset.py

# 3. Run demo
python run_demo.py

# 4. Interactive mode
python run_interactive.py
```

## Deliverables

| Deliverable | Location |
|-------------|----------|
| 150+ Alpaca BFSI dataset | `data/alpaca_dataset.json` |
| Fine-tuned SLM weights | `models/slm_weights/` (run `scripts/finetune_slm.py`) |
| Structured RAG knowledge | `data/rag_knowledge/*.md` |
| End-to-end demo | `run_demo.py`, `run_interactive.py` |
| Technical documentation | `TECHNICAL_DOCUMENTATION.md` |
| Architecture | `ARCHITECTURE.md` |

## Configuration

Edit `config/settings.yaml` for thresholds, paths, and model settings.

## Guardrails

- Rejects queries containing password, PIN, CVV, account number, SSN
- Never guesses financial numbers or invents policies
- Maintains BFSI regulatory compliance

## License

Internal use. BFSI compliance required.
