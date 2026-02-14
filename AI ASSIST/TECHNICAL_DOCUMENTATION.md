# BFSI Call Center AI Assistant – Technical Documentation

## 1. Overview

This document describes the architecture, response logic, implementation, and compliance guardrails of the BFSI Call Center AI Assistant. The system is designed for production-grade use in banking, financial services, and insurance (BFSI) call center operations.

---

## 2. System Architecture

### 2.1 Pipeline (Exact Order – Non-Negotiable)

```
User Query
    ↓
[1] Guardrails Check (reject unsafe/sensitive queries)
    ↓ (passed)
[2] Alpaca Dataset Similarity Check (Tier 1)
    ↓ (strong match? → Return stored response exactly)
    ↓ (no match)
[3] Complex Query Check
    ↓ (complex? → Tier 3: RAG)
    ↓ (not complex? → Tier 2: SLM)
[4] Response Generation
    ↓
Final Response
```

### 2.2 Response Priority (Strict – MUST NOT Be Violated)

| Tier | Component | Condition | Action |
|------|-----------|-----------|--------|
| 1 | Dataset Match | Similarity score ≥ threshold (default 0.85) | Return stored dataset output **exactly**, no modification |
| 2 | Fine-Tuned SLM | No dataset match; query not complex | Generate response via local SLM |
| 3 | RAG Retrieval | No dataset match; complex query (interest, EMI, penalties, policy) | Retrieve from knowledge base, generate grounded response |

---

## 3. Core Components

### 3.1 Dataset (Primary Response Layer)

- **Location:** `data/alpaca_dataset.json`
- **Format:** Alpaca (instruction, input, output)
- **Size:** 150+ BFSI conversation samples
- **Coverage:** Loan eligibility, application status, EMI, interest rates, payments, account support, insurance, credit cards
- **Similarity:** Embedding-based (sentence-transformers). Query is compared against concatenated instruction+input text.
- **Output:** When similarity ≥ threshold, the stored `output` is returned **exactly** without any modification.

**Similarity Logic:**
- Embedding model: `all-MiniLM-L6-v2` (lightweight, runs locally)
- Cosine similarity between query embedding and dataset embeddings
- Top-K candidates considered; best score must exceed threshold
- No paraphrasing or post-processing of matched output

### 3.2 Small Language Model (SLM)

- **Model:** TinyLlama-1.1B-Chat (or fine-tuned variant)
- **Execution:** Local (CPU or GPU)
- **Fine-tuning:** Alpaca BFSI dataset via `scripts/finetune_slm.py`
- **Role:** Used **only** when:
  1. No strong dataset match, and
  2. Query is not a complex financial/policy query

**Guardrails in SLM Prompt:**
- Instructed not to guess financial numbers
- Instructed not to invent interest rates or policies
- Instructed to direct to official channels when unsure

### 3.3 RAG Layer

- **Knowledge Base:** `data/rag_knowledge/*.md`
- **Documents:** interest_rates_policy.md, emi_breakdown_policy.md, penalties_policy.md, policy_rules.md
- **Chunking:** By section (##, ###)
- **Retrieval:** Embedding similarity; top chunks above threshold
- **Role:** Used **only** for complex queries (interest, EMI breakdowns, penalties, policy rules)
- **Generation:** RAG context is provided to SLM for grounded, factual responses

**RAG Trigger Keywords:** interest rate, EMI formula, penalty, policy, foreclosure charge, prepayment charge, KYC, regulatory, etc.

---

## 4. Guardrails and Security (Absolute Enforcement)

### 4.1 Programmatic Enforcement

- **Sensitive Data Rejection:** Queries containing password, PIN, CVV, account number, SSN, etc., are rejected.
- **Query Length:** Maximum 512 characters.
- **No Financial Guessing:** Response logic and prompts instruct the system never to invent numbers, rates, or policies.
- **Output Integrity:** Tier 1 responses are returned exactly; no post-processing that could alter factual content.

### 4.2 Implementation

- `src/guardrails.py`: `Guardrails.check()` returns (allowed, rejection_reason).
- Orchestrator invokes guardrails before any other processing.
- Sanitization for logging: sensitive patterns masked before logging.

---

## 5. Scalability and Maintainability

### 5.1 Scaling for Call Volumes

- Dataset similarity: Embedding model loads once; inference is fast.
- SLM: Runs locally; batch inference possible for queued queries.
- RAG: Vector similarity search; can be replaced with FAISS or Chroma for larger knowledge bases.
- Stateless design: No session state; horizontally scalable.

### 5.2 Version Control

- `data/alpaca_dataset.json`: Version controlled; changes tracked.
- `models/slm_weights/`: Fine-tuned model checkpoint; versioned separately.
- `data/rag_knowledge/`: Structured documents; easily updated without retraining.

### 5.3 Policy Updates

- Financial policies and documents in `data/rag_knowledge/` can be updated without retraining the SLM.
- Dataset can be augmented with new samples without changing code.
- Configuration in `config/settings.yaml` (thresholds, paths, models).

---

## 6. File Structure

```
bfsi-ai-assistant/
├── config/
│   └── settings.yaml           # Configuration
├── data/
│   ├── alpaca_dataset.json     # 150+ BFSI Alpaca samples
│   └── rag_knowledge/          # Structured policy documents
│       ├── interest_rates_policy.md
│       ├── emi_breakdown_policy.md
│       ├── penalties_policy.md
│       └── policy_rules.md
├── models/
│   └── slm_weights/            # Fine-tuned SLM checkpoint (after finetune)
├── src/
│   ├── __init__.py
│   ├── dataset_similarity.py   # Tier 1: Alpaca similarity
│   ├── slm_inference.py        # Tier 2: Local SLM
│   ├── rag_retrieval.py        # Tier 3: RAG
│   ├── guardrails.py           # Safety and compliance
│   ├── orchestrator.py         # Response priority logic
│   └── demo.py                 # End-to-end demo
├── scripts/
│   ├── generate_dataset.py     # Generate Alpaca BFSI dataset
│   └── finetune_slm.py         # Fine-tune SLM on BFSI dataset
├── requirements.txt
├── ARCHITECTURE.md
└── TECHNICAL_DOCUMENTATION.md  # This document
```

---

## 7. Usage

### 7.1 Installation

```bash
pip install -r requirements.txt
```

### 7.2 Generate Dataset (if not present)

```bash
python scripts/generate_dataset.py
```

### 7.3 Fine-Tune SLM (optional; base model used if not run)

```bash
python scripts/finetune_slm.py
```

### 7.4 Run Demo

```bash
python src/demo.py
```

### 7.5 Programmatic Usage

```python
from src.orchestrator import BFSIOrchestrator

orch = BFSIOrchestrator()
result = orch.process("How do I check my loan eligibility?")
print(result["response"])
print(result["source"])  # dataset, slm, or rag
```

---

## 8. Configuration Reference

| Key | Description | Default |
|-----|-------------|---------|
| similarity.threshold | Strong match threshold | 0.85 |
| similarity.embedding_model | Embedding model | all-MiniLM-L6-v2 |
| slm.model_name | Base SLM | TinyLlama-1.1B-Chat-v1.0 |
| slm.weights_path | Fine-tuned weights | models/slm_weights |
| rag.similarity_threshold | RAG retrieval threshold | 0.7 |
| guardrails.reject_queries_containing | Rejected keywords | password, pin, cvv, etc. |

---

## 9. Compliance Summary

- **BFSI Regulatory Alignment:** Guardrails and response logic designed for BFSI compliance.
- **No Fabrication:** Dataset and RAG provide factual, policy-aligned content; SLM instructed not to invent.
- **Auditability:** Source (dataset/slm/rag) tracked per response; version-controlled components.
- **Maintainability:** Policies updatable without full retraining; dataset extendable.

---

*End of Technical Documentation*
