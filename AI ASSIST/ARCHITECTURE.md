# BFSI Call Center AI Assistant – System Architecture

## Document Control
- **Version:** 1.0
- **Classification:** Technical Design
- **Compliance:** BFSI Regulatory Alignment

---

## 1. Executive Summary

The BFSI Call Center AI Assistant is a lightweight, compliant, and efficient system for handling common banking, financial services, and insurance queries. The system prioritizes curated dataset responses to ensure safety and regulatory compliance, with fallback to a local fine-tuned SLM and RAG for complex queries.

---

## 2. System Pipeline (Exact Order – Non-Negotiable)

```
User Query
    ↓
[1] Alpaca Dataset Similarity Check
    ↓ (strong match? → Return stored response exactly)
    ↓ (no match)
[2] Local Fine-Tuned Small Language Model (SLM)
    ↓ (appropriate for SLM? → Generate response)
    ↓ (complex query requiring knowledge?)
[3] RAG Layer (Retrieval-Augmented Generation)
    ↓
Final Response
```

**Critical:** The response priority MUST NOT be violated. Dataset match takes absolute precedence.

---

## 3. Response Logic (Strict Priority Order)

| Tier | Component | Condition | Action |
|------|-----------|-----------|--------|
| 1 | Dataset Match | Similarity score ≥ threshold | Return stored dataset response exactly, no modification |
| 2 | Fine-Tuned SLM | No dataset match; query within SLM scope | Generate response via local SLM |
| 3 | RAG Retrieval | Complex query (interest, EMI, penalties, policy rules) | Retrieve from knowledge base, generate grounded response |

---

## 4. Core Components

### 4.1 Dataset (Primary Response Layer)

- **Format:** Alpaca (Instruction, Input, Output)
- **Minimum Size:** 150+ BFSI conversation samples
- **Role:** Primary source of truth. Strong similarity match → return stored response directly.
- **Tone:** Professional, compliant, standardized

### 4.2 Small Language Model (Local SLM)

- **Type:** Lightweight instruction-based
- **Fine-tuning:** Alpaca BFSI dataset
- **Execution:** Local on modest hardware
- **Role:** Used ONLY when no strong dataset match exists

### 4.3 RAG Layer

- **Purpose:** Complex financial/policy queries
- **Use Cases:** Interest explanations, EMI breakdowns, penalties, policy rules
- **Storage:** Structured documents
- **Role:** Retrieve relevant information; generate factual, policy-aligned responses

---

## 5. Guardrails and Security (Absolute Enforcement)

- NO guessing of financial numbers
- NO generation of fake interest rates or policies
- NO exposure of sensitive customer data
- MUST reject out-of-domain or unsafe queries
- MUST maintain BFSI regulatory compliance at all times

---

## 6. Scalability and Maintainability

- Support scaling for higher call volumes
- Dataset, model weights, and RAG documents version controlled
- Financial policies and documents updatable without full retraining

---

## 7. File Structure

```
bfsi-ai-assistant/
├── config/
│   └── settings.yaml
├── data/
│   ├── alpaca_dataset.json          # 150+ BFSI samples
│   └── rag_knowledge/               # Structured knowledge documents
├── models/
│   └── slm_weights/                 # Fine-tuned SLM checkpoint
├── src/
│   ├── dataset_similarity.py        # Alpaca similarity check
│   ├── slm_inference.py             # Local SLM inference
│   ├── rag_retrieval.py             # RAG layer
│   ├── guardrails.py                # Safety and compliance
│   ├── orchestrator.py              # Response priority logic
│   └── demo.py                      # End-to-end demo
├── scripts/
│   └── finetune_slm.py              # SLM fine-tuning pipeline
├── requirements.txt
├── ARCHITECTURE.md                  # This document
└── TECHNICAL_DOCUMENTATION.md       # Detailed technical spec
```
