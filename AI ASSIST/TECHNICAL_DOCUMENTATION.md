# BFSI Call Center AI Assistant – Technical Documentation

## 1. System Architecture

### 1.1 Pipeline (Exact Order – Non-Negotiable)

```
User Query
    ↓
[1] Guardrails Check
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

The system processes every query in this strict sequence. Guardrails run first; then the dataset is checked; if no match, the query is classified as complex or not, and either RAG or SLM is invoked accordingly.

---

## 2. Response Logic (Strict Priority – MUST NOT Be Violated)

| Tier | Component | Condition | Action |
|------|-----------|-----------|--------|
| 1 | Dataset Match | Similarity score ≥ threshold | Return stored dataset output **exactly**, no modification |
| 2 | Fine-Tuned SLM | No dataset match; query not complex | Generate response via local SLM |
| 3 | RAG Retrieval | No dataset match; complex query | Retrieve from knowledge base; generate grounded response |

**Priority rule:** Tier 1 overrides all others. Tier 2 and Tier 3 are mutually exclusive and apply only when Tier 1 does not match.

---

## 3. Component Architecture

### 3.1 Dataset Layer (Tier 1 – Primary)

**Role:** Primary response source. When the user query is semantically close to a dataset sample, the system returns that sample’s stored response without change.

**Logic:**
- Each dataset sample has: `instruction`, `input`, `output` (Alpaca format).
- The searchable text is `instruction` + `input` (user intent).
- User query is embedded; cosine similarity is computed against all dataset embeddings.
- If the best similarity score ≥ threshold (e.g. 0.85), the corresponding `output` is returned exactly.
- No rewriting, paraphrasing, or post-processing of matched responses.

**Why:** Curated, compliant responses are prioritized over generative outputs.

---

### 3.2 Small Language Model (Tier 2)

**Role:** Handles queries with no dataset match that are not complex.

**Logic:**
- Invoked only when:
  1. Dataset similarity is below threshold, and
  2. The query is not classified as complex (see Tier 3).
- A local instruction-tuned SLM generates the response.
- System prompt instructs the model not to guess financial numbers, invent rates, or fabricate policy.

**Why:** Provides answers for simple, non-standard questions while keeping generation constrained.

---

### 3.3 RAG Layer (Tier 3)

**Role:** Handles complex financial or policy queries by retrieving from structured documents and generating grounded responses.

**Logic:**
- A query is treated as complex if it matches certain keywords (e.g. interest rate, EMI formula, penalty, foreclosure charge, policy, KYC, regulatory).
- For complex queries with no dataset match:
  1. Relevant chunks are retrieved from the knowledge base via embedding similarity.
  2. Retrieved chunks are given to the SLM as context.
  3. The SLM generates a response conditioned on this context.
- Responses are intended to be factual and aligned with policy, not invented.

**Why:** Complex topics require grounding in authoritative documents rather than free-form generation.

---

## 4. Decision Logic

```
                    ┌─────────────────┐
                    │   User Query    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Guardrails    │  ──reject──► Return rejection message
                    │     Check       │
                    └────────┬────────┘
                             │ pass
                             ▼
                    ┌─────────────────┐
                    │ Dataset         │
                    │ Similarity      │  ──score ≥ threshold──► Return stored output (Tier 1)
                    │ Check           │
                    └────────┬────────┘
                             │ no match
                             ▼
                    ┌─────────────────┐
                    │ Complex Query?  │
                    │ (keywords)      │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     ┌─────────────────┐           ┌─────────────────┐
     │ Yes             │           │ No              │
     │ → RAG (Tier 3)  │           │ → SLM (Tier 2)  │
     │ Retrieve +      │           │ Generate        │
     │ Generate        │           │                 │
     └─────────────────┘           └─────────────────┘
```

---

## 5. Guardrails (Logic)

Guardrails are enforced before any response logic:

- **Sensitive-data rejection:** Queries containing terms such as password, PIN, CVV, account number, SSN are rejected. A safe, generic rejection message is returned.
- **Length limit:** Queries exceeding the maximum length are rejected.
- **No fabrication:** Tier 1 returns fixed, curated text. Tiers 2 and 3 use prompts that forbid inventing financial numbers, rates, or policy details.
- **Output integrity:** Tier 1 output is not modified or reformatted after retrieval.

---

## 6. Tier Interaction Summary

| Scenario | Dataset Match | Complex Query | Path |
|----------|---------------|---------------|------|
| Strong match | Yes | N/A | Tier 1 → stored response |
| No match, complex | No | Yes | Tier 3 → RAG + SLM |
| No match, simple | No | No | Tier 2 → SLM only |

Tiers 2 and 3 never run when Tier 1 produces a match.