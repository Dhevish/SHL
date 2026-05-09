# Approach Document — SHL Conversational Assessment Recommender

## Problem Framing

The core challenge is bridging vague hiring intent ("I need a Java developer") and a structured 51-product catalog. The agent must clarify, recommend, refine, compare, and refuse — all within 8 turns and 30 seconds per call.

## Design Choices

### 1. Retrieval: In-Context over Vector Search

The full SHL Individual Test Solutions catalog (51 products) fits in ~4,000 tokens. Rather than a vector database (Chroma, FAISS, pgvector), I inject the complete catalog directly into the system prompt. This eliminates retrieval latency, embedding drift, and chunk fragmentation — all sources of recall loss at this catalog scale. If the catalog grew to thousands of items, I would switch to BM25 + semantic re-ranking.

### 2. LLM: Claude 3.5 Haiku

Selected for:
- **Speed**: p50 latency ~2–4s, well within 30s timeout
- **JSON instruction-following**: reliably produces structured output without fine-tuning
- **Context window**: 200k tokens handles the full catalog + conversation history with headroom

### 3. Structured Output via JSON-only Prompt

Rather than function-calling/tool-use (which adds round-trips), I instruct the model to output only a JSON object with exactly three keys: `reply`, `recommendations`, `end_of_conversation`. A post-processing layer strips any stray markdown fences and validates the schema. This approach is simpler, faster, and equally reliable for this use case.

### 4. Hallucination Guard

After every LLM call, recommendations are cross-validated against a set of known catalog URLs. Any recommendation whose URL is not in the catalog is either corrected by name lookup or dropped entirely. This guarantees the "items from catalog only" hard eval.

### 5. Clarification Strategy

The agent asks **one focused question per turn** — never a list of questions. Priority: (1) role/title, (2) seniority, (3) technical requirements. Two parameters are usually enough to commit to a shortlist. This keeps conversations short, honoring the 8-turn cap while still gathering signal.

### 6. Scope Enforcement

The system prompt lists explicit refusal categories: general hiring advice, legal questions, competitor comparisons, prompt injection. An LLM trained on RLHF generalises these well. The response validator additionally enforces `recommendations=[]` for non-SHL topics.

## Evaluation Approach

**Hard evals**: Tested with `pytest` unit tests — schema compliance on every response, catalog-only URLs, turn cap, empty recommendations during clarification.

**Behavior probes** (11 tests):
- Vague query → no recommendation on turn 1 ✓
- Off-topic → refusal with empty recommendations ✓
- Prompt injection → refusal ✓
- Legal question → refusal ✓
- Comparison query → empty recommendations ✓
- Constraint refinement → updated shortlist including requested type ✓

**Recall@10**: Evaluated against 3 constructed traces with labeled expected assessments (Java dev, Sales manager, Graduate screening). Mean Recall@10 ≈ 0.55 in local testing.

## What Didn't Work

- **Asking multiple clarifying questions at once**: The model would sometimes produce 3-question lists; I fixed this with explicit "ask ONE focused clarifying question" in the prompt.
- **Vector search at small scale**: Prototype with FAISS showed lower recall than full-context injection due to chunking — dropped it.
- **GPT-4o-mini**: Tested as alternative; less reliable JSON output under adversarial inputs compared to Claude Haiku.

## AI Tools Used

- **Claude** (claude.ai) for iterating on the system prompt and identifying edge cases
- **Agentic coding** (Claude Code) for generating test scaffolding and the scraper skeleton
- All design decisions, prompt engineering, and architecture choices are my own; code reflects genuine understanding

## Stack Summary

| Component | Choice | Reason |
|-----------|--------|--------|
| API framework | FastAPI + Pydantic v2 | Fast, typed, auto-docs |
| LLM | Claude 3.5 Haiku | Speed + JSON reliability |
| Retrieval | In-context (full catalog) | Scale doesn't warrant vector DB |
| Deployment | Render (free tier) | Simple, free, meets wake-up spec |
| Testing | pytest + requests | Straightforward, no extra deps |
