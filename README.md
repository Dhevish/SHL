# SHL Conversational Assessment Recommender

A conversational AI agent that recommends SHL Individual Test Solutions through natural dialogue.

## Architecture

```
POST /chat  →  FastAPI  →  Anthropic Claude (claude-3-5-haiku)
                   ↕
            51-product SHL catalog (in-context retrieval)
```

**Stack:** FastAPI · Anthropic SDK · Pydantic v2 · Uvicorn  
**LLM:** Claude 3.5 Haiku (fast, 200k context — full catalog fits in one prompt)  
**Retrieval:** Full catalog injected into system prompt (no vector DB needed at this scale)

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Test it

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a mid-level Java developer with 4 years experience"}
    ]
  }'
```

## API Reference

### `GET /health`
Returns `{"status": "ok"}` with HTTP 200.

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```
- Stateless: pass the full conversation history on every call.
- Last message must be from `user`.

**Response:**
```json
{
  "reply": "Here are 4 assessments suited for a mid-level Java developer.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```
- `recommendations` is `[]` while clarifying or refusing.
- `test_type` is a single letter: A B C D E K M P S.
- `end_of_conversation` is `true` only when the task is complete.

## Deployment (Render — free tier)

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → New Web Service → Connect your repo.
3. Render auto-detects `render.yaml`.
4. Add environment variable: `ANTHROPIC_API_KEY` = your key.
5. Deploy. Your URL will be `https://shl-assessment-recommender.onrender.com`.

**Note:** Render free tier sleeps after 15 min inactivity. First `/health` call allows up to 2 min for wake-up (within the evaluator's spec).

## Running Tests

```bash
# Unit + behavior probes (requires running server)
pip install pytest
python -m pytest tests/ -v

# Quick smoke test
python tests/test_agent.py

# Integration recall tests
python -m pytest tests/ -v -m integration
```

## Agent Design Decisions

### Why in-context retrieval instead of vector search?
The full SHL catalog (51 products) fits in ~4,000 tokens — well within Claude's 200k context window. Vector search adds latency, infrastructure, and retrieval errors without benefit at this scale.

### Why Claude 3.5 Haiku?
- Fast enough to meet the 30s timeout
- Follows structured JSON output instructions reliably
- Cheap enough for high-volume evaluation

### Clarification strategy
The agent asks ONE focused question per turn (not multiple). It asks about:
1. Role/job title (if missing)
2. Seniority level (if missing)
3. Key requirements (technical skills, leadership, customer-facing, etc.)

Once it has role + level, it recommends. Additional context improves recall.

### Hallucination prevention
- All recommendations are post-validated against the catalog URL set.
- Hallucinated names matched by lowercase name lookup; if not found, dropped entirely.
- System prompt explicitly instructs: "only recommend from the catalog below."

### Scope enforcement
The system prompt lists explicit refusal triggers: general hiring advice, legal questions, competitor products, prompt injection. The response validator ensures `recommendations=[]` for refusals.
