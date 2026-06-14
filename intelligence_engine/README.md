# Automated Enterprise Market Intelligence & Fact-Checking Engine

A production-grade, fault-tolerant multi-agent system built on **LangGraph** that:

1. Concurrently queries an internal **Qdrant** vector database and the live web via **Tavily**
2. Passes aggregated context to a **Generator** agent (OpenAI `gpt-4.1-mini`) that outputs a rigid **Pydantic**-validated JSON report
3. Routes it to an aggressive **Critic** agent that cross-checks every claim against ground-truth context
4. Loops back to the generator with correction feedback if the report fails, with a **circuit breaker** to prevent infinite token-drain spirals
5. Evaluates quality automatically via **DeepEval** in CI/CD

---

## Architecture

```
[START]
   │
   ▼
┌─────────────┐
│ ingest_data │   ← asyncio.gather(Qdrant, Tavily) — concurrent retrieval
└─────────────┘
   │
   ▼
┌───────────────────┐
│  generate_report  │ ◄─────────────────────────────────┐
└───────────────────┘                                   │
   │                                                    │
   ▼                                              (correction loop)
┌────────────┐                                          │
│   critic   │                                          │
└────────────┘                                          │
   │                                                    │
   ▼                                                    │
route_after_critic ─── loop_counter >= 3 ──────► [END] (circuit breaker)
   │                                                    │
   ├── error_trace set ───────────────────────────────── ┘
   │
   ├── critic_feedback != 'PASSED' ──────────────────── ┘
   │
   └── PASSED ──────────────────────────────────► [END]
```

---

## Project Structure

```
intelligence_engine/
├── config.py                 # pydantic-settings config + ChatOpenAI init
├── schema.py                 # MarketIntelligenceReport + AgentState TypedDict
├── tools/
│   ├── __init__.py
│   ├── vector_search.py      # async_vector_query() → AsyncQdrantClient
│   └── web_search.py         # async_web_query() → Tavily SDK
├── graph.py                  # LangGraph StateGraph, all nodes, circuit breaker
├── tests/
│   ├── test_golden_dataset.json   # 3 curated evaluation scenarios
│   └── test_eval.py              # pytest + DeepEval metrics
├── main.py                   # CLI entry point with step-level streaming logs
├── requirements.txt
└── .env.example
```

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd intelligence_engine
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your real API keys
```

Required keys:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Powers Generator, Critic, and DeepEval judge |
| `TAVILY_API_KEY` | Live web search retrieval |

### 3. (Optional) Start a local Qdrant instance

```bash
docker run -p 6333:6333 qdrant/qdrant
```

If Qdrant is unavailable the system degrades gracefully to web-only context.

### 4. Run the pipeline

```bash
python -m intelligence_engine.main
```

You'll see step-by-step terminal output:

```
[▶ NODE]  INGEST_DATA
  Retrieved 8 context chunks total.
[✓ NODE]  INGEST_DATA completed in 1.23s

[▶ NODE]  GENERATE_REPORT
  ✓ Generated report for: NVIDIA Corporation
[✓ NODE]  GENERATE_REPORT completed in 4.51s

[▶ NODE]  CRITIC
  ✓ Fact-check PASSED — all claims grounded.
[✓ NODE]  CRITIC completed in 2.18s

Total pipeline elapsed time: 8.12s
Total critic loops completed: 1
```

### 5. Run the evaluation suite

```bash
pytest tests/test_eval.py -v
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | OpenAI API key (generator, critic, DeepEval judge) |
| `TAVILY_API_KEY` | ✅ | — | Tavily web-search API key |
| `QDRANT_URL` | ❌ | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_API_KEY` | ❌ | — | Qdrant Cloud key (omit for local) |
| `QDRANT_COLLECTION` | ❌ | `enterprise_intel` | Vector collection name |
| `LLM_MODEL` | ❌ | `gpt-4.1-mini` | OpenAI model identifier |
| `LLM_TEMPERATURE` | ❌ | `0` | Sampling temperature |
| `MAX_SEARCH_RESULTS` | ❌ | `5` | Tavily snippets per query |
| `MAX_VECTOR_RESULTS` | ❌ | `5` | Qdrant hits per query |
| `CIRCUIT_BREAKER_MAX` | ❌ | `3` | Max critic→generator loops |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `asyncio.gather` for retrieval | Qdrant + Tavily fire in parallel — latency = max(t_qdrant, t_tavily) not sum |
| Pydantic strict validation | Generator must produce schema-valid JSON or `error_trace` triggers auto-correction |
| Circuit breaker at N loops | Prevents runaway token consumption on pathological queries |
| Critic as separate LLM call | Separation of generation and verification reduces self-confirmation bias |
| Graceful degradation | Both retrieval tools return empty lists on failure — system keeps running |
| `pydantic-settings` config | 12-factor compliant; zero hard-coded secrets; `.env` for local dev |
| Single `OPENAI_API_KEY` | One key powers the pipeline AND DeepEval's judge — no extra credentials needed |

---

## Evaluation Metrics

| Metric | Threshold | What it checks |
|--------|-----------|----------------|
| `FaithfulnessMetric` | 0.50 (dev), 0.85 (prod) | Every claim grounded in retrieved context |
| `AnswerRelevancyMetric` | 0.70 (dev), 0.90 (prod) | Report addresses the research query intent |
| Schema validity | 100% pass | All required Pydantic fields populated |

---

## Interview Talking Points

**"How do you handle LLM hallucinations in production?"**
> The Critic agent cross-references every single claim in the generated report against the retrieved ground-truth snippets using a dedicated LLM call. If any claim is ungrounded, it returns specific corrections and the state graph routes back to the Generator with the feedback. A circuit breaker prevents infinite loops.

**"How do you optimise a multi-step workflow for latency?"**
> The retrieval step fires Qdrant and Tavily concurrently using `asyncio.gather`, so total retrieval latency equals `max(t_qdrant, t_tavily)` rather than their sum. The generator and critic are separate calls to maximise determinism but the retrieval phase doesn't compound those latencies.

**"How do you ensure output schema compliance?"**
> Every generator output is validated by `MarketIntelligenceReport.model_validate_json()`. On failure, the full Python traceback is stored in `error_trace` in the agent state and injected into the next generator prompt as explicit correction context. The generator never silently produces malformed output.
