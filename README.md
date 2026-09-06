# Agentic Doctor System

An agentic medical assistant that takes a free-text symptom description (or a
button-selected tool) and routes it through a LangGraph pipeline to RAG-backed
tools for medicine suggestions, lab test recommendations, and doctor lookup —
with an optional integration to a separate order-management app for actually
placing medicine/lab orders and booking doctor appointments, gated by human
review (HITL).

This README covers the main chat application. The order-management app
(admin dashboard + MCP server) is a separate project — see
[Related project](#related-project-order-management-app) below.

---

## Table of contents

- [Architecture](#architecture)
- [Features](#features)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the app](#running-the-app)
- [Usage](#usage)
- [API reference](#api-reference)
- [Environment variables](#environment-variables)
- [Caching](#caching)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [Related project: order-management-app](#related-project-order-management-app)

---

## Architecture

Every user turn flows through one LangGraph invocation:

```
User message
     │
     ▼
intake_node          — extracts symptoms from free text
     │
     ▼
cache_check_node     — full-response cache lookup (Redis)
     │
     ├── HIT ──────────────────────────────► instant response
     │
     ▼ MISS
criticality_node     — Ollama call → critical / moderate / mild (cached)
     │
     ▼
planner_node         — Ollama call → ordered list of tools to run (cached)
     │                  e.g. ["medicine", "first_aid"]
     ▼
executor_node        — loops the plan, calling each tool
     │                  (medicine_finder, lab_test_suggester, doctor_finder,
     │                   first_aid_guide — each RAG-backed via ChromaDB,
     │                   each checked against Redis before hitting RAG)
     │
     │                  order_medicine / order_lab / book_doctor route out
     │                  via MCP to the separate order-management-app instead
     │
     ▼
synthesis_node       — merges tool outputs (template-based, no LLM call),
                        caches the full response, offers a lab/doctor
                        follow-up when relevant
     │
     ▼
Response to user
```

Two things can pause mid-graph and require a follow-up message to resume:
- **`doctor_need_location`** — the doctor tool needs a city/state.
- **`need_patient_name`** — an order/booking tool needs a patient name.

Both resume by re-invoking the graph with the missing field added to state
and the rest of `tool_outputs`/`plan` carried forward, so nothing already
computed gets redone.

## Features

- **Multi-tool planning** — a single query can trigger multiple tools
  (e.g. medicine + first aid) in one pass, not just one intent at a time.
- **RAG-backed recommendations** — medicine, lab test, and doctor data are
  retrieved via a hybrid (dense + BM25 → RRF → MMR) retriever over ChromaDB,
  not generated from the LLM's memory.
- **Three-layer Redis caching** — full-response, per-tool, and per-LLM-call
  caching, each with its own TTL, all degrading gracefully to live calls if
  Redis is unreachable.
- **Follow-up flow** — after a symptom-based answer, the assistant offers
  lab/doctor suggestions and only renders what's *new* in the next message,
  not everything already shown.
- **Chat UI** — a plain HTML/CSS/JS single-page app (no build step) with
  mode pills for Auto / Medicine / Lab / Doctor.
- **Order/booking integration (optional)** — explicit "order this medicine" /
  "book this appointment" requests are routed via MCP to a separate
  order-management app, gated by human review for medicine/lab.

## Project structure

```
Agentic-Doctor-System/
├── api/
│   ├── main.py              FastAPI backend — POST /session, POST /chat
│   └── session_manager.py   In-memory per-session conversation state
├── static/
│   ├── index.html           Chat UI markup
│   ├── style.css            Dark theme styling
│   └── script.js            Chat UI logic
├── src/
│   ├── graph/
│   │   ├── state.py         AgentState schema
│   │   ├── router.py        detect_criticality / detect_plan (LLM calls)
│   │   ├── nodes.py         All LangGraph nodes
│   │   └── graphs.py        build_graph() — wires nodes + conditional edges
│   ├── agents/
│   │   └── agents.py        CrewAI Agent definitions + shared LLM instance
│   ├── tools/
│   │   ├── medicine_tool.py
│   │   ├── lab_tool.py
│   │   ├── doctor_tool.py
│   │   └── first_aid_tool.py
│   ├── rag/
│   │   ├── config.py, document_builder.py, ingest.py,
│   │   ├── retriever.py, vector_store_manager.py
│   ├── cache/
│   │   └── redis_cache.py   Redis get/set with graceful fallback
│   ├── integrations/
│   │   └── order_client.py  MCP client — calls the order-management-app
│   └── schema/
│       └── schemas.py       Pydantic validation for RAG records/tool I/O
├── test.py                  CLI test harness (single-graph, single invoke)
└── requirements.txt
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally with the `llama3` model pulled
  (`ollama pull llama3`)
- Redis (e.g. via Docker: `docker run -d -p 6379:6379 redis`)
- A Google API key with Gemini embeddings access (used only for RAG
  embeddings, not for chat generation)
- Your RAG source JSON files (medicine/doctor/first-aid/lab-test data) —
  see `src/rag/config.py` → `DataPaths` for expected file locations

## Setup

```bash
git clone <your-repo-url>
cd Agentic-Doctor-System
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # macOS/Linux

pip install -r requirements.txt
pip install fastapi uvicorn redis fastmcp   # if not already in requirements.txt
```

Create a `.env` file at the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
HF_TOKEN=
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=agentic-doctor-system

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

ORDER_MCP_URL=http://127.0.0.1:9100/mcp
```

Ingest your RAG data (run once, or after updating source JSON):

```bash
python -m src.rag.ingest
```

## Running the app

Make sure Ollama and Redis are running, then:

```bash
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000**.

If you're using the order-management-app integration, also start its MCP
server (`python -m mcp_server.server`, port 9100) and admin dashboard
(`uvicorn admin.main:app --port 9000`) from that project's directory — see
its own README for details.

## Usage

- Type a symptom description in Auto mode and the planner decides which
  tool(s) to run.
- Click a mode pill (Medicine / Lab / Doctor) before sending to force that
  single tool, skipping planning entirely.
- After a symptom-based answer, reply `lab`, `doctor`, `both`, or `no` to
  the follow-up offer.
- Say something like *"order the paracetamol for me"* or *"book an
  appointment with Dr. X"* to trigger the order/booking flow (requires the
  order-management-app to be running).

## API reference

### `POST /session`
Creates a new chat session.

**Response**
```json
{ "session_id": "b6e2..." }
```

### `POST /chat`
Sends one message within a session.

**Request**
```json
{
  "session_id": "b6e2...",
  "message": "cold, cough and fever",
  "tool": null
}
```
`tool` is one of `"medicine"`, `"lab"`, `"doctor"`, `"first_aid"`, or `null`
for Auto mode.

**Response**
```json
{
  "session_id": "b6e2...",
  "reply": "...",
  "criticality": "mild",
  "plan": ["medicine", "first_aid"],
  "awaiting": null
}
```
`awaiting` is `"location"`, `"follow_up"`, `"patient_name"`, or `null`. When
non-null, the *next* message you send is interpreted as an answer to that
specific pending question — unless it clearly looks like an unrelated new
request, in which case it's treated as a fresh query instead (see
`_looks_like_location_reply` / `_looks_like_name_reply` /
`_follow_up_selection` in `api/main.py`).

## Environment variables

| Variable | Used by | Default |
|---|---|---|
| `GOOGLE_API_KEY` | RAG embeddings (Gemini) | — |
| `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` | LangSmith tracing | — |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_DB` | `src/cache/redis_cache.py` | `localhost` / `6379` / `0` |
| `ORDER_MCP_URL` | `src/integrations/order_client.py` | `http://127.0.0.1:9100/mcp` |

## Caching

Three layers, all in Redis, all fail-open (if Redis is down, everything
just runs live instead of erroring):

| Layer | Key basis | TTL |
|---|---|---|
| Full response | symptoms + age + location + forced_tool | 6h |
| Per-tool RAG result | tool-specific args (symptoms, state/city, etc.) | 7d |
| LLM classification (criticality/plan/triage) | symptoms / user_input | 24h |

Flush everything during development with:
```bash
docker exec -it <redis-container> redis-cli FLUSHDB
```

## Testing

```bash
python test.py                  # free-text mode, planner decides
python test.py --tool medicine  # forced-tool mode
```

`test.py` is a CLI harness that exercises the same `build_graph()` the API
uses — useful for debugging the pipeline without the web UI in the loop.

## Known limitations

- **Session store is in-memory** (`api/session_manager.py`) — conversation
  state is lost on server restart and isn't shared across multiple worker
  processes. A Redis-backed or LangGraph-checkpointer-backed store would
  fix both; not yet implemented.
- **`patient_name` doesn't persist across separate fresh queries** — each
  new (non-resumed) message starts a clean graph state, so ordering twice
  in one conversation asks for the name each time.
- **Item extraction for orders is free-text, not structured** — "order the
  paracetamol" passes that phrase through as the item description rather
  than matching it against a specific prior recommendation; the admin
  review step is the safety net for this.
- **Pause-detection heuristics are approximate** — whether an incoming
  message answers a pending question (location/patient-name/follow-up) or
  is a new unrelated request is decided by simple word-count/keyword rules
  in `api/main.py`, not an LLM classification. Edge cases can misfire.
- **No authentication** on either this app or the admin dashboard — both
  assume local/trusted use for now.

## Related project: order-management-app

A separate FastAPI + MySQL project providing:
- An MCP server (`place_medicine_order`, `place_lab_order`,
  `book_doctor_appointment` tools) that this project calls via
  `src/integrations/order_client.py`.
- An admin dashboard (three tabs: Medicine, Lab, Doctor Appointments) where
  a human confirms pending medicine/lab orders before they're considered
  placed. Doctor appointments are booked directly, with no gate.

See that project's own README for its setup and run instructions.
