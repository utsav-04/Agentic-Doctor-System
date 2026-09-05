"""
api/main.py
============
FastAPI backend for the Agentic Doctor System web chat.

Endpoints:
  GET  /            -> serves the chat UI (static/index.html)
  POST /session     -> creates a new chat session, returns session_id
  POST /chat        -> sends one message within a session, returns the reply

Run from your project root:
    uvicorn api.main:app --reload --port 8000
Then open http://localhost:8000
"""

import re
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.graphs import build_graph
from api.session_manager import append_history, create_session, get_session, update_session
from src.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Agentic Doctor System")

# Graph is built once at startup and reused for every request — it's just
# a compiled node/edge definition, no per-conversation state lives on it.
graph = build_graph()

STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

AGE_PATTERN = re.compile(r"\b(\d{1,3})\s*(?:years?|yrs?|y/?o)\b", re.IGNORECASE)
LOCATION_SPLIT = re.compile(r"\s*,\s*")


class ChatRequest(BaseModel):
    session_id: str
    message: str
    tool: Optional[str] = None  # "medicine" | "lab" | "doctor" | "first_aid" | None (= Auto)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    criticality: Optional[str] = None
    plan: Optional[List[str]] = None
    awaiting: Optional[str] = None  # "location" | "follow_up" | None


def _extract_age(text: str) -> str:
    """Best-effort — picks up '25 years old', '8 yrs', etc. from free text.
    If nothing matches, medicine_finder defaults to adult dosage (age=-1)."""
    match = AGE_PATTERN.search(text)
    return match.group(1) if match else ""


@app.get("/")
def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/session")
def new_session():
    session_id = create_session()
    logger.info("New session created: %s", session_id)
    return {"session_id": session_id}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id — call POST /session first.")

    append_history(req.session_id, "user", req.message)
    waiting_for = session.get("waiting_for")
    prev_state = session.get("graph_state", {})

    # ── Resuming: doctor tool paused for location ───────────────────
    if waiting_for == "location":
        parts = LOCATION_SPLIT.split(req.message.strip())
        if len(parts) >= 2:
            user_city, user_state = parts[0], parts[1]
        else:
            user_city, user_state = "", parts[0]

        result = graph.invoke({
            **prev_state,
            "user_state": user_state.title(),
            "user_city": user_city.title(),
        })

    # ── Resuming: answer to "want lab/doctor too?" ──────────────────
    elif waiting_for == "follow_up":
        offered = prev_state.get("follow_up_offered", [])
        answer = req.message.strip().lower()

        if "both" in answer or "yes" in answer:
            selected = offered
        else:
            selected = [t for t in offered if t in answer]

        if not selected:
            reply = "No problem — let me know if you need anything else."
            update_session(req.session_id, waiting_for=None)
            append_history(req.session_id, "assistant", reply)
            return ChatResponse(session_id=req.session_id, reply=reply, awaiting=None)

        extended_plan = list(dict.fromkeys(prev_state.get("plan", []) + selected))
        result = graph.invoke({**prev_state, "plan": extended_plan})

    # ── Fresh query — new symptoms/question, nothing pending ────────
    else:
        result = graph.invoke({
            "user_input": req.message,
            "user_age": _extract_age(req.message),
            "forced_tool": req.tool,
        })

    # ── Doctor tool needs location — pause and ask ───────────────────
    if result.get("stage") == "doctor_need_location":
        reply = "To find a nearby doctor, could you share your city and state? (e.g. 'Mumbai, Maharashtra')"
        update_session(req.session_id, graph_state=result, waiting_for="location")
        append_history(req.session_id, "assistant", reply)
        return ChatResponse(session_id=req.session_id, reply=reply, awaiting="location")

    # ── Normal completion (possibly with a follow-up offer attached) ─
    final_response = result.get("final_response", "Sorry, I couldn't process that.")
    awaiting = "follow_up" if result.get("follow_up_offered") else None

    update_session(req.session_id, graph_state=result, waiting_for=awaiting)
    append_history(req.session_id, "assistant", final_response)

    return ChatResponse(
        session_id=req.session_id,
        reply=final_response,
        criticality=result.get("criticality"),
        plan=result.get("plan"),
        awaiting=awaiting,
    )