"""
api/session_manager.py
=======================
In-memory conversation session store, one entry per browser chat session.

Holds:
  graph_state  — the full dict returned by the last graph.invoke() call.
                 Passed back in as previous_state when resuming (location
                 reply, follow-up answer) so executor_node can skip tools
                 already in tool_outputs.
  waiting_for  — None | "location" | "follow_up" — tells /chat how to
                 interpret the NEXT message that comes in for this session.
  history      — plain chat log for the UI to redraw on reconnect.

This is fine for a single-process deployment. If you ever run multiple
backend workers/replicas, swap this for a Redis-backed store (you already
have Redis running for the tool/LLM caches) — the shape here is
deliberately simple so that's a drop-in replacement later, not a rewrite.
"""

import uuid
from typing import Any, Dict, Optional

_sessions: Dict[str, Dict[str, Any]] = {}


def _blank_session() -> Dict[str, Any]:
    return {"graph_state": {}, "waiting_for": None, "history": []}


def create_session() -> str:
    session_id = str(uuid.uuid4())
    _sessions[session_id] = _blank_session()
    return session_id


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    return _sessions.get(session_id)


def update_session(session_id: str, **fields) -> None:
    session = _sessions.setdefault(session_id, _blank_session())
    session.update(fields)


def append_history(session_id: str, role: str, content: str) -> None:
    session = _sessions.setdefault(session_id, _blank_session())
    session["history"].append({"role": role, "content": content})