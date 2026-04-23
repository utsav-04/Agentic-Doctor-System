"""
src/graph/graph.py
===================
Builds and compiles the LangGraph StateGraph.

Graph structure (matches workflow diagram):

  START
    │
    ▼
  welcome_node
    │
    ▼
  intake_node
    │
    ▼
  criticality_node ──── conditional ────────────────────┐
    │ (not critical)                                     │ (critical)
    ▼                                                    ▼
  intent_node ─── conditional ───────────────────── doctor_node
    │ (medicine)       │ (lab)       │ (doctor)           │
    ▼                  ▼             └──────────────────► │
  medicine_node    lab_node                               │
    │                  │                                  │
    └──────────────────┴──────────────────────────────────┘
                       │
                       ▼
                  hitl_node ─── conditional ────┐
                       │ (approved)             │ (revise → loop back)
                       ▼                        │
                  response_node ◄───────────────┘
                       │
                       ▼
                      END

Run:
    from src.graph.graph import build_graph
    graph = build_graph()
    result = graph.invoke({"user_input": "I have chest pain", "messages": []})
"""

import sys
from pathlib import Path

from langgraph.graph import StateGraph, END

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graph.state import AgentState
from src.graph.nodes import (
    welcome_node,
    intake_node,
    criticality_node,
    intent_node,
    doctor_node,
    medicine_node,
    lab_node,
    hitl_node,
    response_node,
)
from src.graph.edges import (
    route_after_criticality,
    route_after_intent,
    route_after_hitl,
)


def build_graph():
    """
    Builds and compiles the full LangGraph StateGraph.
    Returns a compiled graph ready to invoke.
    """
    builder = StateGraph(AgentState)

    # ── Add all nodes ─────────────────────────────────────────────
    builder.add_node("welcome",     welcome_node)
    builder.add_node("intake",      intake_node)
    builder.add_node("criticality", criticality_node)
    builder.add_node("intent",      intent_node)
    builder.add_node("doctor",      doctor_node)
    builder.add_node("medicine",    medicine_node)
    builder.add_node("lab",         lab_node)
    builder.add_node("hitl",        hitl_node)
    builder.add_node("response",    response_node)

    # ── Entry point ───────────────────────────────────────────────
    builder.set_entry_point("welcome")

    # ── Fixed edges (no branching) ────────────────────────────────
    builder.add_edge("welcome",     "intake")
    builder.add_edge("intake",      "criticality")

    # ── Conditional edge 1: after criticality ─────────────────────
    builder.add_conditional_edges(
        "criticality",
        route_after_criticality,
        {
            "doctor": "doctor",
            "intent": "intent",
        },
    )

    # ── Conditional edge 2: after intent ─────────────────────────
    builder.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "medicine": "medicine",
            "lab":      "lab",
            "doctor":   "doctor",
            "response": "response",
        },
    )

    # ── All agent nodes converge into HITL ───────────────────────
    builder.add_edge("medicine", "hitl")
    builder.add_edge("lab",      "hitl")
    builder.add_edge("doctor",   "hitl")

    # ── Conditional edge 3: after HITL ───────────────────────────
    builder.add_conditional_edges(
        "hitl",
        route_after_hitl,
        {
            "response": "response",
            "medicine": "medicine",
            "lab":      "lab",
            "doctor":   "doctor",
        },
    )

    # ── Response → END ───────────────────────────────────────────
    builder.add_edge("response", END)

    return builder.compile()


# Singleton instance — import this in app.py
graph = build_graph()
