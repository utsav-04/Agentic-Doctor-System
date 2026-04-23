"""
src/graph/edges.py
===================
All conditional edge functions for the LangGraph state machine.
Each function receives the state and returns a string key
that maps to the next node name in the graph.

Routing logic matches the workflow diagram exactly:

  criticality_node  →  route_after_criticality
      critical  → doctor_node
      not critical → intent_node

  intent_node  →  route_after_intent
      medicine  → medicine_node
      lab       → lab_node
      both      → medicine_node  (then lab runs via sequential edge)
      doctor    → doctor_node
      first_aid → doctor_node    (first aid given in criticality; doctor still recommended)
      general   → response_node  (generic response, no RAG needed)

  hitl_node  →  route_after_hitl
      approved  → response_node
      revise    → medicine_node / lab_node / doctor_node  (loop back)
"""

from src.graph.state import AgentState


def route_after_criticality(state: AgentState) -> str:
    """
    First conditional edge — triggered after criticality_node.
    Critical → doctor (emergency path).
    Not critical → intent (ask what user needs).
    """
    criticality = state.get("criticality", "mild")

    if criticality == "critical":
        return "doctor"

    return "intent"


def route_after_intent(state: AgentState) -> str:
    """
    Second conditional edge — triggered after intent_node.
    Routes to the correct RAG agent based on what the user wants.
    """
    intent = state.get("intent", "general")

    if intent == "medicine":
        return "medicine"

    if intent == "lab":
        return "lab"

    if intent == "both":
        return "medicine"

    if intent == "doctor":
        return "doctor"

    if intent == "first_aid":
        return "medicine"

    return "response"


def route_after_hitl(state: AgentState) -> str:
    """
    Third conditional edge — triggered after hitl_node.
    Approved → response_node.
    Revise → loop back to the appropriate agent node.
    """
    if state.get("hitl_approved"):
        return "response"

    stage = state.get("stage", "")

    if "medicine" in stage:
        return "medicine"
    if "lab" in stage:
        return "lab"
    if "doctor" in stage:
        return "doctor"

    return "medicine"
