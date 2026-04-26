
"""
src/graph/graphs.py
====================
Two separate graphs matching the test.py flow exactly.

graph_1 (intake): welcome → intake → criticality → END
graph_2 (action): intent  → doctor/medicine/lab  → END
graph_3 (doctor): doctor  → END   (critical path, no intent step)

Why split:
  LangGraph runs all connected nodes in one invoke().
  We need user input BETWEEN criticality and intent.
  Two graphs lets test.py collect input in between.
"""

from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.graph.nodes import (
    welcome_node,
    intake_node,
    criticality_node,
    intent_node,
    doctor_node,
    medicine_node,
    lab_node,
)


def build_intake_graph():
    """welcome → intake → criticality → END"""
    builder = StateGraph(AgentState)

    builder.add_node("welcome",     welcome_node)
    builder.add_node("intake",      intake_node)
    builder.add_node("criticality", criticality_node)

    builder.set_entry_point("welcome")
    builder.add_edge("welcome",     "intake")
    builder.add_edge("intake",      "criticality")
    builder.add_edge("criticality", END)

    return builder.compile()


def _route_after_intent(state: AgentState) -> str:
    intent = state.get("intent", "medicine")
    if intent == "doctor":
        return "doctor"
    elif intent == "lab":
        return "lab"
    else:
        return "medicine"


def build_action_graph():
    """intent → doctor/medicine/lab → END"""
    builder = StateGraph(AgentState)

    builder.add_node("intent",   intent_node)
    builder.add_node("doctor",   doctor_node)
    builder.add_node("medicine", medicine_node)
    builder.add_node("lab",      lab_node)

    builder.set_entry_point("intent")

    builder.add_conditional_edges(
        "intent",
        _route_after_intent,
        {
            "doctor":   "doctor",
            "medicine": "medicine",
            "lab":      "lab",
        },
    )

    builder.add_edge("doctor",   END)
    builder.add_edge("medicine", END)
    builder.add_edge("lab",      END)

    return builder.compile()


def build_doctor_graph():
    """doctor → END  (critical path — skips intent)"""
    builder = StateGraph(AgentState)
    builder.add_node("doctor", doctor_node)
    builder.set_entry_point("doctor")
    builder.add_edge("doctor", END)
    return builder.compile()

# graph = build_doctor_graph()
# print(graph.get_graph().draw_ascii())