"""
src/graph/graphs.py
====================
ONE graph, ONE invoke() per user turn. Replaces the old 3-graph split
(build_intake_graph / build_action_graph / build_doctor_graph) that required
input() between graphs.

    welcome -> intake -> [forced_tool?]
                            |-- yes --> planner (bypass) --------------------\
                            |-- no  --> criticality -> [plan already set?]    \
                                            |-- yes (critical) --> executor ---+--> [need location?]
                                            |-- no             --> planner ---/         |-- yes --> END (pause)
                                                                                          |-- no  --> synthesis --> END

Two legitimate reasons a single invoke() can end without a final_response:
  1. forced_tool='doctor' (or plan includes 'doctor') but no user_state yet
     -> stage == 'doctor_need_location'. Caller re-invokes with user_state/
        user_city set and the same plan/tool_outputs carried forward.
"""

from langgraph.graph import StateGraph, END

from src.graph.state import AgentState
from src.graph.nodes import (
    welcome_node,
    intake_node,
    criticality_node,
    planner_node,
    executor_node,
    synthesis_node,
)


def _route_after_intake(state: AgentState) -> str:
    """Skip criticality when: a UI button was pressed (forced_tool), or this
    is a resumed turn after a doctor_need_location pause (plan already set —
    planner_node will just pass it through, no re-planning)."""
    if state.get("forced_tool") or state.get("plan"):
        return "planner"
    return "criticality"


def _route_after_criticality(state: AgentState) -> str:
    """criticality_node already set plan=['first_aid','doctor'] for the
    critical case — no need to ask the planner LLM to figure that out."""
    if state.get("plan"):
        return "executor"
    return "planner"


def _route_after_executor(state: AgentState) -> str:
    """Pause here if the doctor tool is waiting on user_state/user_city."""
    if state.get("stage") == "doctor_need_location":
        return "end"
    return "synthesis"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("welcome", welcome_node)
    builder.add_node("intake", intake_node)
    builder.add_node("criticality", criticality_node)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("synthesis", synthesis_node)

    builder.set_entry_point("welcome")
    builder.add_edge("welcome", "intake")

    builder.add_conditional_edges(
        "intake",
        _route_after_intake,
        {"planner": "planner", "criticality": "criticality"},
    )

    builder.add_conditional_edges(
        "criticality",
        _route_after_criticality,
        {"executor": "executor", "planner": "planner"},
    )

    builder.add_edge("planner", "executor")

    builder.add_conditional_edges(
        "executor",
        _route_after_executor,
        {"synthesis": "synthesis", "end": END},
    )

    builder.add_edge("synthesis", END)

    return builder.compile()


# graph = build_graph()
# print(graph.get_graph().draw_ascii())