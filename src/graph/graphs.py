# """
# src/graph/graphs.py
# ====================
# ONE graph, ONE invoke() per user turn.

#     welcome -> intake -> cache_check -> [cache hit?]
#                                            |-- yes --> END (instant return)
#                                            |-- no  --> [forced_tool or resumed plan?]
#                                                           |-- yes --> planner (bypass/resume)
#                                                           |-- no  --> criticality -> [plan already set?]
#                                                                           |-- yes (critical) --> executor
#                                                                           |-- no             --> planner
#                                           planner --> executor --> [need location?]
#                                                                        |-- yes --> END (pause)
#                                                                        |-- no  --> synthesis --> END

# Two legitimate reasons a single invoke() can end without going through
# synthesis: a full-response cache hit (stage == 'cache_hit'), or the doctor
# tool pausing for location (stage == 'doctor_need_location'). Both are
# handled by the caller (test.py / API layer) — a cache hit needs no further
# action, a location pause needs a re-invoke with user_state/user_city set.
# """

# from langgraph.graph import StateGraph, END

# from src.graph.state import AgentState
# from src.graph.nodes import (
#     welcome_node,
#     intake_node,
#     cache_check_node,
#     criticality_node,
#     planner_node,
#     executor_node,
#     synthesis_node,
# )


# def _route_after_cache_check(state: AgentState) -> str:
#     """Cache hit -> we're done, no need to touch criticality/planner/executor/
#     synthesis at all. Otherwise fall through to the same forced_tool/resume
#     logic as before."""
#     if state.get("stage") == "cache_hit":
#         return "end"
#     if state.get("forced_tool") or state.get("plan"):
#         return "planner"
#     return "criticality"


# def _route_after_criticality(state: AgentState) -> str:
#     """criticality_node already set plan=['first_aid','doctor'] for the
#     critical case — no need to ask the planner LLM to figure that out."""
#     if state.get("plan"):
#         return "executor"
#     return "planner"


# def _route_after_executor(state: AgentState) -> str:
#     """Pause here if the doctor tool is waiting on user_state/user_city."""
#     if state.get("stage") == "doctor_need_location":
#         return "end"
#     return "synthesis"


# def build_graph():
#     builder = StateGraph(AgentState)

#     builder.add_node("welcome", welcome_node)
#     builder.add_node("intake", intake_node)
#     builder.add_node("cache_check", cache_check_node)
#     builder.add_node("criticality", criticality_node)
#     builder.add_node("planner", planner_node)
#     builder.add_node("executor", executor_node)
#     builder.add_node("synthesis", synthesis_node)

#     builder.set_entry_point("welcome")
#     builder.add_edge("welcome", "intake")
#     builder.add_edge("intake", "cache_check")

#     builder.add_conditional_edges(
#         "cache_check",
#         _route_after_cache_check,
#         {"end": END, "planner": "planner", "criticality": "criticality"},
#     )

#     builder.add_conditional_edges(
#         "criticality",
#         _route_after_criticality,
#         {"executor": "executor", "planner": "planner"},
#     )

#     builder.add_edge("planner", "executor")

#     builder.add_conditional_edges(
#         "executor",
#         _route_after_executor,
#         {"synthesis": "synthesis", "end": END},
#     )

#     builder.add_edge("synthesis", END)

#     return builder.compile()


# # graph = build_graph()
# # print(graph.get_graph().draw_ascii())


"""
src/graph/graphs.py
====================
ONE graph, ONE invoke() per user turn.

    welcome -> intake -> cache_check -> [cache hit?]
                                           |-- yes --> END (instant return)
                                           |-- no  --> [forced_tool or resumed plan?]
                                                          |-- yes --> planner (bypass/resume)
                                                          |-- no  --> criticality -> [plan already set?]
                                                                          |-- yes (critical) --> executor
                                                                          |-- no             --> planner
                                          planner --> executor --> [need location?]
                                                                       |-- yes --> END (pause)
                                                                       |-- no  --> synthesis --> END

Two legitimate reasons a single invoke() can end without going through
synthesis: a full-response cache hit (stage == 'cache_hit'), or the doctor
tool pausing for location (stage == 'doctor_need_location'). Both are
handled by the caller (test.py / API layer) — a cache hit needs no further
action, a location pause needs a re-invoke with user_state/user_city set.
"""

from langgraph.graph import StateGraph, END

from src.graph.state import AgentState
from src.graph.nodes import (
    welcome_node,
    intake_node,
    cache_check_node,
    criticality_node,
    planner_node,
    executor_node,
    synthesis_node,
)


def _route_after_cache_check(state: AgentState) -> str:
    """Cache hit -> we're done, no need to touch criticality/planner/executor/
    synthesis at all. Otherwise fall through to the same forced_tool/resume
    logic as before."""
    if state.get("stage") == "cache_hit":
        return "end"
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
    """Pause here if a tool is waiting on missing info — either the doctor
    tool needing a location, or an order/booking tool needing a patient name."""
    if state.get("stage") in ("doctor_need_location", "need_patient_name"):
        return "end"
    return "synthesis"


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("welcome", welcome_node)
    builder.add_node("intake", intake_node)
    builder.add_node("cache_check", cache_check_node)
    builder.add_node("criticality", criticality_node)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("synthesis", synthesis_node)

    builder.set_entry_point("welcome")
    builder.add_edge("welcome", "intake")
    builder.add_edge("intake", "cache_check")

    builder.add_conditional_edges(
        "cache_check",
        _route_after_cache_check,
        {"end": END, "planner": "planner", "criticality": "criticality"},
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