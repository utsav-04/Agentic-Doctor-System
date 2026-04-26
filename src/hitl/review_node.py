"""
src/hitl/review_node.py
========================
Human-in-the-Loop using LangGraph's interrupt mechanism.

How LangGraph interrupt works:
  1. Graph executes normally until hitl_node is reached
  2. `interrupt(payload)` pauses the graph and surfaces the payload
     to the caller (Streamlit UI or CLI)
  3. The graph state is checkpointed — nothing is lost
  4. The caller collects the human decision
  5. `graph.invoke(Command(resume=decision), config=config)` resumes
     the graph from the exact point it paused
  6. hitl_node receives the resumed value and routes accordingly

This is the correct pattern for HITL in LangGraph v0.2+.
The old input() approach blocks the event loop and doesn't work
with Streamlit or any async UI.

CLI usage:
    result = graph.invoke(state, config={"configurable": {"thread_id": "1"}})
    # graph pauses at interrupt — result contains the interrupt payload
    decision = input("Approve? (A/R): ")
    result   = graph.invoke(Command(resume={"decision": decision}), config=config)

Streamlit usage:
    See src/ui/app.py — the UI renders a review panel and resumes on button click.
"""

from langgraph.types import interrupt, Command
from langsmith import traceable

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graph.state import AgentState


@traceable(name="hitl_review_node", run_type="chain")
def hitl_node(state: AgentState) -> dict:
    """
    Pauses graph execution for human review.

    Surfaces the agent output to the reviewer via LangGraph interrupt.
    The graph is checkpointed here — resumes when the reviewer responds.

    LangSmith traces:
        - agent_output shown to reviewer
        - decision received (approved / revised)
        - feedback if revised
        - which node the graph routes to next
    """
    agent_output = state.get("agent_output", "")
    stage        = state.get("stage", "")
    criticality  = state.get("criticality", "mild")
    symptoms     = state.get("symptoms", "")

    # Pause the graph and surface the review payload to the caller.
    # The caller (Streamlit or CLI) receives this dict and shows it to the human.
    human_decision = interrupt({
        "type":        "hitl_review",
        "agent_output": agent_output,
        "stage":        stage,
        "criticality":  criticality,
        "symptoms":     symptoms,
        "instructions": (
            "Review the agent output above.\n"
            "Approve to send to user, or provide feedback for revision."
        ),
    })

    # Graph resumes here after the human responds.
    # human_decision is the dict passed via Command(resume=...) by the caller.
    decision = human_decision.get("decision", "approve").lower()
    feedback = human_decision.get("feedback", "")

    messages = state.get("messages", [])

    if decision in ("approve", "a"):
        messages.append("[HITL] Approved by human reviewer.")
        return {
            "hitl_approved": True,
            "hitl_feedback": "",
            "messages":      messages,
            "stage":         "hitl_approved",
        }

    messages.append(f"[HITL] Revision requested: {feedback}")
    return {
        "hitl_approved": False,
        "hitl_feedback": feedback,
        "messages":      messages,
        "stage":         f"hitl_revise_{stage}",
    }