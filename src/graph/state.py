"""
src/graph/state.py
===================
Central state schema for the LangGraph state machine.

Changes from the old version:
  - intent (single string)      → plan (list[str])   — supports multiple tools, any order
  - (new) tool_outputs           — raw output per tool, collected before synthesis
  - (new) forced_tool            — set by the UI tool buttons; bypasses planner entirely
  - (new) final_response stays, but is now built by synthesis_node, not response_node
"""

from typing import TypedDict, Optional, List, Dict


class AgentState(TypedDict, total=False):
    # ── Conversation ──────────────────────────────────────────────
    messages:        List[str]      # full conversation history
    user_input:      str            # latest raw user message

    # ── User profile (collected during intake) ────────────────────
    user_name:       str
    user_age:        str
    user_gender:     str
    user_info:       dict
    symptoms:        str            # cleaned symptom string
    user_state:      str            # Indian state e.g. 'Maharashtra'
    user_city:       str            # city e.g. 'Mumbai'

    # ── Routing ───────────────────────────────────────────────────
    criticality:     str            # 'critical' | 'moderate' | 'mild'
    forced_tool:     Optional[str]  # 'medicine' | 'lab' | 'doctor' | None
                                     # set by UI button — skips planner_node when present
    plan:            List[str]      # ordered list of tools to call this turn
                                     # e.g. ["medicine"], ["medicine", "lab"], ["doctor"]
    stage:           str            # current node name (debugging/logging)

    # ── Agent output ────────────────────────────────────────────
    tool_outputs:    Dict[str, str] # {"medicine": "...", "lab": "..."} — raw tool text,
                                     # keyed by tool name, filled in by executor_node
    agent_output:    str            # kept for backward compat / single-tool debugging
    follow_up_offered: List[str]    # e.g. ["lab", "doctor"] — set by synthesis_node when
                                     # it asked "want lab/doctor too?"; the caller (test.py /
                                     # API layer) reads this to know what a yes/no answers
    delivered_tools:  List[str]     # tool names whose output has already been shown to the
                                     # user (set by synthesis_node each turn). On a follow-up
                                     # resume, only tools NOT in this list get rendered —
                                     # otherwise every follow-up re-shows everything already
                                     # sent in an earlier chat message.

    # ── HITL (unused for now — kept as optional hook for later) ──
    hitl_approved:   Optional[bool]
    hitl_feedback:   str

    # ── Final ─────────────────────────────────────────────────────
    final_response:  str