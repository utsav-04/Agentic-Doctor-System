"""
src/graph/state.py
===================
Central state schema for the LangGraph state machine.
"""

from typing import TypedDict, Optional, List


class AgentState(TypedDict, total=False):
    # ── Conversation ──────────────────────────────────────────────
    messages:        List[str]      # full conversation history
    user_input:      str            # latest raw user message

    # ── User profile (collected during intake) ────────────────────
    user_name:       str
    user_age:        str
    user_gender:     str
    user_info:       dict           # full collected intake dict
    symptoms:        str            # cleaned symptom string
    user_state:      str            # Indian state e.g. 'Maharashtra'
    user_city:       str            # city e.g. 'Mumbai'

    # ── Routing ───────────────────────────────────────────────────
    criticality:     str            # 'critical' | 'moderate' | 'mild'
    intent:          str            # 'medicine' | 'lab' | 'doctor' | 'both'
    stage:           str            # current node name

    # ── Agent output (before HITL review) ────────────────────────
    agent_output:    str

    # ── HITL ──────────────────────────────────────────────────────
    hitl_approved:   Optional[bool]
    hitl_feedback:   str

    # ── Final ─────────────────────────────────────────────────────
    final_response:  str