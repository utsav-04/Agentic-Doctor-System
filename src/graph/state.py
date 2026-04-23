"""
src/graph/state.py
===================
Central state schema for the LangGraph state machine.

Every node reads from and writes to this TypedDict.
LangGraph merges returned dicts into the running state,
so nodes only need to return the keys they changed.

State lifecycle:
  welcome_node      → sets: messages, stage
  intake_node       → sets: user_info, symptoms, stage
  criticality_node  → sets: criticality, stage
  intent_node       → sets: intent, stage           (only if NOT critical)
  doctor_node       → sets: agent_output, stage     (if critical)
  medicine_node     → sets: agent_output, stage     (if intent=medicine)
  lab_node          → sets: agent_output, stage     (if intent=lab)
  hitl_node         → sets: hitl_approved, hitl_feedback, stage
  response_node     → sets: final_response, stage
"""

from typing import TypedDict, Optional, List


class AgentState(TypedDict, total=False):
    # ── Conversation ──────────────────────────────────────────────
    messages:        List[str]          # full conversation history
    user_input:      str                # latest raw user message

    # ── User profile (collected by intake_node) ───────────────────
    user_info:       dict               # name, age, gender, history
    symptoms:        str                # cleaned symptom string
    user_state:      str                # Indian state e.g. 'Maharashtra'
    user_city:       str                # city e.g. 'Mumbai'

    # ── Routing ───────────────────────────────────────────────────
    criticality:     str                # 'critical' | 'moderate' | 'mild'
    intent:          str                # 'medicine' | 'lab' | 'both' | 'none'
    stage:           str                # current node name (for tracing)

    # ── Agent output (raw, before HITL review) ────────────────────
    agent_output:    str                # raw output from medicine/lab/doctor agent

    # ── HITL ──────────────────────────────────────────────────────
    hitl_approved:   Optional[bool]     # True=approved, False=revise
    hitl_feedback:   str                # doctor's revision comment

    # ── Final ─────────────────────────────────────────────────────
    final_response:  str                # verified response shown to user
