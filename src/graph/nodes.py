"""
src/graph/nodes.py
===================
LangGraph nodes. Nodes process state — they do NOT call input().
All user prompting happens in the API layer / UI.

Flow this file supports (wired together in graphs.py):

    welcome -> intake -> criticality -> [critical]-----> executor -> synthesis -> END
                                     -> [not critical] -> planner -> executor -> synthesis -> END

    forced_tool (UI button) set -> planner (bypass mode) -> executor -> synthesis -> END

Removed from the old version: doctor_node / medicine_node / lab_node / intent_node /
response_node (replaced by planner_node + executor_node + synthesis_node), hitl_node
(never wired into a graph — reintroduce later if you actually want human review),
and ~800 lines of commented-out earlier implementations.
"""

import re
import sys
from pathlib import Path

from crewai import Crew, Task
from langsmith import traceable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graph.state import AgentState
from src.graph.router import detect_criticality, detect_plan
from src.agents.agents import intake_agent, criticality_agent, llm
from src.tools.medicine_tool import medicine_finder
from src.tools.lab_tool import lab_test_suggester
from src.tools.doctor_tool import doctor_finder
from src.tools.first_aid_tool import first_aid_guide
from src.logger import get_logger

logger = get_logger(__name__)

NEED_LOCATION = "NEED_LOCATION"


def _clean(text: str) -> str:
    """Strip <think>...</think> blocks that some local models add."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _extract_symptoms(raw: str) -> str:
    """
    Extracts only the symptom text from the intake string.
    Input:  'Name: Utsav Raj\\nAge: 5\\nGender: Male\\nSymptoms: Cold from yesterday'
    Output: 'Cold from yesterday'
    """
    match = re.search(r"[Ss]ymptoms?\s*:\s*(.+?)(?:\n|$)", raw)
    if match:
        return match.group(1).strip()
    return raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# WELCOME NODE
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="welcome_node", run_type="chain")
def welcome_node(state: AgentState) -> dict:
    logger.info("welcome_node executed")
    return {"messages": [], "stage": "welcome"}


# ─────────────────────────────────────────────────────────────────────────────
# INTAKE NODE
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="intake_node", run_type="chain")
def intake_node(state: AgentState) -> dict:
    user_input = state.get("user_input", "")
    logger.info("intake_node — processing: '%s'", user_input[:80])

    symptoms_only = _extract_symptoms(user_input) or user_input

    messages = list(state.get("messages", []))
    messages.append(f"[Intake] symptoms='{symptoms_only}'")

    logger.info("intake_node complete — symptoms: '%s'", symptoms_only[:80])

    return {
        "symptoms": symptoms_only,
        "messages": messages,
        "stage": "intake",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CRITICALITY NODE
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="criticality_node", run_type="chain")
def criticality_node(state: AgentState) -> dict:
    symptoms = state.get("symptoms", state.get("user_input", ""))
    logger.info("criticality_node — symptoms: '%s'", symptoms[:80])

    criticality = detect_criticality(symptoms)
    logger.info("criticality_node — result: %s", criticality.upper())

    task = Task(
        description=f"""
        Patient symptoms: {symptoms}
        Assessed severity: {criticality.upper()}

        Write a short, clear, empathetic triage message (2-3 sentences, text only —
        do NOT call any tool):
        - CRITICAL: Tell the patient this is an emergency. Tell them to call 108 or
          go to the emergency room NOW. First aid steps will be provided separately.
        - MODERATE: Reassure them but advise seeing a doctor soon.
        - MILD: Reassure them it can likely be managed at home.
        """,
        agent=criticality_agent,
        expected_output="A short empathetic triage message, text only, no tool calls.",
    )

    crew = Crew(agents=[criticality_agent], tasks=[task], verbose=False)
    result = crew.kickoff()
    output = _clean(str(result))

    messages = list(state.get("messages", []))
    messages.append(f"[Criticality: {criticality.upper()}]\n{output}")

    update = {
        "criticality": criticality.lower(),
        "agent_output": output,
        "messages": messages,
        "stage": "criticality",
    }

    # Critical cases skip the planner entirely — the plan is fixed:
    # first aid guidance immediately, then point to a doctor.
    if criticality.lower() == "critical":
        update["plan"] = ["first_aid", "doctor"]

    return update


# ─────────────────────────────────────────────────────────────────────────────
# PLANNER NODE  (replaces intent_node — produces a list, not a single label)
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="planner_node", run_type="chain")
def planner_node(state: AgentState) -> dict:
    """
    Two modes:
      1. forced_tool is set (UI button pressed) -> plan is exactly that one tool.
         No LLM call needed — this is a hard bypass.
      2. forced_tool is None -> ask the LLM to plan the full tool list for this
         free-text query (see detect_plan in router.py).
    """
    forced_tool = state.get("forced_tool")

    if forced_tool:
        logger.info("planner_node — forced_tool='%s', skipping LLM planning", forced_tool)
        return {"plan": [forced_tool], "stage": "planned_forced"}

    # Resuming after a doctor_need_location pause — the plan was already
    # decided last turn (and possibly partly executed). Don't re-plan.
    if state.get("plan"):
        logger.info("planner_node — resuming existing plan: %s", state["plan"])
        return {"plan": state["plan"], "stage": "planned_resumed"}

    user_input = state.get("user_input", "")
    symptoms = state.get("symptoms", user_input)

    plan = detect_plan(user_input=user_input, symptoms=symptoms)
    logger.info("planner_node — plan: %s", plan)

    messages = list(state.get("messages", []))
    messages.append(f"[Plan] {plan}")

    return {"plan": plan, "messages": messages, "stage": "planned"}


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTOR NODE  (replaces doctor_node / medicine_node / lab_node)
# ─────────────────────────────────────────────────────────────────────────────

def _call_medicine(state: AgentState) -> str:
    symptoms = state.get("symptoms", "")
    age = state.get("user_age")
    age_val = int(age) if age and str(age).isdigit() else -1

    return medicine_finder.run(**{
        "symptoms": symptoms,
        "disease": "",
        "patient_age": age_val,
        "top_k": 3,
    })


def _call_lab(state: AgentState) -> str:
    symptoms = state.get("symptoms", "")
    return lab_test_suggester.run(**{
        "symptoms": symptoms,
        "severity": "",
        "top_k": 3,
    })


def _call_first_aid(state: AgentState) -> str:
    symptoms = state.get("symptoms", "")
    criticality = state.get("criticality", "mild")
    severity_map = {"critical": "life_threatening", "moderate": "urgent", "mild": "mild"}
    return first_aid_guide.run(**{
        "condition": symptoms,
        "severity": severity_map.get(criticality, ""),
        "top_k": 2,
    })


def _call_doctor(state: AgentState) -> str:
    symptoms = state.get("symptoms", "")
    user_state = state.get("user_state", "")
    user_city = state.get("user_city", "")
    criticality = state.get("criticality", "moderate")

    if not user_state:
        return NEED_LOCATION

    department = ""
    s = symptoms.lower()
    if "chest pain" in s:
        department = "Cardiology"
    elif "fever" in s:
        department = "General Physician"
    elif "skin" in s:
        department = "Dermatology"
    elif "child" in s or "kid" in s:
        department = "Pediatrics"
    elif "pregnant" in s or "pregnancy" in s:
        department = "Gynecology"

    return doctor_finder.run(**{
        "symptoms": symptoms,
        "state": user_state,
        "city": user_city,
        "department": department,
        "emergency": criticality == "critical",
        "top_k": 3,
    })


_TOOL_DISPATCH = {
    "medicine": _call_medicine,
    "lab": _call_lab,
    "first_aid": _call_first_aid,
    "doctor": _call_doctor,
}


@traceable(name="executor_node", run_type="chain")
def executor_node(state: AgentState) -> dict:
    """
    Walks state['plan'] in order and calls each tool directly — no CrewAI
    Agent/Task/Crew wrapping. The plan already says exactly which tool to
    call; there's no decision left for an agent to make.

    Special case: doctor tool needs a location. If it's missing, execution
    stops and signals the caller (API/UI layer) to ask for state + city,
    rather than silently skipping the doctor step.
    """
    plan = state.get("plan", [])
    logger.info("executor_node — plan: %s", plan)

    tool_outputs = dict(state.get("tool_outputs", {}))
    messages = list(state.get("messages", []))

    for tool_name in plan:
        if tool_name in tool_outputs:
            continue  # already have this one (e.g. re-entry after location was provided)

        handler = _TOOL_DISPATCH.get(tool_name)
        if handler is None:
            logger.warning("executor_node — unknown tool in plan: %s", tool_name)
            continue

        output = handler(state)

        if output == NEED_LOCATION:
            logger.info("executor_node — doctor tool needs location, pausing")
            messages.append("[Executor] Waiting for user_state/user_city to find a doctor.")
            return {
                "tool_outputs": tool_outputs,
                "messages": messages,
                "stage": "doctor_need_location",
            }

        tool_outputs[tool_name] = _clean(output)
        messages.append(f"[{tool_name.upper()}]\n{output}")

    logger.info("executor_node complete — tools run: %s", list(tool_outputs.keys()))

    return {
        "tool_outputs": tool_outputs,
        "messages": messages,
        "stage": "executed",
    }


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHESIS NODE  (replaces response_node — merges N tool outputs into 1 answer)
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="synthesis_node", run_type="chain")
def synthesis_node(state: AgentState) -> dict:
    forced_tool = state.get("forced_tool")
    tool_outputs = state.get("tool_outputs", {})
    criticality = state.get("criticality", "mild")

    # Button mode: return exactly that tool's output, nothing added, nothing
    # merged, and no follow-up offer — "just give me medicine" means exactly that.
    if forced_tool:
        final = tool_outputs.get(forced_tool, "No result available.")
        logger.info("synthesis_node — forced_tool mode, raw passthrough for '%s'", forced_tool)
        return {"final_response": final, "stage": "done"}

    # No tool applied — general/greeting query.
    if not tool_outputs:
        final = state.get("agent_output") or (
            "Could you tell me a bit more about your symptoms so I can help?"
        )
        return {"final_response": final, "stage": "done"}

    header = {
        "critical": "URGENT — please act immediately.",
        "moderate": "Please consult a doctor soon.",
        "mild": "Here is the information based on your query.",
    }.get(criticality, "")

    if len(tool_outputs) == 1:
        core = next(iter(tool_outputs.values()))
    else:
        combined_raw = "\n\n".join(
            f"[{name.upper()} RESULT]\n{output}" for name, output in tool_outputs.items()
        )
        prompt = f"""
You are combining the results of multiple medical tools into one clear response
for a patient. Do not invent new medical facts — only reorganize and connect
what's already in the tool results below into a single, well-structured answer.
Keep all dosages, warnings, and disclaimers exactly as given.

{combined_raw}

Write one combined response covering all of the above, in a natural order
(medicine before lab before doctor referral, if all present).
"""
        response = llm.call(prompt)
        core = _clean(str(response))

    final = f"{header}\n\n{core}" if header else core

    messages = list(state.get("messages", []))
    messages.append(f"[Synthesis]\n{final}")

    update = {"final_response": final, "messages": messages, "stage": "done"}

    # Offer lab test / doctor as a follow-up when this was a symptom-based
    # query (medicine or first_aid ran) that didn't already cover them.
    # Skipped for critical cases — doctor referral is already mandatory there.
    if criticality != "critical":
        missing = [t for t in ("lab", "doctor") if t not in tool_outputs]
        symptom_based = any(t in tool_outputs for t in ("medicine", "first_aid"))
        if missing and symptom_based:
            offer_text = " and ".join(missing)
            update["final_response"] += (
                f"\n\nWould you also like {offer_text} suggestions? "
                f"(reply with 'lab', 'doctor', 'both', or 'no')"
            )
            update["follow_up_offered"] = missing

    logger.info("synthesis_node complete — tools: %s, follow_up_offered: %s",
                list(tool_outputs.keys()), update.get("follow_up_offered"))

    return update