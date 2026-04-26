"""
main.py
========
CLI entry point — handles all user interaction and conversation flow.

Flow:
  1. Show welcome message
  2. Collect intake info (name, age, symptoms) — ask step by step
  3. Run criticality check — show result to user
  4. If CRITICAL → ask location → run doctor search → HITL → show result
  5. If NOT CRITICAL → ask what they need → run medicine/lab/doctor → HITL → show result

Run:
    python main.py
"""

import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT",    "agentic-doctor-system")

from langgraph.types import Command
from src.graph.graphs import build_graph
from src.logger import get_logger

logger = get_logger(__name__)


def divider(char="─", width=60):
    print(char * width)


def ask(prompt: str) -> str:
    """Prompt the user and return stripped input."""
    return input(prompt).strip()


def show(text: str):
    """Print agent output with a divider."""
    divider()
    print(text)
    divider()


def get_interrupt_payload(graph, config):
    """Extract interrupt payload from graph state if paused."""
    try:
        state_snapshot = graph.get_state(config)
        if not state_snapshot.next:
            return None
        for task in state_snapshot.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                return task.interrupts[0].value
    except Exception:
        pass
    return None


def handle_hitl(graph, config) -> dict:
    """
    Handles the HITL review loop.
    Shows agent output to reviewer, collects approve/revise decision,
    resumes graph, repeats if revised.
    Returns final graph result.
    """
    result = {}
    while True:
        payload = get_interrupt_payload(graph, config)
        if not payload:
            break

        print("\n" + "="*60)
        print("  HUMAN REVIEW REQUIRED")
        print("="*60)
        print(f"\nSymptoms    : {payload.get('symptoms', '')}")
        print(f"Criticality : {payload.get('criticality', '').upper()}")
        print(f"Stage       : {payload.get('stage', '')}")
        print("\nAgent Output:\n")
        print(payload.get("agent_output", ""))
        divider()
        print("  [A] Approve — send to user")
        print("  [R] Revise  — provide feedback")
        divider()

        decision = ask("\nYour decision (A/R): ").upper()
        feedback = ""
        if decision != "A":
            feedback = ask("Feedback for revision: ")

        result = graph.invoke(
            Command(resume={
                "decision": "approve" if decision == "A" else "revise",
                "feedback": feedback,
            }),
            config=config,
        )

        # If revised, loop back to check for another interrupt
        if decision == "A":
            break

    return result


def run_cli():
    print("\n" + "="*60)
    print("  Agentic Doctor System — v0.1")
    print("="*60)

    graph     = build_graph()
    thread_id = str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    logger.info("Session started — thread_id: %s", thread_id)

    # ── STEP 1: Welcome (ONLY FIRST INVOKE) ───────────────────────
    result = graph.invoke({"messages": [], "stage": ""}, config=config)
    print(result.get("agent_output", ""))

    # ── STEP 2: Collect intake info ───────────────────────────────
    print("\nLet's start by getting to know you a little.")
    print("(You can skip any field by pressing Enter)\n")

    name    = ask("Your name: ")
    age     = ask("Your age: ")
    gender  = ask("Your gender (Male/Female/Other): ")
    symptoms = ask("\nDescribe your symptoms in detail:\n> ")

    if not symptoms:
        print("No symptoms provided. Exiting.")
        return

    intake_text = (
        f"Name: {name or 'not provided'}. "
        f"Age: {age or 'not provided'}. "
        f"Gender: {gender or 'not provided'}. "
        f"Symptoms: {symptoms}."
    )

    print("\nProcessing your information...\n")

    # ── STEP 3: Resume graph → intake → criticality ───────────────
    result = graph.invoke(
        Command(resume={
            "user_input": intake_text,
            "user_name":  name,
            "user_age":   age,
            "user_gender": gender,
            "symptoms":   symptoms,
        }),
        config=config,
    )

    # Handle HITL if triggered early (rare but safe)
    payload = get_interrupt_payload(graph, config)
    if payload:
        result = handle_hitl(graph, config)

    criticality = result.get("criticality", "")
    crit_output = result.get("agent_output", "")

    print("\n")
    show(crit_output)

    # ── STEP 4: Critical vs Non-critical ─────────────────────────
    if criticality == "critical":
        print("\n⚠️ This is an emergency situation.")
        print("To find a nearby doctor, I need your location.\n")

        user_state = ask("Your state: ").strip().title()
        user_city  = ask("Your city: ").strip().title()

        result = graph.invoke(
            Command(resume={
                "user_state": user_state,
                "user_city":  user_city,
                "user_input": f"emergency in {user_city}, {user_state}",
            }),
            config=config,
        )

    else:
        print("\nHow can I help you further?")
        print("  1. Medicine suggestions")
        print("  2. Lab test recommendations")
        print("  3. Find a nearby doctor")
        print("  4. First aid / home remedies\n")

        user_choice = ask("Your choice: ")

        user_state = ""
        user_city  = ""

        if any(k in user_choice.lower() for k in ["doctor", "hospital", "3"]):
            user_state = ask("Your state: ").strip().title()
            user_city  = ask("Your city: ").strip().title()

        result = graph.invoke(
            Command(resume={
                "user_input": user_choice,
                "user_state": user_state,
                "user_city":  user_city,
            }),
            config=config,
        )

    # ── STEP 5: Handle NEED_LOCATION loop ────────────────────────
    while result.get("agent_output") == "NEED_LOCATION":
        print("\nTo find a doctor I need your location.\n")

        user_state = ask("Your state: ").strip().title()
        user_city  = ask("Your city: ").strip().title()

        result = graph.invoke(
            Command(resume={
                "user_state": user_state,
                "user_city":  user_city,
            }),
            config=config,
        )

    # ── STEP 6: HITL Review ───────────────────────────────────────
    payload = get_interrupt_payload(graph, config)
    if payload:
        result = handle_hitl(graph, config)

    # ── STEP 7: Final Output ─────────────────────────────────────
    final = result.get("final_response", "")

    if final:
        print("\n" + "="*60)
        print("  VERIFIED RESPONSE")
        print("="*60)
        print(final)
        print("="*60)
    else:
        out = result.get("agent_output", "")
        if out and out != "NEED_LOCATION":
            print("\n" + "="*60)
            print(out)
            print("="*60)

    print(f"\nSession ID : {thread_id}")
    print("Traces     : https://smith.langchain.com")
    print("Log file   : logs/agentic_doctor.log\n")

    logger.info("Session complete — thread_id: %s", thread_id)

if __name__ == "__main__":
    run_cli()