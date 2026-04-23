"""
src/graph/nodes.py
===================
Every LangGraph node as a standalone function.
Each node receives the full AgentState and returns
only the keys it wants to update — LangGraph merges the rest.

Node order matches the workflow diagram:
  welcome → intake → criticality → [critical: doctor] or [not critical: intent → medicine/lab] → hitl → response
"""

import sys
from pathlib import Path

from crewai import Agent, Crew, Task
from langsmith import traceable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graph.state   import AgentState
from src.graph.router  import detect_criticality, detect_intent
from src.agents.agents import (
    intake_agent,
    criticality_agent,
    medicine_agent,
    doctor_agent,
    lab_agent,
)

WELCOME_MESSAGE = """
Welcome to the Agentic Doctor System!

I can help you with:
  1. Checking how serious your symptoms are (criticality check)
  2. Finding the right medicines for your condition
  3. Suggesting lab tests you should get done
  4. Finding nearby doctors and hospitals
  5. First aid and Indian home remedies

Please tell me your name, age, and what symptoms you are experiencing.
"""


# ─────────────────────────────────────────────────────────────────────────────
# WELCOME NODE
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="welcome_node", run_type="chain")
def welcome_node(state: AgentState) -> dict:
    """Greets the user and lists available features."""
    print(WELCOME_MESSAGE)
    return {
        "messages": [WELCOME_MESSAGE],
        "stage":    "welcome",
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTAKE NODE
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="intake_node", run_type="chain")
def intake_node(state: AgentState) -> dict:
    """Collects user name, age, symptoms, and basic history via intake_agent."""
    user_input = state.get("user_input", "")

    task = Task(
        description=f"""
        Collect the following information from the user in a warm and clear manner:
        1. Full name
        2. Age
        3. Gender
        4. Primary symptoms and how long they have been experiencing them
        5. Any known medical conditions or allergies

        If the user has already provided some information below, extract it
        and ask only for what is missing.

        User message: {user_input}
        """,
        agent=intake_agent,
        expected_output=(
            "A structured summary: name, age, gender, symptoms (as a clean sentence), "
            "duration, known conditions, allergies."
        ),
    )

    crew   = Crew(agents=[intake_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    output = str(result)

    user_info = {
        "raw_intake": output,
        "user_input": user_input,
    }

    messages = state.get("messages", [])
    messages.append(f"[Intake] {output}")

    return {
        "user_info": user_info,
        "symptoms":  user_input,
        "messages":  messages,
        "stage":     "intake",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CRITICALITY NODE
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="criticality_node", run_type="chain")
def criticality_node(state: AgentState) -> dict:
    """Triages the user's symptoms and assigns a severity level."""
    symptoms = state.get("symptoms", state.get("user_input", ""))

    criticality = detect_criticality(symptoms)
    print(f"\n[Criticality] Assessed as: {criticality.upper()}")

    task = Task(
        description=f"""
        The patient reports these symptoms: {symptoms}

        Severity has been assessed as: {criticality.upper()}

        Your job:
        - If CRITICAL: immediately provide first aid steps using the first_aid_guide tool,
          tell the user to call 108 or go to the nearest emergency room without delay.
        - If MODERATE: reassure the user and explain they need to see a doctor soon.
        - If MILD: reassure the user that their condition can likely be managed with
          medicine or home remedies.
        """,
        agent=criticality_agent,
        expected_output="A clear, empathetic triage message appropriate for the severity level.",
    )

    crew   = Crew(agents=[criticality_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    output = str(result)

    messages = state.get("messages", [])
    messages.append(f"[Criticality: {criticality}] {output}")

    return {
        "criticality": criticality,
        "agent_output": output,
        "messages":    messages,
        "stage":       "criticality",
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTENT NODE  (only reached when NOT critical)
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="intent_node", run_type="chain")
def intent_node(state: AgentState) -> dict:
    """Asks the user what kind of help they need and classifies their intent."""
    user_input = state.get("user_input", "")

    PROMPT = (
        "\nHow can I help you further?\n"
        "  1. Medicine suggestions\n"
        "  2. Lab test suggestions\n"
        "  3. Find a nearby doctor\n"
        "  4. First aid / home remedies\n"
        "  Type your choice or just describe what you need."
    )
    print(PROMPT)

    intent = detect_intent(user_input)
    print(f"[Intent] Detected: {intent}")

    messages = state.get("messages", [])
    messages.append(f"[Intent detected: {intent}]")

    return {
        "intent":   intent,
        "messages": messages,
        "stage":    "intent",
    }


# ─────────────────────────────────────────────────────────────────────────────
# DOCTOR NODE  (triggered when criticality = critical)
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="doctor_node", run_type="chain")
def doctor_node(state: AgentState) -> dict:
    """Finds nearby doctors — triggered for critical or doctor-intent cases."""
    symptoms   = state.get("symptoms", state.get("user_input", ""))
    user_state = state.get("user_state", "")
    user_city  = state.get("user_city", "")

    if not user_state:
        ask = (
            "\nTo find a nearby doctor, I need your location.\n"
            "Please tell me your state and city (e.g. Maharashtra, Mumbai)."
        )
        print(ask)
        messages = state.get("messages", [])
        messages.append("[Doctor] Waiting for location from user.")
        return {
            "messages": messages,
            "stage":    "doctor_waiting_location",
            "agent_output": ask,
        }

    task = Task(
        description=f"""
        The patient has reported these symptoms: {symptoms}
        Their severity is: {state.get('criticality', 'critical').upper()}
        Their location: {user_city}, {user_state}

        Use the doctor_finder tool to find suitable nearby doctors.
        If severity is critical, set emergency=True in the tool call.
        Present results clearly so the patient can act immediately.
        """,
        agent=doctor_agent,
        expected_output="A clear list of nearby doctors with contact details and availability.",
    )

    crew   = Crew(agents=[doctor_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    output = str(result)

    messages = state.get("messages", [])
    messages.append(f"[Doctor Finder] {output}")

    return {
        "agent_output": output,
        "messages":     messages,
        "stage":        "doctor",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MEDICINE NODE
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="medicine_node", run_type="chain")
def medicine_node(state: AgentState) -> dict:
    """Provides medicine recommendations using RAG."""
    symptoms  = state.get("symptoms", state.get("user_input", ""))
    user_info = state.get("user_info", {})
    age       = user_info.get("age", -1)

    task = Task(
        description=f"""
        The patient reports: {symptoms}
        Patient age: {age if age != -1 else 'unknown'}

        Use the medicine_finder tool to find appropriate medicines.
        Make sure to use the correct dosage for the patient's age group.
        Always include side effects and at least one important warning.
        Never recommend medicines without using the tool.
        """,
        agent=medicine_agent,
        expected_output=(
            "A clear list of medicines with name, dosage, duration, side effects, and warnings."
        ),
    )

    crew   = Crew(agents=[medicine_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    output = str(result)

    messages = state.get("messages", [])
    messages.append(f"[Medicine] {output}")

    return {
        "agent_output": output,
        "messages":     messages,
        "stage":        "medicine",
    }


# ─────────────────────────────────────────────────────────────────────────────
# LAB NODE
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="lab_node", run_type="chain")
def lab_node(state: AgentState) -> dict:
    """Suggests lab tests using RAG."""
    symptoms = state.get("symptoms", state.get("user_input", ""))

    task = Task(
        description=f"""
        The patient reports: {symptoms}

        Use the lab_test_suggester tool to recommend appropriate diagnostic tests.
        Explain why each test is relevant to the patient's symptoms.
        Remind the patient that a doctor must order and interpret the tests.
        Never suggest tests without using the tool.
        """,
        agent=lab_agent,
        expected_output="A clear list of recommended lab tests with reasons for each.",
    )

    crew   = Crew(agents=[lab_agent], tasks=[task], verbose=True)
    result = crew.kickoff()
    output = str(result)

    messages = state.get("messages", [])
    messages.append(f"[Lab Tests] {output}")

    return {
        "agent_output": output,
        "messages":     messages,
        "stage":        "lab",
    }


# ─────────────────────────────────────────────────────────────────────────────
# HITL NODE  (human-in-the-loop review)
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="hitl_node", run_type="chain")
def hitl_node(state: AgentState) -> dict:
    """
    Interrupts the graph for a human reviewer to approve or revise agent output.
    In production this integrates with a Streamlit UI review panel.
    In CLI mode it prompts the terminal directly.
    """
    agent_output = state.get("agent_output", "")

    print("\n" + "="*60)
    print("  HUMAN REVIEW REQUIRED")
    print("="*60)
    print("\nAgent output to review:\n")
    print(agent_output)
    print("\n" + "─"*60)
    print("Options:")
    print("  [A] Approve — send this to user")
    print("  [R] Revise  — provide feedback for revision")
    print("─"*60)

    decision = input("\nYour decision (A/R): ").strip().upper()

    if decision == "A":
        messages = state.get("messages", [])
        messages.append("[HITL] Approved by human reviewer.")
        return {
            "hitl_approved":  True,
            "hitl_feedback":  "",
            "messages":       messages,
            "stage":          "hitl_approved",
        }

    feedback = input("Enter revision feedback: ").strip()
    messages = state.get("messages", [])
    messages.append(f"[HITL] Revision requested: {feedback}")
    return {
        "hitl_approved": False,
        "hitl_feedback": feedback,
        "messages":      messages,
        "stage":         "hitl_revise",
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE NODE
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="response_node", run_type="chain")
def response_node(state: AgentState) -> dict:
    """Formats and delivers the final verified response to the user."""
    agent_output = state.get("agent_output", "")
    criticality  = state.get("criticality", "mild")

    header = {
        "critical": "URGENT — Please act on this information immediately.",
        "moderate": "Please review the following and consult a doctor soon.",
        "mild":     "Here is the information based on your query.",
    }.get(criticality, "")

    final = f"{header}\n\n{agent_output}\n\nIf you have more questions, feel free to ask."

    print("\n" + "="*60)
    print(final)
    print("="*60 + "\n")

    messages = state.get("messages", [])
    messages.append(f"[Response] {final}")

    return {
        "final_response": final,
        "messages":       messages,
        "stage":          "done",
    }
