# """
# src/graph/nodes.py
# ===================
# Every LangGraph node as a standalone function.
# Each node receives the full AgentState and returns
# only the keys it wants to update — LangGraph merges the rest.

# Node order matches the workflow diagram:
#   welcome → intake → criticality → [critical: doctor] or [not critical: intent → medicine/lab] → hitl → response
# """

# import sys
# from pathlib import Path

# from crewai import Agent, Crew, Task
# from langsmith import traceable

# sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# from src.graph.state   import AgentState
# from src.graph.router  import detect_criticality, detect_intent
# from src.agents.agents import (
#     intake_agent,
#     criticality_agent,
#     medicine_agent,
#     doctor_agent,
#     lab_agent,
# )

# WELCOME_MESSAGE = """
# Welcome to the Agentic Doctor System!

# I can help you with:
#   1. Checking how serious your symptoms are (criticality check)
#   2. Finding the right medicines for your condition
#   3. Suggesting lab tests you should get done
#   4. Finding nearby doctors and hospitals
#   5. First aid and Indian home remedies

# Please tell me your name, age, and what symptoms you are experiencing.
# """


# # ─────────────────────────────────────────────────────────────────────────────
# # WELCOME NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="welcome_node", run_type="chain")
# def welcome_node(state: AgentState) -> dict:
#     """Greets the user and lists available features."""
#     print(WELCOME_MESSAGE)
#     return {
#         "messages": [WELCOME_MESSAGE],
#         "stage":    "welcome",
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # INTAKE NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="intake_node", run_type="chain")
# def intake_node(state: AgentState) -> dict:
#     """Collects user name, age, symptoms, and basic history via intake_agent."""
#     user_input = state.get("user_input", "")

#     task = Task(
#         description=f"""
#         Collect the following information from the user in a warm and clear manner:
#         1. Full name
#         2. Age
#         3. Gender
#         4. Primary symptoms and how long they have been experiencing them
#         5. Any known medical conditions or allergies

#         If the user has already provided some information below, extract it
#         and ask only for what is missing.

#         User message: {user_input}
#         """,
#         agent=intake_agent,
#         expected_output=(
#             "A structured summary: name, age, gender, symptoms (as a clean sentence), "
#             "duration, known conditions, allergies."
#         ),
#     )

#     crew   = Crew(agents=[intake_agent], tasks=[task], verbose=True)
#     result = crew.kickoff()
#     output = str(result)

#     user_info = {
#         "raw_intake": output,
#         "user_input": user_input,
#     }

#     messages = state.get("messages", [])
#     messages.append(f"[Intake] {output}")

#     return {
#         "user_info": user_info,
#         "symptoms":  user_input,
#         "messages":  messages,
#         "stage":     "intake",
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # CRITICALITY NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="criticality_node", run_type="chain")
# def criticality_node(state: AgentState) -> dict:
#     """Triages the user's symptoms and assigns a severity level."""
#     symptoms = state.get("symptoms", state.get("user_input", ""))

#     criticality = detect_criticality(symptoms)
#     print(f"\n[Criticality] Assessed as: {criticality.upper()}")

#     task = Task(
#         description=f"""
#         The patient reports these symptoms: {symptoms}

#         Severity has been assessed as: {criticality.upper()}

#         Your job:
#         - If CRITICAL: immediately provide first aid steps using the first_aid_guide tool,
#           tell the user to call 108 or go to the nearest emergency room without delay.
#         - If MODERATE: reassure the user and explain they need to see a doctor soon.
#         - If MILD: reassure the user that their condition can likely be managed with
#           medicine or home remedies.
#         """,
#         agent=criticality_agent,
#         expected_output="A clear, empathetic triage message appropriate for the severity level.",
#     )

#     crew   = Crew(agents=[criticality_agent], tasks=[task], verbose=True)
#     result = crew.kickoff()
#     output = str(result)

#     messages = state.get("messages", [])
#     messages.append(f"[Criticality: {criticality}] {output}")

#     return {
#         "criticality": criticality,
#         "agent_output": output,
#         "messages":    messages,
#         "stage":       "criticality",
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # INTENT NODE  (only reached when NOT critical)
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="intent_node", run_type="chain")
# def intent_node(state: AgentState) -> dict:
#     """Asks the user what kind of help they need and classifies their intent."""
#     user_input = state.get("user_input", "")

#     PROMPT = (
#         "\nHow can I help you further?\n"
#         "  1. Medicine suggestions\n"
#         "  2. Lab test suggestions\n"
#         "  3. Find a nearby doctor\n"
#         "  4. First aid / home remedies\n"
#         "  Type your choice or just describe what you need."
#     )
#     print(PROMPT)

#     intent = detect_intent(user_input)
#     print(f"[Intent] Detected: {intent}")

#     messages = state.get("messages", [])
#     messages.append(f"[Intent detected: {intent}]")

#     return {
#         "intent":   intent,
#         "messages": messages,
#         "stage":    "intent",
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # DOCTOR NODE  (triggered when criticality = critical)
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="doctor_node", run_type="chain")
# def doctor_node(state: AgentState) -> dict:
#     """Finds nearby doctors — triggered for critical or doctor-intent cases."""
#     symptoms   = state.get("symptoms", state.get("user_input", ""))
#     user_state = state.get("user_state", "")
#     user_city  = state.get("user_city", "")

#     if not user_state:
#         ask = (
#             "\nTo find a nearby doctor, I need your location.\n"
#             "Please tell me your state and city (e.g. Maharashtra, Mumbai)."
#         )
#         print(ask)
#         messages = state.get("messages", [])
#         messages.append("[Doctor] Waiting for location from user.")
#         return {
#             "messages": messages,
#             "stage":    "doctor_waiting_location",
#             "agent_output": ask,
#         }

#     task = Task(
#         description=f"""
#         The patient has reported these symptoms: {symptoms}
#         Their severity is: {state.get('criticality', 'critical').upper()}
#         Their location: {user_city}, {user_state}

#         Use the doctor_finder tool to find suitable nearby doctors.
#         If severity is critical, set emergency=True in the tool call.
#         Present results clearly so the patient can act immediately.
#         """,
#         agent=doctor_agent,
#         expected_output="A clear list of nearby doctors with contact details and availability.",
#     )

#     crew   = Crew(agents=[doctor_agent], tasks=[task], verbose=True)
#     result = crew.kickoff()
#     output = str(result)

#     messages = state.get("messages", [])
#     messages.append(f"[Doctor Finder] {output}")

#     return {
#         "agent_output": output,
#         "messages":     messages,
#         "stage":        "doctor",
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # MEDICINE NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="medicine_node", run_type="chain")
# def medicine_node(state: AgentState) -> dict:
#     """Provides medicine recommendations using RAG."""
#     symptoms  = state.get("symptoms", state.get("user_input", ""))
#     user_info = state.get("user_info", {})
#     age       = user_info.get("age", -1)

#     task = Task(
#         description=f"""
#         The patient reports: {symptoms}
#         Patient age: {age if age != -1 else 'unknown'}

#         Use the medicine_finder tool to find appropriate medicines.
#         Make sure to use the correct dosage for the patient's age group.
#         Always include side effects and at least one important warning.
#         Never recommend medicines without using the tool.
#         """,
#         agent=medicine_agent,
#         expected_output=(
#             "A clear list of medicines with name, dosage, duration, side effects, and warnings."
#         ),
#     )

#     crew   = Crew(agents=[medicine_agent], tasks=[task], verbose=True)
#     result = crew.kickoff()
#     output = str(result)

#     messages = state.get("messages", [])
#     messages.append(f"[Medicine] {output}")

#     return {
#         "agent_output": output,
#         "messages":     messages,
#         "stage":        "medicine",
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # LAB NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="lab_node", run_type="chain")
# def lab_node(state: AgentState) -> dict:
#     """Suggests lab tests using RAG."""
#     symptoms = state.get("symptoms", state.get("user_input", ""))

#     task = Task(
#         description=f"""
#         The patient reports: {symptoms}

#         Use the lab_test_suggester tool to recommend appropriate diagnostic tests.
#         Explain why each test is relevant to the patient's symptoms.
#         Remind the patient that a doctor must order and interpret the tests.
#         Never suggest tests without using the tool.
#         """,
#         agent=lab_agent,
#         expected_output="A clear list of recommended lab tests with reasons for each.",
#     )

#     crew   = Crew(agents=[lab_agent], tasks=[task], verbose=True)
#     result = crew.kickoff()
#     output = str(result)

#     messages = state.get("messages", [])
#     messages.append(f"[Lab Tests] {output}")

#     return {
#         "agent_output": output,
#         "messages":     messages,
#         "stage":        "lab",
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # HITL NODE  (human-in-the-loop review)
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="hitl_node", run_type="chain")
# def hitl_node(state: AgentState) -> dict:
#     """
#     Interrupts the graph for a human reviewer to approve or revise agent output.
#     In production this integrates with a Streamlit UI review panel.
#     In CLI mode it prompts the terminal directly.
#     """
#     agent_output = state.get("agent_output", "")

#     print("\n" + "="*60)
#     print("  HUMAN REVIEW REQUIRED")
#     print("="*60)
#     print("\nAgent output to review:\n")
#     print(agent_output)
#     print("\n" + "─"*60)
#     print("Options:")
#     print("  [A] Approve — send this to user")
#     print("  [R] Revise  — provide feedback for revision")
#     print("─"*60)

#     decision = input("\nYour decision (A/R): ").strip().upper()

#     if decision == "A":
#         messages = state.get("messages", [])
#         messages.append("[HITL] Approved by human reviewer.")
#         return {
#             "hitl_approved":  True,
#             "hitl_feedback":  "",
#             "messages":       messages,
#             "stage":          "hitl_approved",
#         }

#     feedback = input("Enter revision feedback: ").strip()
#     messages = state.get("messages", [])
#     messages.append(f"[HITL] Revision requested: {feedback}")
#     return {
#         "hitl_approved": False,
#         "hitl_feedback": feedback,
#         "messages":      messages,
#         "stage":         "hitl_revise",
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # RESPONSE NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="response_node", run_type="chain")
# def response_node(state: AgentState) -> dict:
#     """Formats and delivers the final verified response to the user."""
#     agent_output = state.get("agent_output", "")
#     criticality  = state.get("criticality", "mild")

#     header = {
#         "critical": "URGENT — Please act on this information immediately.",
#         "moderate": "Please review the following and consult a doctor soon.",
#         "mild":     "Here is the information based on your query.",
#     }.get(criticality, "")

#     final = f"{header}\n\n{agent_output}\n\nIf you have more questions, feel free to ask."

#     print("\n" + "="*60)
#     print(final)
#     print("="*60 + "\n")

#     messages = state.get("messages", [])
#     messages.append(f"[Response] {final}")

#     return {
#         "final_response": final,
#         "messages":       messages,
#         "stage":          "done",
#     }




# """
# src/graph/nodes.py
# ===================
# Every LangGraph node as a standalone function.

# KEY DESIGN CHANGE:
#   Nodes do NOT call input() or prompt the user.
#   All user interaction happens in main.py (CLI) or app.py (Streamlit).
#   Nodes receive data via state and return processed results.

#   The conversation flow is:
#     main.py collects user input → passes to graph via state
#     graph node processes it → returns output via state
#     main.py reads output and shows it to user → collects next input
# """

# import re
# import sys
# from pathlib import Path

# from crewai import Crew, Task
# from langsmith import traceable

# sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# from src.graph.state   import AgentState
# from src.graph.router  import detect_criticality, detect_intent
# from src.hitl.review_node import hitl_node          # noqa: F401 — re-exported for graph.py
# from src.agents.agents import (
#     intake_agent,
#     criticality_agent,
#     medicine_agent,
#     doctor_agent,
#     lab_agent,
# )
# from src.logger import get_logger

# logger = get_logger(__name__)


# def _clean(text: str) -> str:
#     """Strip <think>...</think> blocks that DeepSeek-R1 adds."""
#     cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
#     return cleaned.strip()


# WELCOME_MESSAGE = """\
# ============================================================
#   Welcome to the Agentic Doctor System!
# ============================================================

# I can help you with:
#   1. Criticality check — how serious are your symptoms?
#   2. Medicine suggestions
#   3. Lab test recommendations
#   4. Finding nearby doctors and hospitals
#   5. First aid and Indian home remedies

# Let's start by collecting some basic information.
# ============================================================"""


# # ─────────────────────────────────────────────────────────────────────────────
# # WELCOME NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="welcome_node", run_type="chain")
# def welcome_node(state: AgentState) -> dict:
#     """Returns the welcome message — main.py prints it."""
#     logger.info("welcome_node executed")
#     return {
#         "messages": [WELCOME_MESSAGE],
#         "stage":    "welcome",
#         "agent_output": WELCOME_MESSAGE,
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # INTAKE NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="intake_node", run_type="chain")
# def intake_node(state: AgentState) -> dict:
#     """
#     Receives user_input (name + age + symptoms collected by main.py)
#     and uses intake_agent to parse and structure it.
#     """
#     user_input = state.get("user_input", "")
#     logger.info("intake_node — processing: '%s'", user_input[:80])

#     task = Task(
#         description=f"""
#         The user has provided this information:
#         {user_input}

#         Extract and structure:
#         - Name (if provided)
#         - Age (if provided)
#         - Gender (if provided)
#         - Primary symptoms and duration
#         - Known conditions or allergies (if mentioned)

#         If some fields are missing, note them as 'not provided'.
#         Return a clean structured summary.
#         """,
#         agent=intake_agent,
#         expected_output=(
#             "Structured summary: Name, Age, Gender, Symptoms, Duration, "
#             "Known Conditions, Allergies."
#         ),
#     )

#     crew   = Crew(agents=[intake_agent], tasks=[task], verbose=False)
#     result = crew.kickoff()
#     output = _clean(str(result))

#     logger.info("intake_node complete")

#     messages = list(state.get("messages", []))
#     messages.append(f"[Intake]\n{output}")

#     return {
#         "user_info":    {"raw_intake": output, "user_input": user_input},
#         "symptoms":     user_input,
#         "agent_output": output,
#         "messages":     messages,
#         "stage":        "intake",
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # CRITICALITY NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="criticality_node", run_type="chain")
# def criticality_node(state: AgentState) -> dict:
#     """
#     Triages symptoms. Returns criticality level + triage message.
#     main.py reads agent_output and shows it to the user.
#     """
#     symptoms = state.get("symptoms", state.get("user_input", ""))
#     logger.info("criticality_node — symptoms: '%s'", symptoms[:80])

#     criticality = detect_criticality(symptoms)
#     logger.info("criticality_node — result: %s", criticality.upper())

#     task = Task(
#         description=f"""
#         Patient symptoms: {symptoms}
#         Assessed severity: {criticality.upper()}

#         Write a clear, empathetic triage message:
#         - CRITICAL: Tell the patient this is an emergency.
#           Use the first_aid_guide tool for immediate steps.
#           Tell them to call 108 or go to emergency room NOW.
#         - MODERATE: Reassure them but advise seeing a doctor soon.
#         - MILD: Reassure them it can likely be managed at home.

#         Keep your response concise and actionable.
#         """,
#         agent=criticality_agent,
#         expected_output="A clear empathetic triage message appropriate for the severity.",
#     )

#     crew   = Crew(agents=[criticality_agent], tasks=[task], verbose=False)
#     result = crew.kickoff()
#     output = _clean(str(result))

#     logger.info("criticality_node complete — %s", criticality)

#     messages = list(state.get("messages", []))
#     messages.append(f"[Criticality: {criticality.upper()}]\n{output}")

#     return {
#         "criticality":  criticality.lower(),
#         "agent_output": output,
#         "messages":     messages,
#         "stage":        "criticality",
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # INTENT NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="intent_node", run_type="chain")
# def intent_node(state: AgentState) -> dict:
#     """
#     Classifies the user's intent from their latest user_input.
#     main.py has already asked "what help do you need?" and stored the answer.
#     """
#     user_input = state.get("user_input", "")
#     logger.info("intent_node — classifying: '%s'", user_input[:80])

#     intent = detect_intent(user_input)
#     logger.info("intent_node — result: %s", intent)

#     messages = list(state.get("messages", []))
#     messages.append(f"[Intent: {intent}]")

#     return {
#         "intent":   intent,
#         "messages": messages,
#         "stage":    "intent",
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # DOCTOR NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="doctor_node", run_type="chain")
# def doctor_node(state: AgentState) -> dict:
#     """Finds nearby doctors using doctor_agent + doctor_finder tool."""
#     symptoms   = state.get("symptoms", state.get("user_input", ""))
#     user_state = state.get("user_state", "")
#     user_city  = state.get("user_city", "")
#     criticality = state.get("criticality", "moderate")

#     logger.info(
#         "doctor_node — symptoms: '%s' | location: %s, %s",
#         symptoms[:60], user_city, user_state
#     )

#     if not user_state:
#         # Signal main.py to ask for location
#         logger.info("doctor_node — location missing, requesting from user")
#         return {
#             "stage":        "doctor_need_location",
#             "agent_output": "NEED_LOCATION",
#         }

#     task = Task(
#         description=f"""
#         Patient symptoms: {symptoms}
#         Severity: {criticality.upper()}
#         Location: {user_city}, {user_state}

#         Use the doctor_finder tool to find nearby doctors.
#         Set emergency=True if severity is critical.
#         Present results clearly with name, hospital, phone, and availability.
#         """,
#         agent=doctor_agent,
#         expected_output="A clear list of nearby doctors with contact and availability details.",
#     )

#     crew   = Crew(agents=[doctor_agent], tasks=[task], verbose=False)
#     result = crew.kickoff()
#     output = _clean(str(result))

#     logger.info("doctor_node complete")

#     messages = list(state.get("messages", []))
#     messages.append(f"[Doctor Finder]\n{output}")

#     return {
#         "agent_output": output,
#         "messages":     messages,
#         "stage":        "doctor",
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # MEDICINE NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="medicine_node", run_type="chain")
# def medicine_node(state: AgentState) -> dict:
#     """Recommends medicines using medicine_agent + medicine_finder tool."""
#     raw = state.get("symptoms", "")

#     # Extract only symptom line (basic clean)
#     symptoms = raw.split("Symptoms:")[-1].strip()
#     user_info = state.get("user_info", {})
#     # age       = user_info.get("age", "unknown")
#     age = state.get("user_age")

#     logger.info("medicine_node — symptoms: '%s' | age: %s", symptoms[:60], age)

#     task = Task(
#         description=f"""
#         Patient symptoms: {symptoms}
#         Patient age: {age}
#         IMPORTANT:
#         When calling the medicine_finder tool, you MUST pass arguments EXACTLY like this:
#         {{
#             "symptoms": "{symptoms}",
#             "disease": "{symptoms}",
#             "patient_age": {age if age else 12},
#             "top_k": 2
#         }}

#         Do NOT send schema, properties, or metadata.
#         Only send valid JSON input.
#         """,
#         agent=medicine_agent,
#         expected_output="A clear list of medicines with dosage, side effects, and warnings.",
#     )

#     crew   = Crew(agents=[medicine_agent], tasks=[task], verbose=False)
#     result = crew.kickoff()
#     output = _clean(str(result))

#     logger.info("medicine_node complete")

#     messages = list(state.get("messages", []))
#     messages.append(f"[Medicine]\n{output}")

#     return {
#         "agent_output": output,
#         "messages":     messages,
#         "stage":        "medicine",
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # LAB NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="lab_node", run_type="chain")
# def lab_node(state: AgentState) -> dict:
#     """Suggests lab tests using lab_agent + lab_test_suggester tool."""
#     symptoms = state.get("symptoms", state.get("user_input", ""))
#     logger.info("lab_node — symptoms: '%s'", symptoms[:60])

#     task = Task(
#         description=f"""
#         Patient symptoms: {symptoms}

#         Use the lab_test_suggester tool to recommend diagnostic tests.
#         Explain why each test is relevant to the symptoms.
#         Remind the patient a doctor must order and interpret the tests.
#         Never suggest tests without using the tool.
#         """,
#         agent=lab_agent,
#         expected_output="A clear list of recommended lab tests with reasons for each.",
#     )

#     crew   = Crew(agents=[lab_agent], tasks=[task], verbose=False)
#     result = crew.kickoff()
#     output = _clean(str(result))

#     logger.info("lab_node complete")

#     messages = list(state.get("messages", []))
#     messages.append(f"[Lab Tests]\n{output}")

#     return {
#         "agent_output": output,
#         "messages":     messages,
#         "stage":        "lab",
#     }


# # hitl_node imported from src.hitl.review_node and re-exported via import above


# # ─────────────────────────────────────────────────────────────────────────────
# # RESPONSE NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="response_node", run_type="chain")
# def response_node(state: AgentState) -> dict:
#     """Formats the final verified response. main.py prints it."""
#     agent_output = state.get("agent_output", "")
#     criticality  = state.get("criticality", "mild")

#     header = {
#         "critical": "⚠️  URGENT — Please act on this information immediately.",
#         "moderate": "Please review the following and consult a doctor soon.",
#         "mild":     "Here is the information based on your query.",
#     }.get(criticality, "")

#     final = f"{header}\n\n{agent_output}"
#     logger.info("response_node complete — criticality: %s", criticality)

#     messages = list(state.get("messages", []))
#     messages.append(f"[Final Response]\n{final}")

#     return {
#         "final_response": final,
#         "messages":       messages,
#         "stage":          "done",
#     }



"""
src/graph/nodes.py
===================
Every LangGraph node. Nodes process state — they do NOT call input().
All user prompting happens in test.py / main.py.
"""

import re
import sys
from pathlib import Path

from crewai import Crew, Task
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
from src.logger import get_logger

logger = get_logger(__name__)


def _clean(text: str) -> str:
    """Strip <think>...</think> blocks that DeepSeek / Ollama models add."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _extract_symptoms(raw: str) -> str:
    """
    Extracts only the symptom text from the intake string.
    Input:  'Name: Utsav Raj\\nAge: 5\\nGender: Male\\nSymptoms: Cold from yesterday'
    Output: 'Cold from yesterday'
    """
    # Try to find 'Symptoms:' label first
    match = re.search(r"[Ss]ymptoms?\s*:\s*(.+?)(?:\n|$)", raw)
    if match:
        return match.group(1).strip()
    # Fallback: return the raw string trimmed
    return raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# WELCOME NODE
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="welcome_node", run_type="chain")
def welcome_node(state: AgentState) -> dict:
    logger.info("welcome_node executed")
    return {
        "messages": [],
        "stage":    "welcome",
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTAKE NODE
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="intake_node", run_type="chain")
def intake_node(state: AgentState) -> dict:
    user_input = state.get("user_input", "")
    logger.info("intake_node — processing: '%s'", user_input[:80])

    task = Task(
        description=f"""
        The user has provided this information:
        {user_input}

        Extract and structure:
        - Name (if provided)
        - Age (if provided)
        - Gender (if provided)
        - Primary symptoms and duration
        - Known conditions or allergies (if mentioned)

        If some fields are missing, note them as 'not provided'.
        Return a clean structured summary.
        """,
        agent=intake_agent,
        expected_output=(
            "Structured summary: Name, Age, Gender, Symptoms, Duration, "
            "Known Conditions, Allergies."
        ),
    )

    crew   = Crew(agents=[intake_agent], tasks=[task], verbose=False)
    result = crew.kickoff()
    output = _clean(str(result))

    # Extract only the symptom text — not the full intake string
    symptoms_only = _extract_symptoms(user_input)

    logger.info("intake_node complete — symptoms extracted: '%s'", symptoms_only[:80])

    messages = list(state.get("messages", []))
    messages.append(f"[Intake]\n{output}")

    return {
        "user_info":    {"raw_intake": output, "user_input": user_input},
        "symptoms":     symptoms_only,          # ← clean symptom string only
        "agent_output": output,
        "messages":     messages,
        "stage":        "intake",
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

        Write a clear, empathetic triage message:
        - CRITICAL: Tell the patient this is an emergency.
          Use the first_aid_guide tool for immediate steps.
          Tell them to call 108 or go to emergency room NOW.
        - MODERATE: Reassure them but advise seeing a doctor soon.
        - MILD: Reassure them it can likely be managed at home.

        Keep your response concise and actionable.
        """,
        agent=criticality_agent,
        expected_output="A clear empathetic triage message appropriate for the severity.",
    )

    crew   = Crew(agents=[criticality_agent], tasks=[task], verbose=False)
    result = crew.kickoff()
    output = _clean(str(result))

    logger.info("criticality_node complete — %s", criticality)

    messages = list(state.get("messages", []))
    messages.append(f"[Criticality: {criticality.upper()}]\n{output}")

    return {
        "criticality":  criticality.lower(),
        "agent_output": output,
        "messages":     messages,
        "stage":        "criticality",
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTENT NODE
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="intent_node", run_type="chain")
def intent_node(state: AgentState) -> dict:
    """
    Classifies user's choice (medicine / lab / doctor).
    user_input here is the user's second input — their choice after criticality.
    NOT the original intake text.
    """
    user_input = state.get("user_input", "")
    logger.info("intent_node — classifying: '%s'", user_input[:80])

    intent = detect_intent(user_input)
    logger.info("intent_node — result: %s", intent)

    messages = list(state.get("messages", []))
    messages.append(f"[Intent: {intent}]")

    return {
        "intent":   intent,
        "messages": messages,
        "stage":    "intent",
    }


# ─────────────────────────────────────────────────────────────────────────────
# DOCTOR NODE
# ─────────────────────────────────────────────────────────────────────────────
from src.tools.doctor_tool import doctor_finder
@traceable(name="doctor_node", run_type="chain")
def doctor_node(state: AgentState) -> dict:
    symptoms    = state.get("symptoms", "")
    user_state  = state.get("user_state", "")
    user_city   = state.get("user_city", "")
    criticality = state.get("criticality", "moderate")
    department = ""
    if "chest pain" in symptoms.lower():
        department = "Cardiology"
    elif "fever" in symptoms.lower():
        department = "General Physician"
    elif "cancer" in symptoms.lower():
        department = "Oncology"
    elif "skin" in symptoms.lower():
        department = "Dermatology"
    elif "child" in symptoms.lower() or "kid" in symptoms.lower():
        department = "Pediatrics"
    elif "pregnant" in symptoms.lower() or "pregnancy" in symptoms.lower():
        department = "Gynecology"
    
    emergency = "False"
    if criticality == "critical":
        emergency = "True"
    
    logger.info(
        "doctor_node — symptoms: '%s' | location: %s, %s",
        symptoms[:60], user_city, user_state,
    )

    if not user_state:
        logger.warning("doctor_node — user_state missing")
        return {
            "stage":        "doctor_need_location",
            "agent_output": "NEED_LOCATION",
        }

    # task = Task(
    #     description=f"""
    #     symptoms: {symptoms}
    #     state: {user_state}
    #     # city: {user_city}
    #     # department: {department}
    #     # emergency: {emergency}
    #     # top_k: 2
        
    #     Use the doctor_finder tool to find nearby doctors.
    #     Set emergency=True if severity is critical.
    #     Present results clearly with name, hospital, phone, and availability.
    #     """,
    #     agent=doctor_agent,
    #     expected_output="A clear list of nearby doctors with contact and availability details.",
    # )

    # crew   = Crew(agents=[doctor_agent], tasks=[task], verbose=False)
    # result = crew.kickoff()
    # output = _clean(str(result))

    # logger.info("doctor_node complete")

    # messages = list(state.get("messages", []))
    # messages.append(f"[Doctor Finder]\n{output}")

    # return {
    #     "agent_output": output,
    #     "messages":     messages,
    #     "stage":        "doctor",
    # }
    
    output = doctor_finder.run(**{
        "symptoms":   symptoms,
        "state":      user_state,
        "city":       user_city,
        "department": department,
        "emergency":  emergency,
        "top_k":      2,
    })
 
    logger.info("doctor_node complete")
 
    messages = list(state.get("messages", []))
    messages.append(f"[Doctor Finder]\n{output}")
 
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
    symptoms = state.get("symptoms", "")
    age      = state.get("user_age", None)

    logger.info("medicine_node — symptoms: '%s' | age: %s", symptoms[:60], age)

    age_val = int(age) if age and str(age).isdigit() else 18

    task = Task(
        description=f"""
        Patient symptoms: {symptoms}
        Patient age: {age_val}

        Use the medicine_finder tool with these exact arguments:
        {{
            "symptoms": "{symptoms}",
            "disease": "{symptoms}",
            "patient_age": {age_val},
            "top_k": 2
        }}

        Do NOT pass schema or metadata. Only valid JSON.
        Always include dosage appropriate for age {age_val},
        side effects, and at least one warning.
        """,
        agent=medicine_agent,
        expected_output="A clear list of medicines with dosage, side effects, and warnings.",
    )

    crew   = Crew(agents=[medicine_agent], tasks=[task], verbose=False)
    result = crew.kickoff()
    output = _clean(str(result))

    logger.info("medicine_node complete")

    messages = list(state.get("messages", []))
    messages.append(f"[Medicine]\n{output}")

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
    symptoms = state.get("symptoms", "")
    logger.info("lab_node — symptoms: '%s'", symptoms[:60])

    task = Task(
        description=f"""
        Patient symptoms: {symptoms}

        Use the lab_test_suggester tool to recommend diagnostic tests.
        Explain why each test is relevant to the symptoms.
        Remind the patient a doctor must order and interpret the tests.
        Never suggest tests without using the tool.
        """,
        agent=lab_agent,
        expected_output="A clear list of recommended lab tests with reasons for each.",
    )

    crew   = Crew(agents=[lab_agent], tasks=[task], verbose=False)
    result = crew.kickoff()
    output = _clean(str(result))

    logger.info("lab_node complete")

    messages = list(state.get("messages", []))
    messages.append(f"[Lab Tests]\n{output}")

    return {
        "agent_output": output,
        "messages":     messages,
        "stage":        "lab",
    }


# ─────────────────────────────────────────────────────────────────────────────
# HITL NODE  (simple inline — no LangGraph interrupt needed for CLI testing)
# ─────────────────────────────────────────────────────────────────────────────

def hitl_node(state: AgentState) -> dict:
    """Simple HITL for CLI — prompts reviewer directly."""
    agent_output = state.get("agent_output", "")

    print("\n" + "="*60)
    print("  HUMAN REVIEW REQUIRED")
    print("="*60)
    print(agent_output)
    print("─"*60)
    print("  [A] Approve   [R] Revise")
    print("─"*60)

    decision = input("\nYour decision (A/R): ").strip().upper()

    if decision == "A":
        messages = list(state.get("messages", []))
        messages.append("[HITL] Approved.")
        return {
            "hitl_approved": True,
            "hitl_feedback": "",
            "messages":      messages,
            "stage":         "hitl_approved",
        }

    feedback = input("Feedback for revision: ").strip()
    messages = list(state.get("messages", []))
    messages.append(f"[HITL] Revision: {feedback}")
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
    agent_output = state.get("agent_output", "")
    criticality  = state.get("criticality", "mild")

    header = {
        "critical": "URGENT — Please act immediately.",
        "moderate": "Please consult a doctor soon.",
        "mild":     "Here is the information based on your query.",
    }.get(criticality, "")

    final = f"{header}\n\n{agent_output}"
    logger.info("response_node complete")

    messages = list(state.get("messages", []))
    messages.append(f"[Response]\n{final}")

    return {
        "final_response": final,
        "messages":       messages,
        "stage":          "done",
    }