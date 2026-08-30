"""
src/graph/router.py
====================
Criticality detection (unchanged) + plan detection (NEW — replaces detect_intent).

detect_plan() no longer returns a single label. It returns an ORDERED LIST
of tool names to call this turn, e.g.:
    ["medicine"]
    ["medicine", "lab"]
    ["doctor"]
    []                      # no medical tool applies — general/greeting message

This is what makes multi-tool execution possible: planner_node calls this once,
gets the full plan, and executor_node just walks the list and calls each tool.
No more hardcoded "both" special-case, no more silently dropping the second tool.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.config import LLMConfig
from src.logger import get_logger
from crewai import LLM

logger = get_logger(__name__)

VALID_TOOLS = {"medicine", "lab", "doctor", "first_aid"}

llm = LLM(
    model="ollama/llama3",
    base_url="http://localhost:11434",
    temperature=LLMConfig.temperature,
)
logger.info("Router LLM ready: %s", llm.model)


def detect_criticality(symptoms: str) -> str:
    """
    Classifies symptoms as critical / moderate / mild.
    Returns one of: 'critical' | 'moderate' | 'mild'
    Unchanged from the original — this part of the design was fine.
    """
    logger.info("Detecting criticality for symptoms: '%s'", symptoms[:80])
    prompt = f"""
You are a medical triage assistant in an Indian healthcare chatbot.

Classify the severity of the patient's symptoms into ONE of these levels:

critical  -> life-threatening symptoms needing IMMEDIATE emergency care.
             Examples: chest pain, stroke, unconsciousness, severe bleeding,
             difficulty breathing, snake bite, heart attack, seizure, high fever in infant.

moderate  -> symptoms that need a doctor soon but are not immediately life-threatening.
             Examples: persistent fever for 3+ days, severe joint pain,
             recurring vomiting, uncontrolled diabetes, UTI with fever.

mild      -> minor symptoms manageable at home with medicine or first aid.
             Examples: common cold, mild fever, indigestion, mild cough,
             minor cut, headache, loose motion without dehydration.

Patient symptoms:
{symptoms}

Return ONLY one word: critical, moderate, or mild.
"""
    response = llm.call(prompt)
    result = response.strip().lower()

    if result not in ("critical", "moderate", "mild"):
        logger.warning("Unexpected criticality value '%s' — defaulting to 'mild'", result)
        result = "mild"

    logger.info("Criticality result: %s", result.upper())
    return result


def detect_plan(user_input: str, symptoms: str = "") -> list:
    """
    Decides which tool(s) to call for this query, and in what order.

    Replaces the old detect_intent() single-label classifier.
    Multiple tools can be returned — e.g. a query that needs both medicine
    AND lab tests returns ["medicine", "lab"], and executor_node will
    actually call both (unlike the old "both" intent, which only ever
    reached the medicine node).

    Returns a list of zero or more values from VALID_TOOLS, in the order
    they should be called. Empty list means no medical tool applies
    (general question / greeting) — synthesis_node will answer directly.
    """
    logger.info("Detecting plan for: '%s'", user_input[:80])

    prompt = f"""
You are a planning assistant in a medical chatbot. Decide which tools are
needed to answer the user's message, and in what order to call them.

Available tools:
  medicine   -> user wants medicine suggestions, drug info, or dosage advice
  lab        -> user wants lab test / diagnostic test suggestions
  doctor     -> user wants to find a nearby doctor or hospital
  first_aid  -> user wants immediate first aid or home remedy guidance

Rules:
  - Return a JSON array of tool names, in the order they should be called.
  - Include a tool ONLY if the user's message actually needs it.
  - A query can need more than one tool (e.g. "what medicine should I take
    and what tests should I get done" -> ["medicine", "lab"]).
  - If the message is a greeting or general question with no medical
    need, return an empty array: []
  - Return ONLY the JSON array. No explanation, no markdown fences.

Patient symptoms (context): {symptoms}
User message: {user_input}

Examples:
"give me medicine for cold and cough" -> ["medicine"]
"what tests should I get for these symptoms and any medicine" -> ["medicine", "lab"]
"find me a doctor nearby" -> ["doctor"]
"hi" -> []
"""
    response = llm.call(prompt)
    plan = _parse_plan(response)

    logger.info("Plan detected: %s", plan)
    return plan


def _parse_plan(raw_response: str) -> list:
    """
    Parses the LLM's response into a validated list of tool names.
    Falls back to ["medicine"] if parsing fails outright, since that's
    the single most common ask — safer than silently returning nothing.
    """
    text = raw_response.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()

    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        text = match.group(0)

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Could not parse plan JSON from: '%s' — defaulting to ['medicine']", raw_response[:120])
        return ["medicine"]

    if not isinstance(parsed, list):
        logger.warning("Plan response was not a list: %r — defaulting to ['medicine']", parsed)
        return ["medicine"]

    seen = set()
    plan = []
    for item in parsed:
        name = str(item).strip().lower()
        if name in VALID_TOOLS and name not in seen:
            plan.append(name)
            seen.add(name)

    return plan