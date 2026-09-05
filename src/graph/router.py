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
from src.cache.redis_cache import make_key, cache_get, cache_set
from src.logger import get_logger
from crewai import LLM

logger = get_logger(__name__)

VALID_TOOLS = {"medicine", "lab", "doctor", "first_aid"}

# Both classifications run at temperature=0.1 (near-deterministic), and
# results don't meaningfully change day to day, so a 24h TTL is safe.
LLM_CACHE_TTL = 60 * 60 * 24

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
    Checks Redis before calling the LLM; caches the result after.
    """
    cache_key = make_key("llm:criticality", symptoms=symptoms)
    cached = cache_get(cache_key)
    if cached:
        return cached

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
    cache_set(cache_key, result, ttl=LLM_CACHE_TTL)
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

    cache_key = make_key("llm:plan", user_input=user_input, symptoms=symptoms)
    cached = cache_get(cache_key)
    if cached:
        try:
            return json.loads(cached)
        except (json.JSONDecodeError, TypeError):
            pass  # fall through and recompute if the cached value is somehow bad

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
  - medicine + first_aid together is the normal default for a plain,
    short-duration symptom description (e.g. "cold, cough and fever",
    "mild headache since morning") — that's enough to help most people.
  - Only include "lab" when there's a SPECIFIC trigger for it: the user
    explicitly asks for tests, OR the symptoms suggest something beyond a
    routine cold/mild fever (e.g. fever lasting 5+ days, high fever,
    symptoms alongside body pain/chills suggesting dengue/malaria/typhoid,
    or the user says symptoms are "not improving" / "getting worse").
    Do NOT add lab "just in case" for an ordinary short-duration query —
    that adds tests most users didn't ask for and don't need yet.
  - Include "doctor" only if the user asks to find one, or if first_aid /
    criticality context already implies urgent referral.
  - A query can need more than one tool when it genuinely does
    (e.g. "what medicine should I take and what tests should I get done"
    -> ["medicine", "lab"] — here the user explicitly asked for tests).
  - If the message is a greeting or general question with no medical
    need, return an empty array: []
  - Return ONLY the JSON array. No explanation, no markdown fences.

Patient symptoms (context): {symptoms}
User message: {user_input}

Examples:
"cold, cough and fever" -> ["medicine", "first_aid"]
"give me medicine for cold and cough" -> ["medicine"]
"fever for 5 days with body pain and chills, not improving" -> ["medicine", "lab"]
"what tests should I get for these symptoms and any medicine" -> ["medicine", "lab"]
"find me a doctor nearby" -> ["doctor"]
"hi" -> []
"""
    response = llm.call(prompt)
    plan = _parse_plan(response)

    logger.info("Plan detected: %s", plan)
    cache_set(cache_key, json.dumps(plan), ttl=LLM_CACHE_TTL)
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