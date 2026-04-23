"""
src/graph/router.py
====================
Intent and criticality detection using Gemini LLM.
Mirrors the reference detect_intent() pattern exactly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.config import GOOGLE_API_KEY, LLMConfig
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model=LLMConfig.MODEL_NAME,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
)


def detect_criticality(symptoms: str) -> str:
    """
    Classifies symptoms as critical / moderate / mild.
    Returns one of: 'critical' | 'moderate' | 'mild'
    """
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
    response = llm.invoke(prompt)
    result   = response.content.strip().lower()

    if result not in ("critical", "moderate", "mild"):
        return "mild"
    return result


def detect_intent(user_input: str) -> str:
    """
    Detects what help the user wants after criticality is assessed as non-critical.
    Returns one of: 'medicine' | 'lab' | 'both' | 'doctor' | 'first_aid' | 'general'
    """
    prompt = f"""
You are an intent classification assistant in a medical chatbot.

Classify the user's request into ONE of the following intents:

medicine   -> user wants medicine suggestions, drug information, or dosage advice
lab        -> user wants lab test suggestions or diagnostic test advice
both       -> user wants both medicine AND lab test suggestions
doctor     -> user wants to find a nearby doctor or hospital
first_aid  -> user wants immediate first aid or home remedy guidance
general    -> general health question or conversation

User message:
{user_input}

Return ONLY the intent name. Example outputs:
medicine
lab
both
doctor
first_aid
general
"""
    response = llm.invoke(prompt)
    result   = response.content.strip().lower()

    valid = {"medicine", "lab", "both", "doctor", "first_aid", "general"}
    return result if result in valid else "general"
