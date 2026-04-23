"""
src/tools/first_aid_tool.py  —  async @tool
"""
import asyncio
import sys
from pathlib import Path
from typing import Optional

from crewai.tools import tool
from langsmith import traceable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.retriever import FirstAidRetriever
from src.schema.schemas import FirstAidToolInput, RetrievedFirstAid

_retriever: Optional[FirstAidRetriever] = None


def _get_retriever() -> FirstAidRetriever:
    global _retriever
    if _retriever is None:
        _retriever = FirstAidRetriever()
    return _retriever


@traceable(name="retrieve_first_aid", run_type="retriever")
async def _retrieve_first_aid_async(condition: str, severity: Optional[str], top_k: int) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: _get_retriever().get_first_aid(
            condition=condition, severity=severity, top_k=top_k
        ),
    )


@tool("first_aid_guide")
def first_aid_guide(condition: str, severity: str = "", top_k: int = 2) -> str:
    """
    Provides first aid steps, Indian home remedies, and what NOT to do for a condition.
    Use for immediate care guidance before medical help arrives, or for mild home-manageable conditions.

    Args:
        condition: The condition or emergency e.g. 'heart attack', 'fever', 'minor burn'
        severity:  Optional filter: 'mild', 'urgent', or 'life_threatening'. Leave empty for all.
        top_k:     Number of guides to return. Default is 2.

    Returns:
        Formatted first aid steps, home remedies, do-not-do list, and doctor visit indicators.
    """
    try:
        validated = FirstAidToolInput(
            condition=condition,
            severity=severity if severity else None,
            top_k=top_k,
        )
    except Exception as e:
        return f"Invalid input: {e}."

    raw_result = asyncio.run(
        _retrieve_first_aid_async(
            condition=validated.condition,
            severity=validated.severity,
            top_k=validated.top_k,
        )
    )

    raw_docs = raw_result.get("documents", [])
    if not raw_docs:
        return (
            f"No first aid guide found for: '{condition}'. "
            "If this is an emergency, call 108 immediately."
        )

    guides = []
    for doc in raw_docs:
        try:
            guides.append(RetrievedFirstAid.from_document(doc))
        except Exception:
            continue

    if not guides:
        return "Could not process results. Call 108 for emergencies."

    lines = [
        "EMERGENCY NOTICE: For life-threatening emergencies, call 108 immediately.",
        "",
        f"First Aid Guide for: '{condition}'",
        "",
    ]

    for guide in guides:
        sev = guide.severity_level.replace("_", " ").upper()
        lines += [f"{'─'*50}", f"Condition : {guide.condition}", f"Severity  : {sev}", ""]

        if guide.immediate_steps:
            lines.append("IMMEDIATE STEPS:")
            for j, step in enumerate(guide.immediate_steps[:6], 1):
                lines.append(f"  {j}. {step}")
            lines.append("")

        if guide.home_remedies:
            lines.append("HOME REMEDIES (Indian household):")
            for r in guide.home_remedies[:4]:
                lines.append(f"  - {r}")
            lines.append("")

        if guide.do_not_do:
            lines.append("DO NOT:")
            for d in guide.do_not_do[:3]:
                lines.append(f"  - {d}")
            lines.append("")

        if guide.go_to_doctor_if:
            lines.append("GO TO DOCTOR IF:")
            for s in guide.go_to_doctor_if[:3]:
                lines.append(f"  - {s}")
            lines.append("")

    return "\n".join(lines)
