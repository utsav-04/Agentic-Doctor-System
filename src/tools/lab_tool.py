"""
src/tools/lab_tool.py  —  async @tool backed by real RAG data
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

from crewai.tools import tool
from langsmith import traceable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.retriever import LabTestRetriever
from src.schema.schemas import LabToolInput, LabToolOutput, RetrievedLabTest

_retriever: Optional[LabTestRetriever] = None


def _get_retriever() -> LabTestRetriever:
    global _retriever
    if _retriever is None:
        _retriever = LabTestRetriever()
    return _retriever


@traceable(name="retrieve_lab_tests", run_type="retriever")
async def _retrieve_lab_tests_async(
    symptoms: str,
    severity: Optional[str],
    top_k: int,
) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: _get_retriever().get_lab_tests(
            symptoms=symptoms,
            severity=severity,
            top_k=top_k,
        ),
    )


@tool("lab_test_suggester")
def lab_test_suggester(
    symptoms: str,
    severity: str = "",
    top_k: int = 3,
) -> str:
    """
    Suggests appropriate lab tests for given symptoms using RAG retrieval.
    Use when a user asks what blood tests or diagnostic tests they should get done.

    Args:
        symptoms: User's symptoms in plain language
                  e.g. 'fever from 2 days, body pain, headache and weakness'
        severity: Optional severity filter — 'low', 'medium', 'high', or 'critical'.
                  Leave empty to search all severity levels.
        top_k:    Number of results to return. Default is 3.

    Returns:
        Formatted list of recommended lab tests with reasons,
        possible conditions, and specialist referral.
    """
    try:
        validated = LabToolInput(
            symptoms=symptoms,
            severity=severity if severity else None,
            top_k=top_k,
        )
    except Exception as e:
        return f"Invalid input: {e}. Please describe your symptoms clearly."

    raw_result = asyncio.run(
        _retrieve_lab_tests_async(
            symptoms=validated.symptoms,
            severity=validated.severity,
            top_k=validated.top_k,
        )
    )

    raw_docs = raw_result.get("documents", [])
    if not raw_docs:
        return (
            f"No lab test recommendations found for: '{symptoms}'. "
            "Please consult a doctor for appropriate diagnostic tests."
        )

    lab_results = []
    for doc in raw_docs:
        try:
            lab_results.append(RetrievedLabTest.from_document(doc))
        except Exception:
            continue

    if not lab_results:
        return "Could not process lab test results. Please consult a doctor."

    output = LabToolOutput(
        query=symptoms,
        total_results=len(lab_results),
        lab_results=lab_results,
        retrieval_metadata={
            "stage_counts":    raw_result.get("stage_counts", {}),
            "latency_seconds": raw_result.get("retrieval_latency_seconds", 0),
            "filters_applied": raw_result.get("filters_applied", {}),
        },
    )

    return _format_output(output)


def _format_output(output: LabToolOutput) -> str:
    lines = [f"Lab Test Recommendations for: '{output.query}'", ""]
    seen: set = set()

    for i, result in enumerate(output.lab_results, 1):
        conditions = ", ".join(result.possible_conditions)
        lines += [
            f"Match {i}  |  Severity: {result.severity.upper()}"
            f"  |  Specialist: {result.specialist_referral}",
            f"Possible conditions: {conditions}",
            "",
            "Recommended Tests:",
        ]
        for test in result.recommended_tests:
            name   = test.get("test_name", "")
            reason = test.get("reason", "")
            if name and name not in seen:
                lines.append(f"  - {name}")
                if reason:
                    lines.append(f"    Why: {reason}")
                seen.add(name)
        lines.append("")

    lines.append(f"DISCLAIMER: {output.disclaimer}")
    return "\n".join(lines)