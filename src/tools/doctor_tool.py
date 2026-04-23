"""
src/tools/doctor_tool.py  —  async @tool
"""
import asyncio
import sys
from pathlib import Path
from typing import Optional

from crewai.tools import tool
from langsmith import traceable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.retriever import DoctorRetriever
from src.schema.schemas import DoctorToolInput, RetrievedDoctor

_retriever: Optional[DoctorRetriever] = None


def _get_retriever() -> DoctorRetriever:
    global _retriever
    if _retriever is None:
        _retriever = DoctorRetriever()
    return _retriever


@traceable(name="retrieve_doctors", run_type="retriever")
async def _retrieve_doctors_async(
    symptoms: str,
    state: str,
    city: Optional[str],
    department: Optional[str],
    emergency_only: bool,
    top_k: int,
) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: _get_retriever().get_doctors(
            symptoms=symptoms,
            state=state,
            city=city,
            department=department,
            emergency_only=emergency_only,
            top_k=top_k,
        ),
    )


@tool("doctor_finder")
def doctor_finder(
    symptoms: str,
    state: str,
    city: str = "",
    department: str = "",
    emergency: bool = False,
    top_k: int = 5,
) -> str:
    """
    Finds nearby doctors and hospitals based on symptoms and location.
    State is required. City and department improve accuracy.

    Args:
        symptoms:   User's symptoms e.g. 'chest pain and breathlessness'
        state:      Indian state name e.g. 'Maharashtra' — REQUIRED.
        city:       City name e.g. 'Mumbai'. Leave empty if not known.
        department: Medical department e.g. 'Cardiology'. Leave empty if unsure.
        emergency:  Set True if user needs emergency care immediately.
        top_k:      Number of doctors to return. Default is 5.

    Returns:
        Formatted list of doctors with hospital, contact, and availability details.
    """
    try:
        validated = DoctorToolInput(
            symptoms=symptoms,
            state=state,
            city=city if city else None,
            department=department if department else None,
            emergency=emergency,
            top_k=top_k,
        )
    except Exception as e:
        return f"Invalid input: {e}. Please provide a valid state name."

    raw_result = asyncio.run(
        _retrieve_doctors_async(
            symptoms=validated.symptoms,
            state=validated.state,
            city=validated.city,
            department=validated.department,
            emergency_only=validated.emergency,
            top_k=validated.top_k,
        )
    )

    raw_docs = raw_result.get("documents", [])
    if not raw_docs:
        loc = f"{validated.city}, " if validated.city else ""
        return (
            f"No doctors found in {loc}{validated.state} for: '{symptoms}'. "
            "Try a nearby city or broaden the department filter."
        )

    doctors = []
    for doc in raw_docs:
        try:
            doctors.append(RetrievedDoctor.from_document(doc))
        except Exception:
            continue

    if not doctors:
        return "Could not process doctor results. Please try again."

    loc = f"{validated.city}, " if validated.city else ""
    lines = [f"Found {len(doctors)} doctor(s) in {loc}{validated.state}:\n"]

    for i, doc in enumerate(doctors, 1):
        emergency_tag = " [EMERGENCY AVAILABLE]" if doc.accepts_emergency else ""
        online_tag    = f" | Online: {doc.online_platform}" if doc.is_available_online else ""
        lines += [
            f"{i}. {doc.name}{emergency_tag}",
            f"   Department    : {doc.department}",
            f"   Qualification : {doc.qualification}",
            f"   Experience    : {doc.experience_years} years",
            f"   Hospital      : {doc.hospital_name}",
            f"   Location      : {doc.city}, {doc.state}",
            f"   Phone         : {doc.phone}",
            f"   Fee           : Rs {doc.consultation_fee}",
            f"   Available     : {', '.join(doc.available_days[:3])}",
            f"   Timing        : {doc.timing}",
            f"   Mode          : {', '.join(doc.appointment_mode)}{online_tag}",
            "",
        ]

    return "\n".join(lines)
