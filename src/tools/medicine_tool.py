"""
src/tools/medicine_tool.py  —  async @tool
"""
import asyncio
import sys
from pathlib import Path
from typing import Optional

from crewai.tools import tool
from langsmith import traceable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.retriever import MedicineRetriever
from src.schema.schemas import MedicineToolInput, RetrievedMedicine

_retriever: Optional[MedicineRetriever] = None


def _get_retriever() -> MedicineRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MedicineRetriever()
    return _retriever


@traceable(name="retrieve_medicines", run_type="retriever")
async def _retrieve_medicines_async(symptoms: str, disease: Optional[str], top_k: int) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: _get_retriever().get_medicines(symptoms=symptoms, disease=disease, top_k=top_k),
    )


@tool("medicine_finder")
def medicine_finder(symptoms: str, disease: str = "", patient_age: int = -1, top_k: int = 5) -> str:
    """
    Finds medicines and treatment information for given symptoms or disease.
    Use when a user asks about medications, dosage, or treatment options.

    Args:
        symptoms:    User-described symptoms e.g. 'fever and dry cough'
        disease:     Optional disease name e.g. 'Diabetes Mellitus'. Leave empty if unknown.
        patient_age: Patient age in years. Use -1 if unknown or adult.
        top_k:       Number of medicines to return. Default is 5.

    Returns:
        Formatted medicine list with dosage, side effects, and warnings.
    """
    try:
        validated = MedicineToolInput(
            symptoms=symptoms,
            disease=disease if disease else None,
            patient_age=patient_age if patient_age >= 0 else None,
            top_k=top_k,
        )
    except Exception as e:
        return f"Invalid input: {e}. Please provide valid symptoms."

    raw_result = asyncio.run(
        _retrieve_medicines_async(
            symptoms=validated.symptoms,
            disease=validated.disease,
            top_k=validated.top_k,
        )
    )

    raw_docs = raw_result.get("documents", [])
    if not raw_docs:
        return f"No medicines found for: '{symptoms}'. Please consult a doctor."

    medicines = []
    for doc in raw_docs:
        try:
            medicines.append(RetrievedMedicine.from_document(doc))
        except Exception:
            continue

    if not medicines:
        return "Could not process results. Please consult a doctor."

    use_child = validated.patient_age is not None and validated.patient_age < 18
    lines = [f"Found {len(medicines)} medicine(s) for: '{symptoms}'\n"]

    for i, med in enumerate(medicines, 1):
        dose  = med.child_dosage if use_child else med.adult_dosage
        label = "Child dosage" if use_child else "Adult dosage"
        lines += [
            f"{i}. {med.name}",
            f"   Disease       : {med.disease}",
            f"   Form          : {med.form}",
            f"   {label:<14}: {dose}",
            f"   Duration      : {med.duration}",
            f"   Side effects  : {', '.join(med.side_effects[:3])}",
            f"   Warning       : {med.warnings[0] if med.warnings else 'None'}",
            "",
        ]

    lines.append(
        "DISCLAIMER: For reference only. "
        "Always consult a qualified doctor before taking any medication."
    )
    return "\n".join(lines)
