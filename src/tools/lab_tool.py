"""
src/tools/lab_tool.py  —  async @tool
"""
import asyncio
import sys
from pathlib import Path
from typing import Optional, List, Dict

from crewai.tools import tool
from langsmith import traceable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.retriever import MedicineRetriever
from src.schema.schemas import LabToolInput

_retriever: Optional[MedicineRetriever] = None


def _get_retriever() -> MedicineRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MedicineRetriever()
    return _retriever


_LAB_TEST_MAP: Dict[str, List[Dict[str, str]]] = {
    "diabetes": [
        {"test": "Fasting Blood Glucose",           "reason": "Baseline blood sugar level"},
        {"test": "HbA1c",                            "reason": "3-month average blood sugar control"},
        {"test": "Post-Prandial Blood Glucose",      "reason": "Blood sugar 2 hours after meals"},
        {"test": "Lipid Profile",                    "reason": "Diabetics have high cardiovascular risk"},
        {"test": "Urine Routine and Microscopy",     "reason": "Checks glucose and protein in urine"},
    ],
    "hypertension": [
        {"test": "24-Hour Ambulatory BP Monitoring", "reason": "Confirms hypertension pattern"},
        {"test": "Serum Creatinine and eGFR",        "reason": "Kidney function — BP affects kidneys"},
        {"test": "Serum Electrolytes",               "reason": "Sodium and potassium balance"},
        {"test": "ECG",                              "reason": "Heart strain from high blood pressure"},
        {"test": "Urine Albumin (Microalbuminuria)", "reason": "Early kidney damage marker"},
    ],
    "heart": [
        {"test": "ECG (Electrocardiogram)",          "reason": "Heart rhythm and electrical activity"},
        {"test": "Troponin I and T",                 "reason": "Heart muscle damage marker"},
        {"test": "2D Echocardiography",              "reason": "Heart structure and ejection fraction"},
        {"test": "Lipid Profile",                    "reason": "LDL, HDL, cholesterol, triglycerides"},
        {"test": "CK-MB",                            "reason": "Cardiac enzyme — confirms heart attack"},
    ],
    "liver": [
        {"test": "Liver Function Test (LFT)",        "reason": "ALT, AST, ALP, bilirubin, albumin"},
        {"test": "PT/INR",                           "reason": "Clotting ability produced by liver"},
        {"test": "Hepatitis B Surface Antigen",      "reason": "Hepatitis B screening"},
        {"test": "Hepatitis C Antibody",             "reason": "Hepatitis C screening"},
        {"test": "Ultrasound Abdomen",               "reason": "Liver size and fatty changes"},
    ],
    "kidney": [
        {"test": "Serum Creatinine",                 "reason": "Primary kidney function marker"},
        {"test": "eGFR",                             "reason": "Kidney filtration rate estimate"},
        {"test": "Blood Urea Nitrogen (BUN)",        "reason": "Kidney waste clearance"},
        {"test": "Urine Routine and Microscopy",     "reason": "Protein, RBC, casts in urine"},
        {"test": "Serum Electrolytes",               "reason": "Sodium, potassium, bicarbonate"},
    ],
    "thyroid": [
        {"test": "TSH",                              "reason": "First-line thyroid screen"},
        {"test": "Free T3 and Free T4",              "reason": "Active thyroid hormone levels"},
        {"test": "Anti-TPO Antibodies",              "reason": "Autoimmune thyroid disease"},
        {"test": "Thyroid Ultrasound",               "reason": "Nodules and gland structure"},
    ],
    "infection": [
        {"test": "Complete Blood Count (CBC)",       "reason": "WBC count — elevated in infection"},
        {"test": "CRP (C-Reactive Protein)",         "reason": "Acute inflammation marker"},
        {"test": "ESR",                              "reason": "Chronic infection/inflammation"},
        {"test": "Blood Culture and Sensitivity",    "reason": "Identifies bacteria and sensitivity"},
        {"test": "Urine Culture",                    "reason": "For suspected UTI"},
    ],
    "anemia": [
        {"test": "Complete Blood Count (CBC)",       "reason": "Hemoglobin, RBC count, MCV"},
        {"test": "Peripheral Blood Smear",           "reason": "RBC shape identifies anemia type"},
        {"test": "Serum Ferritin",                   "reason": "Iron stores in body"},
        {"test": "Serum Iron and TIBC",              "reason": "Iron transport capacity"},
        {"test": "Vitamin B12 and Folate",           "reason": "B12-deficiency anemia screen"},
    ],
    "general": [
        {"test": "Complete Blood Count (CBC)",       "reason": "Overall blood health"},
        {"test": "Comprehensive Metabolic Panel",    "reason": "Liver, kidney, electrolytes, glucose"},
        {"test": "Lipid Profile",                    "reason": "Cholesterol and triglycerides"},
        {"test": "Fasting Blood Glucose",            "reason": "Diabetes screening"},
        {"test": "TSH",                              "reason": "Thyroid function"},
        {"test": "Urine Routine Examination",        "reason": "Kidney and bladder health"},
    ],
}


def _match_tests(keywords: List[str]) -> List[Dict[str, str]]:
    matched, seen = [], set()
    for kw in keywords:
        kl = kw.lower()
        for cat, tests in _LAB_TEST_MAP.items():
            if cat in kl or kl in cat:
                for t in tests:
                    if t["test"] not in seen:
                        matched.append(t)
                        seen.add(t["test"])
    if not matched:
        for t in _LAB_TEST_MAP["general"]:
            if t["test"] not in seen:
                matched.append(t)
                seen.add(t["test"])
    return matched


@traceable(name="retrieve_for_lab", run_type="retriever")
async def _retrieve_for_lab_async(symptoms: str, condition: Optional[str]) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: _get_retriever().get_medicines(
            symptoms=symptoms, disease=condition, top_k=5
        ),
    )


@tool("lab_test_suggester")
def lab_test_suggester(
    symptoms: str,
    suspected_condition: str = "",
    top_k: int = 6,
) -> str:
    """
    Suggests appropriate lab tests and diagnostic tests for given symptoms.
    Use when a user asks what blood tests or lab tests they should get done.

    Args:
        symptoms:             User's symptoms e.g. 'frequent urination, excessive thirst'
        suspected_condition:  Optional suspected disease e.g. 'Diabetes'. Leave empty if unknown.
        top_k:                Number of tests to suggest. Default is 6.

    Returns:
        Formatted list of recommended lab tests with reasons for each.
    """
    try:
        validated = LabToolInput(
            symptoms=symptoms,
            suspected_condition=suspected_condition if suspected_condition else None,
            top_k=top_k,
        )
    except Exception as e:
        return f"Invalid input: {e}."

    raw_result = asyncio.run(
        _retrieve_for_lab_async(
            symptoms=validated.symptoms,
            condition=validated.suspected_condition,
        )
    )

    raw_docs  = raw_result.get("documents", [])
    keywords: List[str] = []
    if validated.suspected_condition:
        keywords.append(validated.suspected_condition)
    for doc in raw_docs:
        disease = doc.metadata.get("disease", "")
        if disease and disease not in keywords:
            keywords.append(disease)
    if not keywords:
        keywords = [symptoms]

    tests = _match_tests(keywords)[: validated.top_k]

    lines = [
        f"Suggested Lab Tests for: '{symptoms}'",
        f"Based on: {', '.join(keywords[:3])}",
        "",
    ]
    for i, t in enumerate(tests, 1):
        lines += [f"{i}. {t['test']}", f"   Why: {t['reason']}", ""]

    lines.append(
        "DISCLAIMER: Lab tests must be ordered by a licensed physician "
        "after clinical examination."
    )
    return "\n".join(lines)
