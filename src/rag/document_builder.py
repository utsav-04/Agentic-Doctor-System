"""
document_builder.py
====================
Converts raw JSON data files into LangChain Document objects.

Strategy per collection:
  - page_content  → dense natural-language text that gets embedded (what Gemini reads)
  - metadata      → structured fields used for hard filters BEFORE vector search

The metadata filter approach means:
  state="Maharashtra" + department="Cardiology" narrows the search space BEFORE
  any embedding comparison happens — preventing cross-state hallucination entirely.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document

from config import DataPaths


class MedicineDocumentBuilder:
    """
    Converts medicine_data_full_v3.json into Documents.

    page_content is written as a clinical narrative so the embedding model
    captures semantic meaning (symptoms ↔ medicine relationship).
    Metadata holds filterable scalar fields.
    """

    def build(self) -> List[Document]:
        data = self._load(DataPaths.MEDICINE_JSON)
        docs = []
        for item in data:
            content = self._build_content(item)
            metadata = self._build_metadata(item)
            docs.append(Document(page_content=content, metadata=metadata))
        print(f"[MedicineBuilder] Built {len(docs)} documents")
        return docs

    def _build_content(self, item: Dict[str, Any]) -> str:
        symptoms = ", ".join(item.get("symptoms_treated", []))
        tags = ", ".join(item.get("tags", []))
        warnings = " | ".join(item.get("warnings", []))
        side_effects = ", ".join(item.get("side_effects", []))

        dosage = item.get("dosage_guidelines", {})
        adult_dose = dosage.get("adults", "")
        child_dose = dosage.get("children", "")
        duration = dosage.get("duration", "")

        form = item.get("form", "")
        form_text = f"Form: {form}. " if form else ""

        content = (
            f"Medicine: {item['name']}. "
            f"Used for disease: {item['disease']}. "
            f"{form_text}"
            f"This medicine treats the following symptoms: {symptoms}. "
            f"Adult dosage: {adult_dose}. "
            f"Children dosage: {child_dose}. "
            f"Treatment duration: {duration}. "
            f"Common side effects include: {side_effects}. "
            f"Important warnings: {warnings}. "
            f"Category tags: {tags}."
        )
        return content.strip()

    def _build_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id":          item.get("id", ""),
            "name":        item.get("name", ""),
            "disease":     item.get("disease", ""),
            "form":        item.get("form", "tablet"),
            "source":      "medicine_db",
            "data_type":   "medicine",
            "tags":        "|".join(item.get("tags", [])),
        }

    def _load(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class DoctorDocumentBuilder:
    """
    Converts doctors_full.json into Documents.

    Critical metadata fields:
      - state, city         → geographic hard filter  (no cross-state leakage)
      - department          → specialty hard filter
      - accepts_emergency   → urgency filter
      - is_available_online → mode filter
    """

    def build(self) -> List[Document]:
        data = self._load(DataPaths.DOCTORS_JSON)
        docs = []
        for item in data:
            content = self._build_content(item)
            metadata = self._build_metadata(item)
            docs.append(Document(page_content=content, metadata=metadata))
        print(f"[DoctorBuilder] Built {len(docs)} documents")
        return docs

    def _build_content(self, item: Dict[str, Any]) -> str:
        conditions = ", ".join(item.get("conditions_treated", []))
        languages = ", ".join(item.get("languages_spoken", []))
        days = ", ".join(item.get("available_days", []))
        modes = ", ".join(item.get("appointment_mode", []))

        content = (
            f"Doctor: {item['name']}. "
            f"Department: {item['department']}. "
            f"Qualification: {item['qualification']}. "
            f"Experience: {item['experience_years']} years. "
            f"Hospital: {item['hospital_name']}. "
            f"Location: {item['city']}, {item['state']}. "
            f"Pincode: {item.get('pincode', '')}. "
            f"Address: {item.get('full_address', '')}. "
            f"Treats conditions: {conditions}. "
            f"Languages spoken: {languages}. "
            f"Available on: {days}. "
            f"Timing: {item.get('timing', '')}. "
            f"Consultation fee: Rs {item.get('consultation_fee', '')}. "
            f"Appointment mode: {modes}. "
            f"Accepts emergency: {item.get('accepts_emergency', False)}. "
            f"Online consultation available: {item.get('is_available_online', False)}. "
            f"Online platform: {item.get('online_platform', '')}."
        )
        return content.strip()

    def _build_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id":                  item.get("id", ""),
            "name":                item.get("name", ""),
            "department":          item.get("department", ""),
            "state":               item.get("state", ""),
            "city":                item.get("city", ""),
            "hospital_name":       item.get("hospital_name", ""),
            "consultation_fee":    item.get("consultation_fee", 0),
            "accepts_emergency":   str(item.get("accepts_emergency", False)),
            "is_available_online": str(item.get("is_available_online", False)),
            "source":              "doctor_db",
            "data_type":           "doctor",
        }

    def _load(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class FirstAidDocumentBuilder:
    """
    Converts first_aid_data_v2.json into Documents.

    page_content is deliberately written as instructions — imperative sentences —
    so the model retrieves the most actionable text for urgent queries.
    """

    def build(self) -> List[Document]:
        data = self._load(DataPaths.FIRST_AID_JSON)
        docs = []
        for item in data:
            content = self._build_content(item)
            metadata = self._build_metadata(item)
            docs.append(Document(page_content=content, metadata=metadata))
        print(f"[FirstAidBuilder] Built {len(docs)} documents")
        return docs

    def _build_content(self, item: Dict[str, Any]) -> str:
        steps = " | ".join(item.get("immediate_steps", []))
        remedies = " | ".join(item.get("home_remedies", []))
        do_not = " | ".join(item.get("do_not_do", []))
        go_to_doc = " | ".join(item.get("go_to_doctor_if", []))
        tags = ", ".join(item.get("tags", []))

        content = (
            f"First Aid: {item['name']}. "
            f"Condition: {item['condition']}. "
            f"Category: {item.get('category', '')}. "
            f"Severity level: {item.get('severity_level', '')}. "
            f"When to apply: {item.get('when_to_apply', '')}. "
            f"Immediate steps to take: {steps}. "
            f"Home remedies and Indian household treatments: {remedies}. "
            f"What NOT to do: {do_not}. "
            f"Go to doctor immediately if: {go_to_doc}. "
            f"Tags: {tags}."
        )
        return content.strip()

    def _build_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id":             item.get("id", ""),
            "condition":      item.get("condition", ""),
            "severity_level": item.get("severity_level", "mild"),
            "category":       item.get("category", "first_aid"),
            "source":         "first_aid_db",
            "data_type":      "first_aid",
            "tags":           "|".join(item.get("tags", [])),
        }

    def _load(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

class LabTestDocumentBuilder:
    """
    Converts lab test JSON into Documents.

    page_content = patient scenario + recommended tests + reasoning
    metadata     = filterable fields (severity, condition, etc.)
    """

    def build(self) -> List[Document]:
        data = self._load(DataPaths.LABTEST_JSON)
        docs = []

        for item in data:
            content = self._build_content(item)
            metadata = self._build_metadata(item)

            docs.append(Document(page_content=content, metadata=metadata))

        print(f"[LabTestBuilder] Built {len(docs)} documents")
        return docs

    def _build_content(self, item: Dict[str, Any]) -> str:
        tests = item.get("recommended_tests", [])
        test_lines = [
            f"{t['test_name']} (Reason: {t['reason']})"
            for t in tests
        ]
        tests_text = " | ".join(test_lines)

        conditions = ", ".join(item.get("possible_conditions", []))

        content = (
            f"Patient complaint: {item.get('patient_input', '')}. "
            f"Possible conditions: {conditions}. "
            f"Recommended lab tests: {tests_text}. "
            f"Severity level: {item.get('severity', '')}. "
            f"Suggested specialist: {item.get('specialist_referral', '')}."
        )

        return content.strip()

    def _build_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": item.get("id", ""),
            "severity": item.get("severity", ""),
            "specialist": item.get("specialist_referral", ""),
            "data_type": "lab_test",
            "source": "lab_test_db",
        }

    def _load(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


