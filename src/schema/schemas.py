"""
src/schemas.py
===============
Pydantic v2 models for the entire system.

Every piece of data that flows between modules is validated here:
  - Raw JSON input validation (catches bad data before embedding)
  - Retrieval output validation (catches hallucinated/malformed RAG results)
  - Tool input/output validation (CrewAI tool contracts)
  - Agent state validation (LangGraph state schema)

Why Pydantic here:
  - Medical data must be type-safe — a string where a list is expected
    could cause a tool to silently return wrong results
  - LangSmith traces structured outputs, so validated models produce
    cleaner, more inspectable trace payloads
  - CrewAI tools expect strict input types; Pydantic enforces this at runtime
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# RAW DATA SCHEMAS  (validate JSON before embedding)
# ─────────────────────────────────────────────────────────────────────────────

class MedicineRecord(BaseModel):
    """Validates one record from medicine_data_full_v3.json"""
    id: str = Field(..., description="Unique medicine ID e.g. MED-0001")
    name: str = Field(..., min_length=2)
    disease: str = Field(..., min_length=2)
    form: Optional[str] = Field(default="tablet")
    symptoms_treated: List[str] = Field(..., min_length=1)
    dosage_guidelines: Dict[str, str] = Field(...)
    side_effects: List[str] = Field(..., min_length=1, max_length=10)
    warnings: List[str] = Field(..., min_length=1, max_length=5)
    tags: List[str] = Field(..., min_length=1, max_length=10)

    @field_validator("symptoms_treated", "side_effects", "warnings", "tags")
    @classmethod
    def no_empty_strings(cls, v: List[str]) -> List[str]:
        cleaned = [s.strip() for s in v if s.strip()]
        if not cleaned:
            raise ValueError("List must contain at least one non-empty string")
        return cleaned

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        if not v.startswith("MED-"):
            raise ValueError(f"Medicine ID must start with 'MED-', got: {v}")
        return v


class DosageGuidelines(BaseModel):
    adults: str = Field(..., min_length=3)
    children: str = Field(..., min_length=3)
    duration: Optional[str] = None


class DoctorRecord(BaseModel):
    """Validates one record from doctors_full.json"""
    id: str = Field(...)
    name: str = Field(..., min_length=4)
    qualification: str = Field(..., min_length=2)
    experience_years: int = Field(..., ge=0, le=60)
    languages_spoken: List[str] = Field(..., min_length=1)
    department: str = Field(...)
    conditions_treated: List[str] = Field(..., min_length=1)
    state: str = Field(...)
    city: str = Field(...)
    hospital_name: str = Field(...)
    pincode: str = Field(...)
    full_address: str = Field(...)
    phone: str = Field(...)
    email: str = Field(...)
    consultation_fee: int = Field(..., ge=0, le=10000)
    appointment_mode: List[str] = Field(...)
    available_days: List[str] = Field(...)
    timing: str = Field(...)
    accepts_emergency: bool = Field(...)
    is_available_online: bool = Field(...)
    online_platform: Optional[str] = Field(default="")

    @field_validator("appointment_mode")
    @classmethod
    def validate_modes(cls, v: List[str]) -> List[str]:
        valid = {"in-person", "online"}
        for mode in v:
            if mode not in valid:
                raise ValueError(f"Invalid appointment mode: {mode}")
        return v

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        digits = v.replace("+", "").replace("-", "").replace(" ", "")
        if not digits.isdigit():
            raise ValueError(f"Phone must contain only digits: {v}")
        return v


class FirstAidRecord(BaseModel):
    """Validates one record from first_aid_data_v2.json"""
    id: str = Field(...)
    name: str = Field(...)
    category: str = Field(...)
    condition: str = Field(...)
    severity_level: Literal[
        "mild", "mild_to_moderate", "moderate_to_severe",
        "mild_to_severe", "urgent", "life_threatening", "preventive"
    ]
    when_to_apply: str = Field(...)
    immediate_steps: List[str] = Field(..., min_length=1)
    home_remedies: List[str] = Field(..., min_length=1)
    do_not_do: List[str] = Field(..., min_length=1)
    go_to_doctor_if: List[str] = Field(..., min_length=1)
    tags: List[str] = Field(..., min_length=1)


class RecommendedTest(BaseModel):
    """A single test inside a LabTestRecord."""
    test_name: str = Field(..., min_length=2)
    reason: str = Field(..., min_length=3)


class LabTestRecord(BaseModel):
    """
    Validates one record from lab_test_data.json.

    Example record:
    {
      "id": "MED-001",
      "patient_input": "fever from 2 days, body pain, headache...",
      "recommended_tests": [
        {"test_name": "CBC", "reason": "check infection"},
        ...
      ],
      "possible_conditions": ["Viral Fever", "Dengue"],
      "severity": "medium",
      "specialist_referral": "General Physician"
    }
    """
    id: str = Field(..., min_length=1)
    patient_input: str = Field(..., min_length=5)
    recommended_tests: List[RecommendedTest] = Field(..., min_length=1)
    possible_conditions: List[str] = Field(..., min_length=1)
    severity: Literal["low", "medium", "high", "critical"]
    specialist_referral: str = Field(..., min_length=2)

    @field_validator("possible_conditions")
    @classmethod
    def no_empty_conditions(cls, v: List[str]) -> List[str]:
        cleaned = [c.strip() for c in v if c.strip()]
        if not cleaned:
            raise ValueError("possible_conditions must have at least one non-empty entry")
        return cleaned

    @field_validator("recommended_tests")
    @classmethod
    def at_least_one_test(cls, v: List[RecommendedTest]) -> List[RecommendedTest]:
        if not v:
            raise ValueError("recommended_tests must have at least one test")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL OUTPUT SCHEMAS  (validate what RAG returns)
# ─────────────────────────────────────────────────────────────────────────────

class RetrievedMedicine(BaseModel):
    """A single medicine result returned by the RAG retriever."""
    id: str
    name: str
    disease: str
    form: str
    symptoms_matched: List[str]
    adult_dosage: str
    child_dosage: str
    duration: str
    side_effects: List[str]
    warnings: List[str]
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @classmethod
    def from_document(cls, doc: Any, score: Optional[float] = None) -> "RetrievedMedicine":
        """Build from a LangChain Document + its metadata."""
        meta = doc.metadata
        content = doc.page_content

        def extract(label: str) -> str:
            import re
            pattern = rf"{re.escape(label)}:\s*([^.]+)\."
            m = re.search(pattern, content)
            return m.group(1).strip() if m else ""

        symptoms_raw = extract("This medicine treats the following symptoms")
        symptoms = [s.strip() for s in symptoms_raw.split(",") if s.strip()]

        return cls(
            id=meta.get("id", ""),
            name=meta.get("name", ""),
            disease=meta.get("disease", ""),
            form=meta.get("form", ""),
            symptoms_matched=symptoms,
            adult_dosage=extract("Adult dosage"),
            child_dosage=extract("Children dosage"),
            duration=extract("Treatment duration"),
            side_effects=[s.strip() for s in extract("Common side effects include").split(",") if s.strip()],
            warnings=[w.strip() for w in extract("Important warnings").split("|") if w.strip()],
            relevance_score=score,
        )


class RetrievedDoctor(BaseModel):
    """A single doctor result returned by the RAG retriever."""
    id: str
    name: str
    department: str
    qualification: str
    experience_years: int
    hospital_name: str
    city: str
    state: str
    phone: str
    consultation_fee: int
    appointment_mode: List[str]
    available_days: List[str]
    timing: str
    accepts_emergency: bool
    is_available_online: bool
    online_platform: str
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @classmethod
    def from_document(cls, doc: Any, score: Optional[float] = None) -> "RetrievedDoctor":
        meta = doc.metadata
        content = doc.page_content

        def extract(label: str) -> str:
            import re
            pattern = rf"{re.escape(label)}:\s*([^.]+)\."
            m = re.search(pattern, content)
            return m.group(1).strip() if m else ""

        modes_raw = extract("Appointment mode")
        modes = [m.strip() for m in modes_raw.split(",") if m.strip()]

        days_raw = extract("Available on")
        days = [d.strip() for d in days_raw.split(",") if d.strip()]

        fee_raw = extract("Consultation fee")
        fee = int("".join(filter(str.isdigit, fee_raw))) if fee_raw else 0

        exp_raw = extract("Experience")
        exp = int("".join(filter(str.isdigit, exp_raw.split()[0]))) if exp_raw else 0

        emergency_raw = extract("Accepts emergency")
        online_raw = extract("Online consultation available")

        return cls(
            id=meta.get("id", ""),
            name=meta.get("name", ""),
            department=meta.get("department", ""),
            qualification=extract("Qualification"),
            experience_years=exp,
            hospital_name=meta.get("hospital_name", ""),
            city=meta.get("city", ""),
            state=meta.get("state", ""),
            phone=extract("") or "",
            consultation_fee=fee,
            appointment_mode=modes if modes else ["in-person"],
            available_days=days,
            timing=extract("Timing"),
            accepts_emergency=(meta.get("accepts_emergency", "False") == "True"),
            is_available_online=(meta.get("is_available_online", "False") == "True"),
            online_platform=meta.get("online_platform", ""),
            relevance_score=score,
        )


class RetrievedFirstAid(BaseModel):
    """A single first-aid result returned by the RAG retriever."""
    id: str
    condition: str
    severity_level: str
    when_to_apply: str
    immediate_steps: List[str]
    home_remedies: List[str]
    do_not_do: List[str]
    go_to_doctor_if: List[str]
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @classmethod
    def from_document(cls, doc: Any, score: Optional[float] = None) -> "RetrievedFirstAid":
        meta = doc.metadata
        content = doc.page_content

        def extract_list(label: str) -> List[str]:
            import re
            pattern = rf"{re.escape(label)}:\s*([^.]+(?:\.[^A-Z][^.]*)*)\."
            m = re.search(pattern, content)
            if not m:
                return []
            return [s.strip() for s in m.group(1).split("|") if s.strip()]

        return cls(
            id=meta.get("id", ""),
            condition=meta.get("condition", ""),
            severity_level=meta.get("severity_level", "mild"),
            when_to_apply="",
            immediate_steps=extract_list("Immediate steps to take"),
            home_remedies=extract_list("Home remedies and Indian household treatments"),
            do_not_do=extract_list("What NOT to do"),
            go_to_doctor_if=extract_list("Go to doctor immediately if"),
            relevance_score=score,
        )


class RetrievedLabTest(BaseModel):
    """
    A single lab test result returned by the RAG retriever.
    Built from a LabTestRecord document stored in ChromaDB.
    """
    id: str
    patient_input: str
    possible_conditions: List[str]
    recommended_tests: List[Dict[str, str]]   # [{"test_name": ..., "reason": ...}]
    severity: str
    specialist_referral: str
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @classmethod
    def from_document(cls, doc: Any, score: Optional[float] = None) -> "RetrievedLabTest":
        """
        Parses a LangChain Document back into a RetrievedLabTest.
        Uses metadata first (fast), falls back to content parsing.
        """
        import re
        meta    = doc.metadata
        content = doc.page_content

        def extract(label: str) -> str:
            pattern = rf"{re.escape(label)}:\s*([^.]+)\."
            m = re.search(pattern, content)
            return m.group(1).strip() if m else ""

        # Parse possible_conditions from content
        conditions_raw = extract("Possible conditions")
        conditions = [c.strip() for c in conditions_raw.split(",") if c.strip()]

        # Parse recommended tests from content — format: "TestName (Reason: ...) | ..."
        tests_raw = extract("Recommended lab tests")
        tests: List[Dict[str, str]] = []
        if tests_raw:
            for part in tests_raw.split(" | "):
                part = part.strip()
                m = re.match(r"^(.+?)\s*\(Reason:\s*(.+?)\)$", part)
                if m:
                    tests.append({
                        "test_name": m.group(1).strip(),
                        "reason":    m.group(2).strip(),
                    })
                elif part:
                    tests.append({"test_name": part, "reason": ""})

        return cls(
            id=meta.get("id", ""),
            patient_input=extract("Patient complaint") or content[:120],
            possible_conditions=conditions,
            recommended_tests=tests,
            severity=meta.get("severity", ""),
            specialist_referral=meta.get("specialist", ""),
            relevance_score=score,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TOOL INPUT/OUTPUT SCHEMAS  (CrewAI tool contracts)
# ─────────────────────────────────────────────────────────────────────────────

class MedicineToolInput(BaseModel):
    """Input schema for MedicineTool — validated before RAG call."""
    symptoms: str = Field(
        ...,
        min_length=3,
        description="User-described symptoms e.g. 'fever and dry cough'"
    )
    disease: Optional[str] = Field(
        default=None,
        description="Specific disease name if known e.g. 'Diabetes'"
    )
    patient_age: Optional[int] = Field(
        default=None,
        ge=0, le=120,
        description="Patient age in years to filter adult vs child dosage"
    )
    top_k: int = Field(default=5, ge=1, le=10)


class MedicineToolOutput(BaseModel):
    """Output schema for MedicineTool — validated before passing to agent."""
    query: str
    total_results: int
    medicines: List[RetrievedMedicine]
    disclaimer: str = Field(
        default="This information is for reference only. Always consult a qualified doctor before taking any medication."
    )
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)


class DoctorToolInput(BaseModel):
    """Input schema for DoctorTool."""
    symptoms: str = Field(..., min_length=3)
    state: str = Field(..., min_length=2, description="Indian state name e.g. 'Maharashtra'")
    city: Optional[str] = Field(default=None, description="City name e.g. 'Mumbai'")
    department: Optional[str] = Field(default=None, description="Medical department e.g. 'Cardiology'")
    emergency: bool = Field(default=False, description="If True, only return emergency-accepting doctors")
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("state")
    @classmethod
    def normalize_state(cls, v: str) -> str:
        return v.strip().title()

    @field_validator("city")
    @classmethod
    def normalize_city(cls, v: Optional[str]) -> Optional[str]:
        return v.strip().title() if v else None


class DoctorToolOutput(BaseModel):
    """Output schema for DoctorTool."""
    query: str
    state: str
    city: Optional[str]
    department: Optional[str]
    total_results: int
    doctors: List[RetrievedDoctor]
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)


class FirstAidToolInput(BaseModel):
    """Input schema for FirstAidTool."""
    condition: str = Field(..., min_length=3, description="Condition or emergency e.g. 'heart attack'")
    severity: Optional[str] = Field(
        default=None,
        description="Filter by severity: mild, urgent, life_threatening etc."
    )
    top_k: int = Field(default=3, ge=1, le=5)


class FirstAidToolOutput(BaseModel):
    """Output schema for FirstAidTool."""
    query: str
    total_results: int
    first_aid_guides: List[RetrievedFirstAid]
    emergency_notice: str = Field(
        default="For life-threatening emergencies, call 108 immediately."
    )
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)


class LabToolInput(BaseModel):
    """Input schema for LabTool — now backed by real lab test RAG data."""
    symptoms: str = Field(
        ...,
        min_length=3,
        description="User-described symptoms e.g. 'fever and body pain for 2 days'"
    )
    severity: Optional[str] = Field(
        default=None,
        description="Filter by severity: 'low', 'medium', 'high', 'critical'"
    )
    top_k: int = Field(default=3, ge=1, le=10)


class LabToolOutput(BaseModel):
    """Output schema for LabTool — validated results from lab_test RAG collection."""
    query: str
    total_results: int
    lab_results: List[RetrievedLabTest]
    disclaimer: str = Field(
        default="Lab tests must be ordered by a licensed physician after clinical examination."
    )
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def validate_json_records(data: list, schema_class: type, source_name: str) -> tuple[list, list]:
    """
    Validates a list of raw JSON dicts against a Pydantic model.
    Returns (valid_records, error_log).
    Used in ingest.py before embedding to catch bad data early.
    """
    valid = []
    errors = []
    for i, record in enumerate(data):
        try:
            validated = schema_class(**record)
            valid.append(validated)
        except Exception as e:
            errors.append({
                "index": i,
                "id": record.get("id", f"index_{i}"),
                "source": source_name,
                "error": str(e),
            })
    return valid, errors