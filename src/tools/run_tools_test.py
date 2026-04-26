"""
src/tools/run_tools_test.py
============================
Smoke test for all 4 tools and 5 agents.

FIX: CrewAI @tool decorator wraps functions into Tool objects.
     Call tool.run(...) with a single dict argument, NOT tool(arg=val).

Run:
    python -m src.tools.run_tools_test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.logger import get_logger
logger = get_logger(__name__)


def sep(title: str):
    logger.info("=" * 55)
    logger.info("  %s", title)
    logger.info("=" * 55)

def test_doctor():
    sep("TEST: doctor_finder")
    from src.tools.doctor_tool import doctor_finder

    result = doctor_finder.run(**{
        "symptoms":   "chest pain",
        "state":      "Maharashtra",
        "city":       "Mumbai",
        "department": "Cardiology",
        "emergency":  True,
        "top_k":      3,
    })
    logger.info("\n%s", result)
    assert len(result) > 50, "doctor_finder returned empty result"
    logger.info("[PASS] doctor_finder")

def test_medicine():
    sep("TEST: medicine_finder")
    from src.tools.medicine_tool import medicine_finder

    # CrewAI @tool objects are called via .run() with a single dict
    result = medicine_finder.run(**{
        "symptoms":    "fever and dry cough",
        "patient_age": 8,
        "top_k":       3,
    })
    logger.info("\n%s", result)
    assert len(result) > 50, "medicine_finder returned empty result"
    logger.info("[PASS] medicine_finder")





def test_first_aid():
    sep("TEST: first_aid_guide")
    from src.tools.first_aid_tool import first_aid_guide

    result = first_aid_guide.run(**{
        "condition": "heart attack",
        "severity":  "life_threatening",
        "top_k":     1,
    })
    logger.info("\n%s", result)
    assert len(result) > 50, "first_aid_guide returned empty result"
    logger.info("[PASS] first_aid_guide")


def test_lab():
    sep("TEST: lab_test_suggester")
    from src.tools.lab_tool import lab_test_suggester

    result = lab_test_suggester.run(**{
        "symptoms":  "frequent urination and excessive thirst",
        "severity":  "",
        "top_k":     5,
    })
    logger.info("\n%s", result)
    assert len(result) > 50, "lab_test_suggester returned empty result"
    logger.info("[PASS] lab_test_suggester")


def test_agents_import():
    sep("TEST: agents import")
    from src.agents.agents import (
        intake_agent, criticality_agent,
        medicine_agent, doctor_agent, lab_agent,
    )
    for agent in [intake_agent, criticality_agent, medicine_agent, doctor_agent, lab_agent]:
        assert agent.role, f"Agent missing role: {agent}"
        logger.info("  [OK] %s", agent.role)
    logger.info("[PASS] agents import")


if __name__ == "__main__":
    logger.info("Agentic Doctor System — Module 2 Smoke Test")
    logger.info("LangSmith traces at: https://smith.langchain.com")

    tests  = [test_medicine, test_doctor, test_first_aid, test_lab, test_agents_import]
    passed = failed = 0

    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            logger.error("[FAIL] %s: %s", fn.__name__, e, exc_info=True)
            failed += 1

    logger.info("=" * 55)
    logger.info("  %d passed  |  %d failed", passed, failed)
    logger.info("=" * 55)
    logger.info("Full log at: logs/agentic_doctor.log")

    if failed:
        logger.error(
            "If tests fail:\n"
            "  1. Run ingest first:  python -m src.rag.ingest\n"
            "  2. Check .env has GOOGLE_API_KEY and HF_TOKEN set"
        )
        sys.exit(1)