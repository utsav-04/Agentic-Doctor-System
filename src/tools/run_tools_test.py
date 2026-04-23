"""
src/tools/run_tools_test.py
============================
Run after ingestion to verify tools + agents load correctly.

    python -m src.tools.run_tools_test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def sep(title: str):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")


def test_medicine():
    sep("TEST: medicine_finder")
    from src.tools.medicine_tool import medicine_finder
    result = medicine_finder("fever and dry cough", patient_age=8, top_k=3)
    print(result)
    assert len(result) > 50
    print("[PASS]")


def test_doctor():
    sep("TEST: doctor_finder")
    from src.tools.doctor_tool import doctor_finder
    result = doctor_finder(
        symptoms="chest pain",
        state="Maharashtra",
        city="Mumbai",
        department="Cardiology",
        emergency=True,
        top_k=3,
    )
    print(result)
    assert len(result) > 50
    print("[PASS]")


def test_first_aid():
    sep("TEST: first_aid_guide")
    from src.tools.first_aid_tool import first_aid_guide
    result = first_aid_guide("heart attack", severity="life_threatening", top_k=1)
    print(result)
    assert len(result) > 50
    print("[PASS]")


def test_lab():
    sep("TEST: lab_test_suggester")
    from src.tools.lab_tool import lab_test_suggester
    result = lab_test_suggester(
        symptoms="frequent urination and excessive thirst",
        suspected_condition="Diabetes",
        top_k=5,
    )
    print(result)
    assert len(result) > 50
    print("[PASS]")


def test_agents_import():
    sep("TEST: agents import")
    from src.agents.agents import (
        intake_agent, criticality_agent,
        medicine_agent, doctor_agent, lab_agent,
    )
    for agent in [intake_agent, criticality_agent, medicine_agent, doctor_agent, lab_agent]:
        assert agent.role, f"Agent missing role: {agent}"
        print(f"  [OK] {agent.role}")
    print("[PASS]")


if __name__ == "__main__":
    print("\nAgentic Doctor System — Module 2 Smoke Test")
    print("LangSmith traces → https://smith.langchain.com\n")

    passed, failed = 0, 0
    for fn in [test_medicine, test_doctor, test_first_aid, test_lab, test_agents_import]:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*55}")
    print(f"  {passed} passed  |  {failed} failed")
    print(f"{'='*55}")
    if failed:
        print("\nIf tests fail:")
        print("  1. Run ingest first:   python -m src.rag.ingest")
        print("  2. Check .env has     GOOGLE_API_KEY set")
        sys.exit(1)
