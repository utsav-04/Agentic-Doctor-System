"""
test.py
========
CLI test for the single-graph flow. Replaces the old 3-graph, multi-invoke
version. Every user turn is exactly ONE graph.invoke() call (plus, rarely,
a second one if the doctor tool paused for location).

Run:
    python test.py                  # free-text mode — planner decides tools
    python test.py --tool medicine  # forced-tool mode — button bypass
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.graph.graphs import build_graph


def run_query(
    graph,
    user_input: str,
    user_age: str = "",
    forced_tool: str = None,
    user_state: str = "",
    user_city: str = "",
    previous_state: dict = None,
):
    """
    One call to this = one graph.invoke(). previous_state lets you carry
    forward plan/tool_outputs/etc. when resuming after a location pause.
    """
    state = {
        **(previous_state or {}),
        "user_input": user_input,
        "user_age": user_age,
        "forced_tool": forced_tool,
        "user_state": user_state,
        "user_city": user_city,
    }
    return graph.invoke(state)


def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tool",
        choices=["medicine", "lab", "doctor", "first_aid"],
        default=None,
        help="Force a single tool (simulates a UI tool button). Omit for free-text planning mode.",
    )
    args = parser.parse_args()

    graph = build_graph()

    print("\n" + "=" * 60)
    print("  Agentic Doctor System — Test")
    print("=" * 60)

    if args.tool:
        print(f"\n[Mode: forced tool = '{args.tool}']")
    else:
        print("\n[Mode: free-text — planner will decide which tool(s) to call]")

    print()
    age = input("  Your age (optional): ").strip()
    symptoms = input("  Describe your symptoms / question: ").strip()

    if not symptoms:
        print("No input entered. Exiting.")
        return

    print("\nProcessing...\n")

    result = run_query(
        graph,
        user_input=symptoms,
        user_age=age,
        forced_tool=args.tool,
    )

    # ── Handle the one case where a single turn isn't enough: doctor
    #    needs a location we don't have yet. ─────────────────────────
    if result.get("stage") == "doctor_need_location":
        print("To find a nearby doctor, I need your location.\n")
        user_state = input("  Your state (e.g. Maharashtra): ").strip().title()
        user_city = input("  Your city  (e.g. Mumbai)     : ").strip().title()
        print("\nSearching for doctors...\n")

        result = run_query(
            graph,
            user_input=symptoms,
            user_age=age,
            forced_tool=args.tool,
            user_state=user_state,
            user_city=user_city,
            previous_state=result,  # carries forward plan + tool_outputs already collected
        )

    # ── Show what happened ────────────────────────────────────────────
    _print_result(result)

    # ── Handle the "want lab/doctor too?" follow-up ────────────────────
    offered = result.get("follow_up_offered")
    if offered:
        answer = input(f"\n  Your answer ({'/'.join(offered)}/both/no): ").strip().lower()

        if "both" in answer or "yes" in answer:
            selected = offered
        else:
            selected = [t for t in offered if t in answer]

        if selected:
            print(f"\nGetting {', '.join(selected)}...\n")

            extended_plan = list(dict.fromkeys(result.get("plan", []) + selected))
            user_state = result.get("user_state", "")
            user_city = result.get("user_city", "")

            if "doctor" in selected and not user_state:
                user_state = input("  Your state (e.g. Maharashtra): ").strip().title()
                user_city = input("  Your city  (e.g. Mumbai)     : ").strip().title()

            result = run_query(
                graph,
                user_input=symptoms,
                user_age=age,
                forced_tool=None,
                user_state=user_state,
                user_city=user_city,
                previous_state={**result, "plan": extended_plan},
            )

            if result.get("stage") == "doctor_need_location":
                print("To find a nearby doctor, I need your location.\n")
                user_state = input("  Your state (e.g. Maharashtra): ").strip().title()
                user_city = input("  Your city  (e.g. Mumbai)     : ").strip().title()
                result = run_query(
                    graph,
                    user_input=symptoms,
                    user_age=age,
                    forced_tool=None,
                    user_state=user_state,
                    user_city=user_city,
                    previous_state=result,
                )

            print("\n" + "=" * 60)
            print("  UPDATED RESULT")
            print("=" * 60)
            _print_result(result)


def _print_result(result: dict):
    if result.get("criticality"):
        print("=" * 60)
        print(f"  CRITICALITY: {result['criticality'].upper()}")
        print("=" * 60)

    if result.get("plan"):
        print(f"  Plan executed: {result['plan']}")
        print("=" * 60)

    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(result.get("final_response", "No response returned."))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_test()