# # from src.graph.graphs import build_graph

# # def ask(prompt: str) -> str:
# #     return input(prompt).strip()


# # graph = build_graph()

# # # ── STEP 1: Welcome ─────────────────────
# # result = graph.invoke({
# #     "messages": [],
# #     "stage": "welcome"   # 👈 important
# # })

# # print("\n=== WELCOME ===\n")
# # print(result.get("agent_output"))


# # # ── STEP 2: User input ──────────────────
# # print("\nEnter your details:\n")

# # name = ask("Your name: ")
# # age = ask("Your age: ")
# # gender = ask("Your gender: ")
# # symptoms = ask("Describe your symptoms: ")

# # if not symptoms:
# #     print("No symptoms entered.")
# #     exit()

# # user_input = (
# #     f"Name: {name}. Age: {age}. Gender: {gender}. Symptoms: {symptoms}"
# # )

# # # ── STEP 3: Intake ──────────────────────
# # result = graph.invoke({
# #     **result,
# #     "stage": "intake",          # 👈 force intake
# #     "user_input": user_input,
# #     "symptoms": symptoms
# # })

# # print("\n=== INTAKE OUTPUT ===\n")
# # print(result.get("agent_output"))


# # # ── STEP 4: Criticality ─────────────────
# # result = graph.invoke({
# #     **result,
# #     "stage": "criticality",     # 👈 force criticality
# # })

# # print("\n=== CRITICALITY OUTPUT ===\n")
# # print(result.get("agent_output"))
# # print("Criticality:", result.get("criticality"))

# from src.graph.graphs import build_graph

# graph = build_graph()

# print("\nEnter your details:\n")

# name = input("Your name: ")
# age = input("Your age: ")
# gender = input("Your gender: ")
# symptoms = input("Describe your symptoms: ")

# input_text = f"""
# Name: {name}
# Age: {age}
# Gender: {gender}
# Symptoms: {symptoms}
# """

# result = graph.invoke({
#     "messages": [],
#     "user_input": input_text
# })

# print("\n=== FINAL OUTPUT ===\n")
# print(result.get("agent_output"))
# print("Criticality:", result.get("criticality"))








"""
test.py
========
Simple CLI test for graph flow:

Flow:
welcome → intake → criticality →
    if critical → doctor (ask location)
    else → intent → medicine/lab/doctor
"""

# from src.graph.graphs import build_graph

# def run_test():
#     graph = build_graph()

#     print("\nEnter your details:\n")

#     name = input("Your name: ")
#     age = input("Your age: ")
#     gender = input("Your gender: ")
#     symptoms = input("Describe your symptoms: ")

#     # Prepare input text
#     intake_text = f"""
#     Name: {name}
#     Age: {age}
#     Gender: {gender}
#     Symptoms: {symptoms}
#     """

#     # ── RUN GRAPH ─────────────────────────────────────────
#     result = graph.invoke({
#         "messages": [],
#         "user_input": intake_text,
#         "symptoms": intake_text
#     })

#     # ── SHOW CRITICALITY RESULT ──────────────────────────
#     print("\n=== CRITICALITY ===\n")
#     print(result.get("agent_output", ""))
#     print("\nCriticality:", result.get("criticality"))

#     # ── IF CRITICAL → ASK LOCATION ───────────────────────
#     if result.get("criticality") == "critical":

#         print("\n⚠️ Emergency detected. Need location to find doctor.\n")

#         state = input("Your state: ")
#         city = input("Your city: ")

#         result = graph.invoke({
#             **result,
#             "user_state": state,
#             "user_city": city,
#             "user_input": f"Find doctor in {city}, {state}"
#         })

#     else:
#         print("\nWhat do you need?")
#         print("1. Medicine")
#         print("2. Lab Test")
#         print("3. Doctor\n")

#         choice = input("Enter choice: ")

#         if choice == "1":
#             user_input = "medicine"
#         elif choice == "2":
#             user_input = "lab"
#         elif choice == "3":
#             user_input = "doctor"
#         else:
#             user_input = choice

#         # ── CASE 1: DOCTOR ─────────────────────────
#         if user_input == "doctor":
#             state = input("Your state: ")
#             city = input("Your city: ")

#             result = graph.invoke({
#                 **result,
#                 "user_input": user_input,
#                 "stage": "intent",
#                 "user_state": state,
#                 "user_city": city,
#                 "symptoms": result.get("symptoms")  # keep context
#             })

#         # ── CASE 2: MEDICINE / LAB ────────────────
#         else:
#             result = graph.invoke({
#                 **result,
#                 "user_input": user_input,
#                 "stage": "intent",
#                 "symptoms": result.get("symptoms")  # VERY IMPORTANT
#             })

#     # ── FINAL OUTPUT ─────────────────────────────────────
#     print("\n=== FINAL OUTPUT ===\n")
#     print(result.get("agent_output", ""))


# if __name__ == "__main__":
#     run_test()



"""
test.py
========
Test runner for the LangGraph flow.

Flow:
  1. Collect name, age, gender, symptoms from user
  2. intake_graph:  welcome → intake → criticality → END
  3. Show criticality result
  4. If CRITICAL  → ask location → doctor_graph → show result
  5. If NOT CRITICAL → ask choice → action_graph (intent → medicine/lab/doctor) → show result
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.graph.graphs import build_intake_graph, build_action_graph, build_doctor_graph


def run_test():

    # Build graphs once
    intake_graph = build_intake_graph()
    action_graph = build_action_graph()
    doctor_graph = build_doctor_graph()

    # ── STEP 1: Collect user details ──────────────────────────────
    print("\n" + "="*60)
    print("  Agentic Doctor System — Test")
    print("="*60)
    print("\nPlease enter your details:\n")

    name     = input("  Your name    : ").strip()
    age      = input("  Your age     : ").strip()
    gender   = input("  Your gender  : ").strip()
    print()
    symptoms = input("  Your symptoms: ").strip()

    if not symptoms:
        print("No symptoms entered. Exiting.")
        return

    # Build intake text passed to intake_node
    intake_text = (
        f"Name: {name}\n"
        f"Age: {age}\n"
        f"Gender: {gender}\n"
        f"Symptoms: {symptoms}"
    )

    print("\nAnalysing...\n")

    # ── STEP 2: Run intake + criticality ──────────────────────────
    result = intake_graph.invoke({
        "messages":    [],
        "user_input":  intake_text,
        "user_name":   name,
        "user_age":    age,
        "user_gender": gender,
        "symptoms":    symptoms,    # ← pass clean symptoms directly too
    })

    criticality = result.get("criticality", "mild")
    crit_output = result.get("agent_output", "")

    # ── STEP 3: Show criticality result ───────────────────────────
    print("\n" + "="*60)
    print(f"  CRITICALITY: {criticality.upper()}")
    print("="*60)
    print(crit_output)
    print("="*60)

    # Carry forward shared state for next graph
    shared = {
        "messages":    result.get("messages", []),
        "symptoms":    result.get("symptoms", symptoms),   # use extracted version
        "user_name":   name,
        "user_age":    age,
        "user_gender": gender,
        "user_info":   result.get("user_info", {}),
        "criticality": criticality,
    }

    # ── STEP 4: Route based on criticality ────────────────────────
    if criticality == "critical":

        print("\nEmergency detected. I need your location to find a nearby doctor.\n")
        user_state = input("  Your state (e.g. Maharashtra): ").strip().title()
        user_city  = input("  Your city  (e.g. Mumbai)     : ").strip().title()
        print("\nSearching for doctors...\n")

        result = doctor_graph.invoke({
            **shared,
            "user_state": user_state,
            "user_city":  user_city,
        })

    else:

        print("\nWhat do you need help with?\n")
        print("  1. Medicine")
        print("  2. Lab Test")
        print("  3. Doctor\n")

        choice = input("  Enter choice (1/2/3): ").strip()

        # Map number to keyword
        choice_map = {"1": "medicine", "2": "lab", "3": "doctor"}
        user_input = choice_map.get(choice, choice.lower())

        # If doctor — get location first
        user_state = ""
        user_city  = ""
        if user_input == "doctor":
            print()
            user_state = input("  Your state (e.g. Maharashtra): ").strip().title()
            user_city  = input("  Your city  (e.g. Mumbai)     : ").strip().title()

        print("\nProcessing...\n")

        # Run action graph: intent → medicine/lab/doctor
        result = action_graph.invoke({
            **shared,
            "user_input": user_input,    # ← user's choice, NOT intake text
            "user_state": user_state,
            "user_city":  user_city,
        })

    # ── STEP 5: Show final output ─────────────────────────────────
    print("\n" + "="*60)
    print("  RESULT")
    print("="*60)
    print(result.get("agent_output", "No output returned."))
    print("="*60 + "\n")


if __name__ == "__main__":
    run_test()