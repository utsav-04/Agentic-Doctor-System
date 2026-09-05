# """
# src/graph/nodes.py
# ===================
# LangGraph nodes. Nodes process state — they do NOT call input().
# All user prompting happens in the API layer / UI.

# Flow this file supports (wired together in graphs.py):

#     welcome -> intake -> criticality -> [critical]-----> executor -> synthesis -> END
#                                      -> [not critical] -> planner -> executor -> synthesis -> END

#     forced_tool (UI button) set -> planner (bypass mode) -> executor -> synthesis -> END

# Removed from the old version: doctor_node / medicine_node / lab_node / intent_node /
# response_node (replaced by planner_node + executor_node + synthesis_node), hitl_node
# (never wired into a graph — reintroduce later if you actually want human review),
# and ~800 lines of commented-out earlier implementations.
# """

# import json
# import re
# import sys
# from pathlib import Path

# from langsmith import traceable

# sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# from src.graph.state import AgentState
# from src.graph.router import detect_criticality, detect_plan
# from src.agents.agents import llm
# from src.cache.redis_cache import make_key, cache_get, cache_set
# from src.tools.medicine_tool import medicine_finder
# from src.tools.lab_tool import lab_test_suggester
# from src.tools.doctor_tool import doctor_finder
# from src.tools.first_aid_tool import first_aid_guide
# from src.logger import get_logger

# logger = get_logger(__name__)

# NEED_LOCATION = "NEED_LOCATION"

# # RAG data doesn't change unless you re-run ingestion, so tool results can
# # be cached for a while. If you re-ingest, call
# # cache_delete_prefix("tool:") to invalidate stale entries.
# TOOL_CACHE_TTL = 60 * 60 * 24 * 7  # 7 days

# # Criticality/triage text are near-deterministic (temp=0.1) — cache them.
# LLM_CACHE_TTL = 60 * 60 * 24  # 1 day

# # Full-response cache — shorter TTL since this is the coarsest-grained
# # cache (a hit skips literally everything, including re-triage).
# FULL_RESPONSE_CACHE_TTL = 60 * 60 * 6  # 6 hours


# def _clean(text: str) -> str:
#     """Strip <think>...</think> blocks that some local models add."""
#     cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
#     return cleaned.strip()


# def _extract_symptoms(raw: str) -> str:
#     """
#     Extracts only the symptom text from the intake string.
#     Input:  'Name: Utsav Raj\\nAge: 5\\nGender: Male\\nSymptoms: Cold from yesterday'
#     Output: 'Cold from yesterday'
#     """
#     match = re.search(r"[Ss]ymptoms?\s*:\s*(.+?)(?:\n|$)", raw)
#     if match:
#         return match.group(1).strip()
#     return raw.strip()


# # ─────────────────────────────────────────────────────────────────────────────
# # WELCOME NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="welcome_node", run_type="chain")
# def welcome_node(state: AgentState) -> dict:
#     logger.info("welcome_node executed")
#     return {"messages": [], "stage": "welcome"}


# # ─────────────────────────────────────────────────────────────────────────────
# # INTAKE NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="intake_node", run_type="chain")
# def intake_node(state: AgentState) -> dict:
#     user_input = state.get("user_input", "")
#     logger.info("intake_node — processing: '%s'", user_input[:80])

#     symptoms_only = _extract_symptoms(user_input) or user_input

#     messages = list(state.get("messages", []))
#     messages.append(f"[Intake] symptoms='{symptoms_only}'")

#     logger.info("intake_node complete — symptoms: '%s'", symptoms_only[:80])

#     return {
#         "symptoms": symptoms_only,
#         "messages": messages,
#         "stage": "intake",
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # CRITICALITY NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="criticality_node", run_type="chain")
# def criticality_node(state: AgentState) -> dict:
#     symptoms = state.get("symptoms", state.get("user_input", ""))
#     logger.info("criticality_node — symptoms: '%s'", symptoms[:80])

#     criticality = detect_criticality(symptoms)  # cached inside router.py
#     logger.info("criticality_node — result: %s", criticality.upper())

#     triage_key = make_key("llm:triage_msg", symptoms=symptoms, criticality=criticality)
#     output = cache_get(triage_key)

#     if not output:
#         prompt = f"""
# You are a warm, empathetic medical triage assistant in an Indian healthcare chatbot.

# Patient symptoms: {symptoms}
# Assessed severity: {criticality.upper()}

# Write a short, clear, empathetic triage message (2-3 sentences):
# - CRITICAL: Tell the patient this is an emergency. Tell them to call 108 or
#   go to the emergency room NOW. First aid steps will be provided separately.
# - MODERATE: Reassure them but advise seeing a doctor soon.
# - MILD: Reassure them it can likely be managed at home.
# """
#         response = llm.call(prompt)
#         output = _clean(str(response))
#         cache_set(triage_key, output, ttl=LLM_CACHE_TTL)

#     messages = list(state.get("messages", []))
#     messages.append(f"[Criticality: {criticality.upper()}]\n{output}")

#     update = {
#         "criticality": criticality.lower(),
#         "agent_output": output,
#         "messages": messages,
#         "stage": "criticality",
#     }

#     # Critical cases skip the planner entirely — the plan is fixed:
#     # first aid guidance immediately, then point to a doctor.
#     if criticality.lower() == "critical":
#         update["plan"] = ["first_aid", "doctor"]

#     return update


# # ─────────────────────────────────────────────────────────────────────────────
# # CACHE CHECK NODE  (NEW — full-response short-circuit)
# # ─────────────────────────────────────────────────────────────────────────────

# def _full_cache_key(state: AgentState) -> str:
#     """
#     Built ONLY from pre-planning inputs (symptoms/age/location/forced_tool) —
#     deliberately excludes 'plan', because plan doesn't exist yet on a fresh
#     call. This also means a resumed call (which DOES carry a plan forward)
#     naturally can't collide with this key even if computed again.
#     """
#     return make_key(
#         "full:response",
#         forced_tool=state.get("forced_tool") or "",
#         symptoms=state.get("symptoms", ""),
#         age=state.get("user_age", ""),
#         user_state=state.get("user_state", ""),
#         user_city=state.get("user_city", ""),
#     )


# @traceable(name="cache_check_node", run_type="chain")
# def cache_check_node(state: AgentState) -> dict:
#     """
#     Runs right after intake. If this is a fresh query (no 'plan' carried in
#     from a prior turn — see the docstring on _full_cache_key) and we've
#     answered this exact symptoms+age+location+forced_tool combo before,
#     skip criticality, planning, tool execution, AND synthesis entirely.

#     A carried-in 'plan' means this is a resume (doctor-location pause, or
#     a follow-up like "yes, also lab") — those must never be short-circuited,
#     since the whole point is to add to what's already been computed.
#     """
#     if state.get("plan"):
#         return {"stage": "cache_checked"}

#     key = _full_cache_key(state)
#     cached = cache_get(key)
#     if cached:
#         try:
#             data = json.loads(cached)
#             logger.info("cache_check_node — full response cache HIT")
#             return {
#                 "final_response": data.get("final_response", ""),
#                 "criticality": data.get("criticality", ""),
#                 "plan": data.get("plan", []),
#                 "stage": "cache_hit",
#             }
#         except (json.JSONDecodeError, TypeError):
#             logger.warning("cache_check_node — cached value unparsable, ignoring")

#     return {"stage": "cache_checked"}


# # ─────────────────────────────────────────────────────────────────────────────
# # PLANNER NODE  (replaces intent_node — produces a list, not a single label)
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="planner_node", run_type="chain")
# def planner_node(state: AgentState) -> dict:
#     """
#     Two modes:
#       1. forced_tool is set (UI button pressed) -> plan is exactly that one tool.
#          No LLM call needed — this is a hard bypass.
#       2. forced_tool is None -> ask the LLM to plan the full tool list for this
#          free-text query (see detect_plan in router.py).
#     """
#     forced_tool = state.get("forced_tool")

#     if forced_tool:
#         logger.info("planner_node — forced_tool='%s', skipping LLM planning", forced_tool)
#         return {"plan": [forced_tool], "stage": "planned_forced"}

#     # Resuming after a doctor_need_location pause — the plan was already
#     # decided last turn (and possibly partly executed). Don't re-plan.
#     if state.get("plan"):
#         logger.info("planner_node — resuming existing plan: %s", state["plan"])
#         return {"plan": state["plan"], "stage": "planned_resumed"}

#     user_input = state.get("user_input", "")
#     symptoms = state.get("symptoms", user_input)

#     plan = detect_plan(user_input=user_input, symptoms=symptoms)
#     logger.info("planner_node — plan: %s", plan)

#     messages = list(state.get("messages", []))
#     messages.append(f"[Plan] {plan}")

#     return {"plan": plan, "messages": messages, "stage": "planned"}


# # ─────────────────────────────────────────────────────────────────────────────
# # EXECUTOR NODE  (replaces doctor_node / medicine_node / lab_node)
# # ─────────────────────────────────────────────────────────────────────────────

# def _call_medicine(state: AgentState) -> str:
#     symptoms = state.get("symptoms", "")
#     age = state.get("user_age")
#     age_val = int(age) if age and str(age).isdigit() else -1

#     key = make_key("tool:medicine", symptoms=symptoms, age=age_val)
#     cached = cache_get(key)
#     if cached is not None:
#         return cached

#     result = medicine_finder.run(**{
#         "symptoms": symptoms,
#         "disease": "",
#         "patient_age": age_val,
#         "top_k": 3,
#     })
#     cache_set(key, result, ttl=TOOL_CACHE_TTL)
#     return result


# def _call_lab(state: AgentState) -> str:
#     symptoms = state.get("symptoms", "")

#     key = make_key("tool:lab", symptoms=symptoms)
#     cached = cache_get(key)
#     if cached is not None:
#         return cached

#     result = lab_test_suggester.run(**{
#         "symptoms": symptoms,
#         "severity": "",
#         "top_k": 3,
#     })
#     cache_set(key, result, ttl=TOOL_CACHE_TTL)
#     return result


# def _call_first_aid(state: AgentState) -> str:
#     symptoms = state.get("symptoms", "")
#     criticality = state.get("criticality", "mild")
#     severity_map = {"critical": "life_threatening", "moderate": "urgent", "mild": "mild"}
#     severity = severity_map.get(criticality, "")

#     key = make_key("tool:first_aid", symptoms=symptoms, severity=severity)
#     cached = cache_get(key)
#     if cached is not None:
#         return cached

#     result = first_aid_guide.run(**{
#         "condition": symptoms,
#         "severity": severity,
#         "top_k": 2,
#     })
#     cache_set(key, result, ttl=TOOL_CACHE_TTL)
#     return result


# def _call_doctor(state: AgentState) -> str:
#     symptoms = state.get("symptoms", "")
#     user_state = state.get("user_state", "")
#     user_city = state.get("user_city", "")
#     criticality = state.get("criticality", "moderate")

#     if not user_state:
#         return NEED_LOCATION

#     department = ""
#     s = symptoms.lower()
#     if "chest pain" in s:
#         department = "Cardiology"
#     elif "fever" in s:
#         department = "General Physician"
#     elif "skin" in s:
#         department = "Dermatology"
#     elif "child" in s or "kid" in s:
#         department = "Pediatrics"
#     elif "pregnant" in s or "pregnancy" in s:
#         department = "Gynecology"

#     emergency = criticality == "critical"
#     key = make_key(
#         "tool:doctor", symptoms=symptoms, state=user_state, city=user_city,
#         department=department, emergency=emergency,
#     )
#     cached = cache_get(key)
#     if cached is not None:
#         return cached

#     result = doctor_finder.run(**{
#         "symptoms": symptoms,
#         "state": user_state,
#         "city": user_city,
#         "department": department,
#         "emergency": emergency,
#         "top_k": 3,
#     })
#     cache_set(key, result, ttl=TOOL_CACHE_TTL)
#     return result


# _TOOL_DISPATCH = {
#     "medicine": _call_medicine,
#     "lab": _call_lab,
#     "first_aid": _call_first_aid,
#     "doctor": _call_doctor,
# }


# @traceable(name="executor_node", run_type="chain")
# def executor_node(state: AgentState) -> dict:
#     """
#     Walks state['plan'] in order and calls each tool directly — no CrewAI
#     Agent/Task/Crew wrapping. The plan already says exactly which tool to
#     call; there's no decision left for an agent to make.

#     Special case: doctor tool needs a location. If it's missing, execution
#     stops and signals the caller (API/UI layer) to ask for state + city,
#     rather than silently skipping the doctor step.
#     """
#     plan = state.get("plan", [])
#     logger.info("executor_node — plan: %s", plan)

#     tool_outputs = dict(state.get("tool_outputs", {}))
#     messages = list(state.get("messages", []))

#     for tool_name in plan:
#         if tool_name in tool_outputs:
#             continue  # already have this one (e.g. re-entry after location was provided)

#         handler = _TOOL_DISPATCH.get(tool_name)
#         if handler is None:
#             logger.warning("executor_node — unknown tool in plan: %s", tool_name)
#             continue

#         output = handler(state)

#         if output == NEED_LOCATION:
#             logger.info("executor_node — doctor tool needs location, pausing")
#             messages.append("[Executor] Waiting for user_state/user_city to find a doctor.")
#             return {
#                 "tool_outputs": tool_outputs,
#                 "messages": messages,
#                 "stage": "doctor_need_location",
#             }

#         tool_outputs[tool_name] = _clean(output)
#         messages.append(f"[{tool_name.upper()}]\n{output}")

#     logger.info("executor_node complete — tools run: %s", list(tool_outputs.keys()))

#     return {
#         "tool_outputs": tool_outputs,
#         "messages": messages,
#         "stage": "executed",
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # SYNTHESIS NODE  (replaces response_node — merges N tool outputs into 1 answer)
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="synthesis_node", run_type="chain")
# def synthesis_node(state: AgentState) -> dict:
#     forced_tool = state.get("forced_tool")
#     tool_outputs = state.get("tool_outputs", {})
#     criticality = state.get("criticality", "mild")
#     full_cache_key = _full_cache_key(state)

#     # Button mode: return exactly that tool's output, nothing added, nothing
#     # merged, and no follow-up offer — "just give me medicine" means exactly that.
#     if forced_tool:
#         final = tool_outputs.get(forced_tool, "No result available.")
#         logger.info("synthesis_node — forced_tool mode, raw passthrough for '%s'", forced_tool)
#         cache_set(full_cache_key, json.dumps({"final_response": final}), ttl=FULL_RESPONSE_CACHE_TTL)
#         return {"final_response": final, "stage": "done"}

#     # No tool applied — general/greeting query. Not cached — a generic
#     # fallback prompt isn't worth serving back for a future different query.
#     if not tool_outputs:
#         final = state.get("agent_output") or (
#             "Could you tell me a bit more about your symptoms so I can help?"
#         )
#         return {"final_response": final, "stage": "done"}

#     header = {
#         "critical": "URGENT — please act immediately.",
#         "moderate": "Please consult a doctor soon.",
#         "mild": "Here is the information based on your query.",
#     }.get(criticality, "")

#     if len(tool_outputs) == 1:
#         core = next(iter(tool_outputs.values()))
#     else:
#         combined_raw = "\n\n".join(
#             f"[{name.upper()} RESULT]\n{output}" for name, output in tool_outputs.items()
#         )
#         prompt = f"""
# You are combining the results of multiple medical tools into one clear response
# for a patient. Do not invent new medical facts — only reorganize and connect
# what's already in the tool results below into a single, well-structured answer.
# Keep all dosages, warnings, and disclaimers exactly as given.

# {combined_raw}

# Write one combined response covering all of the above, in a natural order
# (medicine before lab before doctor referral, if all present).
# """
#         response = llm.call(prompt)
#         core = _clean(str(response))

#     final = f"{header}\n\n{core}" if header else core

#     messages = list(state.get("messages", []))
#     messages.append(f"[Synthesis]\n{final}")

#     update = {"final_response": final, "messages": messages, "stage": "done"}

#     # Offer lab test / doctor as a follow-up when this was a symptom-based
#     # query (medicine or first_aid ran) that didn't already cover them.
#     # Skipped for critical cases — doctor referral is already mandatory there.
#     if criticality != "critical":
#         missing = [t for t in ("lab", "doctor") if t not in tool_outputs]
#         symptom_based = any(t in tool_outputs for t in ("medicine", "first_aid"))
#         if missing and symptom_based:
#             offer_text = " and ".join(missing)
#             update["final_response"] += (
#                 f"\n\nWould you also like {offer_text} suggestions? "
#                 f"(reply with 'lab', 'doctor', 'both', or 'no')"
#             )
#             update["follow_up_offered"] = missing

#     # Cache the full response — but only if nothing is still pending
#     # (i.e. we're not mid-follow-up-offer waiting on a doctor location etc).
#     # This is a fresh, complete answer for this exact input combo.
#     cache_set(
#         full_cache_key,
#         json.dumps({
#             "final_response": update["final_response"],
#             "criticality": criticality,
#             "plan": state.get("plan", []),
#         }),
#         ttl=FULL_RESPONSE_CACHE_TTL,
#     )

#     logger.info("synthesis_node complete — tools: %s, follow_up_offered: %s",
#                 list(tool_outputs.keys()), update.get("follow_up_offered"))

#     return update



"""
src/graph/nodes.py
===================
LangGraph nodes. Nodes process state — they do NOT call input().
All user prompting happens in the API layer / UI.

Flow this file supports (wired together in graphs.py):

    welcome -> intake -> criticality -> [critical]-----> executor -> synthesis -> END
                                     -> [not critical] -> planner -> executor -> synthesis -> END

    forced_tool (UI button) set -> planner (bypass mode) -> executor -> synthesis -> END

Removed from the old version: doctor_node / medicine_node / lab_node / intent_node /
response_node (replaced by planner_node + executor_node + synthesis_node), hitl_node
(never wired into a graph — reintroduce later if you actually want human review),
and ~800 lines of commented-out earlier implementations.
"""

# import json
# import re
# import sys
# from pathlib import Path

# from langsmith import traceable

# sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# from src.graph.state import AgentState
# from src.graph.router import detect_criticality, detect_plan
# from src.agents.agents import llm
# from src.cache.redis_cache import make_key, cache_get, cache_set
# from src.tools.medicine_tool import medicine_finder
# from src.tools.lab_tool import lab_test_suggester
# from src.tools.doctor_tool import doctor_finder
# from src.tools.first_aid_tool import first_aid_guide
# from src.logger import get_logger

# logger = get_logger(__name__)

# NEED_LOCATION = "NEED_LOCATION"

# # RAG data doesn't change unless you re-run ingestion, so tool results can
# # be cached for a while. If you re-ingest, call
# # cache_delete_prefix("tool:") to invalidate stale entries.
# TOOL_CACHE_TTL = 60 * 60 * 24 * 7  # 7 days

# # Criticality/triage text are near-deterministic (temp=0.1) — cache them.
# LLM_CACHE_TTL = 60 * 60 * 24  # 1 day

# # Full-response cache — shorter TTL since this is the coarsest-grained
# # cache (a hit skips literally everything, including re-triage).
# FULL_RESPONSE_CACHE_TTL = 60 * 60 * 6  # 6 hours


# def _clean(text: str) -> str:
#     """Strip <think>...</think> blocks that some local models add."""
#     cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
#     return cleaned.strip()


# def _extract_symptoms(raw: str) -> str:
#     """
#     Extracts only the symptom text from the intake string.
#     Input:  'Name: Utsav Raj\\nAge: 5\\nGender: Male\\nSymptoms: Cold from yesterday'
#     Output: 'Cold from yesterday'
#     """
#     match = re.search(r"[Ss]ymptoms?\s*:\s*(.+?)(?:\n|$)", raw)
#     if match:
#         return match.group(1).strip()
#     return raw.strip()


# # ─────────────────────────────────────────────────────────────────────────────
# # WELCOME NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="welcome_node", run_type="chain")
# def welcome_node(state: AgentState) -> dict:
#     logger.info("welcome_node executed")
#     return {"messages": [], "stage": "welcome"}


# # ─────────────────────────────────────────────────────────────────────────────
# # INTAKE NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="intake_node", run_type="chain")
# def intake_node(state: AgentState) -> dict:
#     user_input = state.get("user_input", "")
#     logger.info("intake_node — processing: '%s'", user_input[:80])

#     symptoms_only = _extract_symptoms(user_input) or user_input

#     messages = list(state.get("messages", []))
#     messages.append(f"[Intake] symptoms='{symptoms_only}'")

#     logger.info("intake_node complete — symptoms: '%s'", symptoms_only[:80])

#     return {
#         "symptoms": symptoms_only,
#         "messages": messages,
#         "stage": "intake",
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # CRITICALITY NODE
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="criticality_node", run_type="chain")
# def criticality_node(state: AgentState) -> dict:
#     symptoms = state.get("symptoms", state.get("user_input", ""))
#     logger.info("criticality_node — symptoms: '%s'", symptoms[:80])

#     criticality = detect_criticality(symptoms)  # cached inside router.py
#     logger.info("criticality_node — result: %s", criticality.upper())

#     triage_key = make_key("llm:triage_msg", symptoms=symptoms, criticality=criticality)
#     output = cache_get(triage_key)

#     if not output:
#         prompt = f"""
# You are a warm, empathetic medical triage assistant in an Indian healthcare chatbot.

# Patient symptoms: {symptoms}
# Assessed severity: {criticality.upper()}

# Write a short, clear, empathetic triage message (2-3 sentences):
# - CRITICAL: Tell the patient this is an emergency. Tell them to call 108 or
#   go to the emergency room NOW. First aid steps will be provided separately.
# - MODERATE: Reassure them but advise seeing a doctor soon.
# - MILD: Reassure them it can likely be managed at home.
# """
#         response = llm.call(prompt)
#         output = _clean(str(response))
#         cache_set(triage_key, output, ttl=LLM_CACHE_TTL)

#     messages = list(state.get("messages", []))
#     messages.append(f"[Criticality: {criticality.upper()}]\n{output}")

#     update = {
#         "criticality": criticality.lower(),
#         "agent_output": output,
#         "messages": messages,
#         "stage": "criticality",
#     }

#     # Critical cases skip the planner entirely — the plan is fixed:
#     # first aid guidance immediately, then point to a doctor.
#     if criticality.lower() == "critical":
#         update["plan"] = ["first_aid", "doctor"]

#     return update


# # ─────────────────────────────────────────────────────────────────────────────
# # CACHE CHECK NODE  (NEW — full-response short-circuit)
# # ─────────────────────────────────────────────────────────────────────────────

# def _full_cache_key(state: AgentState) -> str:
#     """
#     Built ONLY from pre-planning inputs (symptoms/age/location/forced_tool) —
#     deliberately excludes 'plan', because plan doesn't exist yet on a fresh
#     call. This also means a resumed call (which DOES carry a plan forward)
#     naturally can't collide with this key even if computed again.
#     """
#     return make_key(
#         "full:response",
#         forced_tool=state.get("forced_tool") or "",
#         symptoms=state.get("symptoms", ""),
#         age=state.get("user_age", ""),
#         user_state=state.get("user_state", ""),
#         user_city=state.get("user_city", ""),
#     )


# @traceable(name="cache_check_node", run_type="chain")
# def cache_check_node(state: AgentState) -> dict:
#     """
#     Runs right after intake. If this is a fresh query (no 'plan' carried in
#     from a prior turn — see the docstring on _full_cache_key) and we've
#     answered this exact symptoms+age+location+forced_tool combo before,
#     skip criticality, planning, tool execution, AND synthesis entirely.

#     A carried-in 'plan' means this is a resume (doctor-location pause, or
#     a follow-up like "yes, also lab") — those must never be short-circuited,
#     since the whole point is to add to what's already been computed.
#     """
#     if state.get("plan"):
#         return {"stage": "cache_checked"}

#     key = _full_cache_key(state)
#     cached = cache_get(key)
#     if cached:
#         try:
#             data = json.loads(cached)
#             logger.info("cache_check_node — full response cache HIT")
#             return {
#                 "final_response": data.get("final_response", ""),
#                 "criticality": data.get("criticality", ""),
#                 "plan": data.get("plan", []),
#                 "stage": "cache_hit",
#             }
#         except (json.JSONDecodeError, TypeError):
#             logger.warning("cache_check_node — cached value unparsable, ignoring")

#     return {"stage": "cache_checked"}


# # ─────────────────────────────────────────────────────────────────────────────
# # PLANNER NODE  (replaces intent_node — produces a list, not a single label)
# # ─────────────────────────────────────────────────────────────────────────────

# @traceable(name="planner_node", run_type="chain")
# def planner_node(state: AgentState) -> dict:
#     """
#     Two modes:
#       1. forced_tool is set (UI button pressed) -> plan is exactly that one tool.
#          No LLM call needed — this is a hard bypass.
#       2. forced_tool is None -> ask the LLM to plan the full tool list for this
#          free-text query (see detect_plan in router.py).
#     """
#     forced_tool = state.get("forced_tool")

#     if forced_tool:
#         logger.info("planner_node — forced_tool='%s', skipping LLM planning", forced_tool)
#         return {"plan": [forced_tool], "stage": "planned_forced"}

#     # Resuming after a doctor_need_location pause — the plan was already
#     # decided last turn (and possibly partly executed). Don't re-plan.
#     if state.get("plan"):
#         logger.info("planner_node — resuming existing plan: %s", state["plan"])
#         return {"plan": state["plan"], "stage": "planned_resumed"}

#     user_input = state.get("user_input", "")
#     symptoms = state.get("symptoms", user_input)

#     plan = detect_plan(user_input=user_input, symptoms=symptoms)
#     logger.info("planner_node — plan: %s", plan)

#     messages = list(state.get("messages", []))
#     messages.append(f"[Plan] {plan}")

#     return {"plan": plan, "messages": messages, "stage": "planned"}


# # ─────────────────────────────────────────────────────────────────────────────
# # EXECUTOR NODE  (replaces doctor_node / medicine_node / lab_node)
# # ─────────────────────────────────────────────────────────────────────────────

# def _call_medicine(state: AgentState) -> str:
#     symptoms = state.get("symptoms", "")
#     age = state.get("user_age")
#     age_val = int(age) if age and str(age).isdigit() else -1

#     key = make_key("tool:medicine", symptoms=symptoms, age=age_val)
#     cached = cache_get(key)
#     if cached is not None:
#         return cached

#     result = medicine_finder.run(**{
#         "symptoms": symptoms,
#         "disease": "",
#         "patient_age": age_val,
#         "top_k": 3,
#     })
#     cache_set(key, result, ttl=TOOL_CACHE_TTL)
#     return result


# def _call_lab(state: AgentState) -> str:
#     symptoms = state.get("symptoms", "")

#     key = make_key("tool:lab", symptoms=symptoms)
#     cached = cache_get(key)
#     if cached is not None:
#         return cached

#     result = lab_test_suggester.run(**{
#         "symptoms": symptoms,
#         "severity": "",
#         "top_k": 3,
#     })
#     cache_set(key, result, ttl=TOOL_CACHE_TTL)
#     return result


# def _call_first_aid(state: AgentState) -> str:
#     symptoms = state.get("symptoms", "")
#     criticality = state.get("criticality", "mild")
#     severity_map = {"critical": "life_threatening", "moderate": "urgent", "mild": "mild"}
#     severity = severity_map.get(criticality, "")

#     key = make_key("tool:first_aid", symptoms=symptoms, severity=severity)
#     cached = cache_get(key)
#     if cached is not None:
#         return cached

#     result = first_aid_guide.run(**{
#         "condition": symptoms,
#         "severity": severity,
#         "top_k": 2,
#     })
#     cache_set(key, result, ttl=TOOL_CACHE_TTL)
#     return result


# def _call_doctor(state: AgentState) -> str:
#     symptoms = state.get("symptoms", "")
#     user_state = state.get("user_state", "")
#     user_city = state.get("user_city", "")
#     criticality = state.get("criticality", "moderate")

#     if not user_state:
#         return NEED_LOCATION

#     department = ""
#     s = symptoms.lower()
#     if "chest pain" in s:
#         department = "Cardiology"
#     elif "fever" in s:
#         department = "General Physician"
#     elif "skin" in s:
#         department = "Dermatology"
#     elif "child" in s or "kid" in s:
#         department = "Pediatrics"
#     elif "pregnant" in s or "pregnancy" in s:
#         department = "Gynecology"

#     emergency = criticality == "critical"
#     key = make_key(
#         "tool:doctor", symptoms=symptoms, state=user_state, city=user_city,
#         department=department, emergency=emergency,
#     )
#     cached = cache_get(key)
#     if cached is not None:
#         return cached

#     result = doctor_finder.run(**{
#         "symptoms": symptoms,
#         "state": user_state,
#         "city": user_city,
#         "department": department,
#         "emergency": emergency,
#         "top_k": 3,
#     })
#     cache_set(key, result, ttl=TOOL_CACHE_TTL)
#     return result


# _TOOL_DISPATCH = {
#     "medicine": _call_medicine,
#     "lab": _call_lab,
#     "first_aid": _call_first_aid,
#     "doctor": _call_doctor,
# }


# @traceable(name="executor_node", run_type="chain")
# def executor_node(state: AgentState) -> dict:
#     """
#     Walks state['plan'] in order and calls each tool directly — no CrewAI
#     Agent/Task/Crew wrapping. The plan already says exactly which tool to
#     call; there's no decision left for an agent to make.

#     Special case: doctor tool needs a location. If it's missing, execution
#     stops and signals the caller (API/UI layer) to ask for state + city,
#     rather than silently skipping the doctor step.
#     """
#     plan = state.get("plan", [])
#     logger.info("executor_node — plan: %s", plan)

#     tool_outputs = dict(state.get("tool_outputs", {}))
#     messages = list(state.get("messages", []))

#     for tool_name in plan:
#         if tool_name in tool_outputs:
#             continue  # already have this one (e.g. re-entry after location was provided)

#         handler = _TOOL_DISPATCH.get(tool_name)
#         if handler is None:
#             logger.warning("executor_node — unknown tool in plan: %s", tool_name)
#             continue

#         output = handler(state)

#         if output == NEED_LOCATION:
#             logger.info("executor_node — doctor tool needs location, pausing")
#             messages.append("[Executor] Waiting for user_state/user_city to find a doctor.")
#             return {
#                 "tool_outputs": tool_outputs,
#                 "messages": messages,
#                 "stage": "doctor_need_location",
#             }

#         tool_outputs[tool_name] = _clean(output)
#         messages.append(f"[{tool_name.upper()}]\n{output}")

#     logger.info("executor_node complete — tools run: %s", list(tool_outputs.keys()))

#     return {
#         "tool_outputs": tool_outputs,
#         "messages": messages,
#         "stage": "executed",
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # SYNTHESIS NODE  (replaces response_node — merges N tool outputs into 1 answer)
# # ─────────────────────────────────────────────────────────────────────────────

# SECTION_TITLES = {
#     "medicine": "MEDICINE",
#     "first_aid": "FIRST AID & HOME CARE",
#     "lab": "LAB TEST RECOMMENDATIONS",
#     "doctor": "DOCTOR / HOSPITAL OPTIONS",
# }
# _SECTION_ORDER = ["medicine", "first_aid", "lab", "doctor"]


# def _merge_tool_outputs_template(tool_outputs: dict) -> str:
#     """
#     Deterministic merge — no LLM call. The tool outputs are already clean,
#     well-formatted text (each with its own headers/dosages/disclaimers);
#     an LLM rewrite was costing 130-190s per call to do something a plain
#     string join achieves in milliseconds with zero information loss (the
#     old prompt explicitly forbade inventing anything anyway — it was only
#     ever reorganizing text that was already correctly formatted).
#     """
#     ordered_names = [n for n in _SECTION_ORDER if n in tool_outputs]
#     ordered_names += [n for n in tool_outputs if n not in _SECTION_ORDER]

#     sections = []
#     for name in ordered_names:
#         title = SECTION_TITLES.get(name, name.upper())
#         sections.append(f"── {title} ──\n{tool_outputs[name].strip()}")

#     return "\n\n".join(sections)


# @traceable(name="synthesis_node", run_type="chain")
# def synthesis_node(state: AgentState) -> dict:
#     forced_tool = state.get("forced_tool")
#     tool_outputs = state.get("tool_outputs", {})
#     criticality = state.get("criticality", "mild")
#     full_cache_key = _full_cache_key(state)

#     # Button mode: return exactly that tool's output, nothing added, nothing
#     # merged, and no follow-up offer — "just give me medicine" means exactly that.
#     if forced_tool:
#         final = tool_outputs.get(forced_tool, "No result available.")
#         logger.info("synthesis_node — forced_tool mode, raw passthrough for '%s'", forced_tool)
#         cache_set(full_cache_key, json.dumps({"final_response": final}), ttl=FULL_RESPONSE_CACHE_TTL)
#         return {"final_response": final, "stage": "done"}

#     # No tool applied — general/greeting query. Not cached — a generic
#     # fallback prompt isn't worth serving back for a future different query.
#     if not tool_outputs:
#         final = state.get("agent_output") or (
#             "Could you tell me a bit more about your symptoms so I can help?"
#         )
#         return {"final_response": final, "stage": "done"}

#     header = {
#         "critical": "URGENT — please act immediately.",
#         "moderate": "Please consult a doctor soon.",
#         "mild": "Here is the information based on your query.",
#     }.get(criticality, "")

#     if len(tool_outputs) == 1:
#         core = next(iter(tool_outputs.values()))
#     else:
#         core = _merge_tool_outputs_template(tool_outputs)

#     final = f"{header}\n\n{core}" if header else core

#     messages = list(state.get("messages", []))
#     messages.append(f"[Synthesis]\n{final}")

#     update = {"final_response": final, "messages": messages, "stage": "done"}

#     # Offer lab test / doctor as a follow-up when this was a symptom-based
#     # query (medicine or first_aid ran) that didn't already cover them.
#     # Skipped for critical cases — doctor referral is already mandatory there.
#     if criticality != "critical":
#         missing = [t for t in ("lab", "doctor") if t not in tool_outputs]
#         symptom_based = any(t in tool_outputs for t in ("medicine", "first_aid"))
#         if missing and symptom_based:
#             offer_text = " and ".join(missing)
#             update["final_response"] += (
#                 f"\n\nWould you also like {offer_text} suggestions? "
#                 f"(reply with 'lab', 'doctor', 'both', or 'no')"
#             )
#             update["follow_up_offered"] = missing

#     # Cache the full response — but only if nothing is still pending
#     # (i.e. we're not mid-follow-up-offer waiting on a doctor location etc).
#     # This is a fresh, complete answer for this exact input combo.
#     cache_set(
#         full_cache_key,
#         json.dumps({
#             "final_response": update["final_response"],
#             "criticality": criticality,
#             "plan": state.get("plan", []),
#         }),
#         ttl=FULL_RESPONSE_CACHE_TTL,
#     )

#     logger.info("synthesis_node complete — tools: %s, follow_up_offered: %s",
#                 list(tool_outputs.keys()), update.get("follow_up_offered"))

#     return update



"""
src/graph/nodes.py
===================
LangGraph nodes. Nodes process state — they do NOT call input().
All user prompting happens in the API layer / UI.

Flow this file supports (wired together in graphs.py):

    welcome -> intake -> criticality -> [critical]-----> executor -> synthesis -> END
                                     -> [not critical] -> planner -> executor -> synthesis -> END

    forced_tool (UI button) set -> planner (bypass mode) -> executor -> synthesis -> END

Removed from the old version: doctor_node / medicine_node / lab_node / intent_node /
response_node (replaced by planner_node + executor_node + synthesis_node), hitl_node
(never wired into a graph — reintroduce later if you actually want human review),
and ~800 lines of commented-out earlier implementations.
"""

import json
import re
import sys
from pathlib import Path

from langsmith import traceable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graph.state import AgentState
from src.graph.router import detect_criticality, detect_plan
from src.agents.agents import llm
from src.cache.redis_cache import make_key, cache_get, cache_set
from src.tools.medicine_tool import medicine_finder
from src.tools.lab_tool import lab_test_suggester
from src.tools.doctor_tool import doctor_finder
from src.tools.first_aid_tool import first_aid_guide
from src.logger import get_logger

logger = get_logger(__name__)

NEED_LOCATION = "NEED_LOCATION"

# RAG data doesn't change unless you re-run ingestion, so tool results can
# be cached for a while. If you re-ingest, call
# cache_delete_prefix("tool:") to invalidate stale entries.
TOOL_CACHE_TTL = 60 * 60 * 24 * 7  # 7 days

# Criticality/triage text are near-deterministic (temp=0.1) — cache them.
LLM_CACHE_TTL = 60 * 60 * 24  # 1 day

# Full-response cache — shorter TTL since this is the coarsest-grained
# cache (a hit skips literally everything, including re-triage).
FULL_RESPONSE_CACHE_TTL = 60 * 60 * 6  # 6 hours


def _clean(text: str) -> str:
    """Strip <think>...</think> blocks that some local models add."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


_FILLER_PATTERNS = [
    re.compile(r"^\s*(hi+|hello+|hey+)[\s,!.]*", re.IGNORECASE),
    re.compile(r"^\s*i\s*am\s*suffering\s*from\s*", re.IGNORECASE),
    re.compile(r"^\s*i\s*(?:have|had|am having|'m having|am experiencing|'m experiencing|feel|am feeling)\s*", re.IGNORECASE),
    re.compile(r"^\s*please\s*help(?:\s*me)?[\s,.]*", re.IGNORECASE),
    re.compile(r"^\s*my\s*symptoms?\s*(?:are|is)\s*", re.IGNORECASE),
]


def _strip_filler(text: str) -> str:
    """
    Best-effort cleanup for casual chat phrasing — repeatedly strips leading
    greeting/filler phrases until nothing more matches, so multi-part openers
    like 'Hi, I am suffering from cold' get stripped in full, not just the
    first pattern that hits. Not real NLU — if you need this to be robust
    against more varied phrasing later, an LLM extraction call (cached, like
    detect_plan) would generalize much better than adding more regexes here.
    """
    cleaned = text.strip()
    changed = True
    while changed:
        changed = False
        for pattern in _FILLER_PATTERNS:
            new = pattern.sub("", cleaned)
            if new != cleaned:
                cleaned = new.strip()
                changed = True
    return cleaned or text.strip()


def _extract_symptoms(raw: str) -> str:
    """
    Extracts only the symptom text from the intake string.
    Structured CLI format:  'Name: X\\nAge: Y\\nSymptoms: cold from yesterday'
                            -> 'cold from yesterday'
    Casual chat format:     'Hi I am suffering from cold,cough and fever'
                            -> 'cold,cough and fever'
    """
    match = re.search(r"[Ss]ymptoms?\s*:\s*(.+?)(?:\n|$)", raw)
    if match:
        return match.group(1).strip()
    return _strip_filler(raw)


# ─────────────────────────────────────────────────────────────────────────────
# WELCOME NODE
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="welcome_node", run_type="chain")
def welcome_node(state: AgentState) -> dict:
    logger.info("welcome_node executed")
    return {"messages": [], "stage": "welcome"}


# ─────────────────────────────────────────────────────────────────────────────
# INTAKE NODE
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="intake_node", run_type="chain")
def intake_node(state: AgentState) -> dict:
    user_input = state.get("user_input", "")
    logger.info("intake_node — processing: '%s'", user_input[:80])

    symptoms_only = _extract_symptoms(user_input) or user_input

    messages = list(state.get("messages", []))
    messages.append(f"[Intake] symptoms='{symptoms_only}'")

    logger.info("intake_node complete — symptoms: '%s'", symptoms_only[:80])

    return {
        "symptoms": symptoms_only,
        "messages": messages,
        "stage": "intake",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CRITICALITY NODE
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="criticality_node", run_type="chain")
def criticality_node(state: AgentState) -> dict:
    symptoms = state.get("symptoms", state.get("user_input", ""))
    logger.info("criticality_node — symptoms: '%s'", symptoms[:80])

    criticality = detect_criticality(symptoms)  # cached inside router.py
    logger.info("criticality_node — result: %s", criticality.upper())

    triage_key = make_key("llm:triage_msg", symptoms=symptoms, criticality=criticality)
    output = cache_get(triage_key)

    if not output:
        prompt = f"""
You are a warm, empathetic medical triage assistant in an Indian healthcare chatbot.

Patient symptoms: {symptoms}
Assessed severity: {criticality.upper()}

Write a short, clear, empathetic triage message (2-3 sentences):
- CRITICAL: Tell the patient this is an emergency. Tell them to call 108 or
  go to the emergency room NOW. First aid steps will be provided separately.
- MODERATE: Reassure them but advise seeing a doctor soon.
- MILD: Reassure them it can likely be managed at home.
"""
        response = llm.call(prompt)
        output = _clean(str(response))
        cache_set(triage_key, output, ttl=LLM_CACHE_TTL)

    messages = list(state.get("messages", []))
    messages.append(f"[Criticality: {criticality.upper()}]\n{output}")

    update = {
        "criticality": criticality.lower(),
        "agent_output": output,
        "messages": messages,
        "stage": "criticality",
    }

    # Critical cases skip the planner entirely — the plan is fixed:
    # first aid guidance immediately, then point to a doctor.
    if criticality.lower() == "critical":
        update["plan"] = ["first_aid", "doctor"]

    return update


# ─────────────────────────────────────────────────────────────────────────────
# CACHE CHECK NODE  (NEW — full-response short-circuit)
# ─────────────────────────────────────────────────────────────────────────────

def _full_cache_key(state: AgentState) -> str:
    """
    Built ONLY from pre-planning inputs (symptoms/age/location/forced_tool) —
    deliberately excludes 'plan', because plan doesn't exist yet on a fresh
    call. This also means a resumed call (which DOES carry a plan forward)
    naturally can't collide with this key even if computed again.
    """
    return make_key(
        "full:response",
        forced_tool=state.get("forced_tool") or "",
        symptoms=state.get("symptoms", ""),
        age=state.get("user_age", ""),
        user_state=state.get("user_state", ""),
        user_city=state.get("user_city", ""),
    )


@traceable(name="cache_check_node", run_type="chain")
def cache_check_node(state: AgentState) -> dict:
    """
    Runs right after intake. If this is a fresh query (no 'plan' carried in
    from a prior turn — see the docstring on _full_cache_key) and we've
    answered this exact symptoms+age+location+forced_tool combo before,
    skip criticality, planning, tool execution, AND synthesis entirely.

    A carried-in 'plan' means this is a resume (doctor-location pause, or
    a follow-up like "yes, also lab") — those must never be short-circuited,
    since the whole point is to add to what's already been computed.
    """
    if state.get("plan"):
        return {"stage": "cache_checked"}

    key = _full_cache_key(state)
    cached = cache_get(key)
    if cached:
        try:
            data = json.loads(cached)
            logger.info("cache_check_node — full response cache HIT")
            return {
                "final_response": data.get("final_response", ""),
                "criticality": data.get("criticality", ""),
                "plan": data.get("plan", []),
                "stage": "cache_hit",
            }
        except (json.JSONDecodeError, TypeError):
            logger.warning("cache_check_node — cached value unparsable, ignoring")

    return {"stage": "cache_checked"}


# ─────────────────────────────────────────────────────────────────────────────
# PLANNER NODE  (replaces intent_node — produces a list, not a single label)
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="planner_node", run_type="chain")
def planner_node(state: AgentState) -> dict:
    """
    Two modes:
      1. forced_tool is set (UI button pressed) -> plan is exactly that one tool.
         No LLM call needed — this is a hard bypass.
      2. forced_tool is None -> ask the LLM to plan the full tool list for this
         free-text query (see detect_plan in router.py).
    """
    forced_tool = state.get("forced_tool")

    if forced_tool:
        logger.info("planner_node — forced_tool='%s', skipping LLM planning", forced_tool)
        return {"plan": [forced_tool], "stage": "planned_forced"}

    # Resuming after a doctor_need_location pause — the plan was already
    # decided last turn (and possibly partly executed). Don't re-plan.
    if state.get("plan"):
        logger.info("planner_node — resuming existing plan: %s", state["plan"])
        return {"plan": state["plan"], "stage": "planned_resumed"}

    user_input = state.get("user_input", "")
    symptoms = state.get("symptoms", user_input)

    plan = detect_plan(user_input=user_input, symptoms=symptoms)
    logger.info("planner_node — plan: %s", plan)

    messages = list(state.get("messages", []))
    messages.append(f"[Plan] {plan}")

    return {"plan": plan, "messages": messages, "stage": "planned"}


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTOR NODE  (replaces doctor_node / medicine_node / lab_node)
# ─────────────────────────────────────────────────────────────────────────────

def _call_medicine(state: AgentState) -> str:
    symptoms = state.get("symptoms", "")
    age = state.get("user_age")
    age_val = int(age) if age and str(age).isdigit() else -1

    key = make_key("tool:medicine", symptoms=symptoms, age=age_val)
    cached = cache_get(key)
    if cached is not None:
        return cached

    result = medicine_finder.run(**{
        "symptoms": symptoms,
        "disease": "",
        "patient_age": age_val,
        "top_k": 3,
    })
    cache_set(key, result, ttl=TOOL_CACHE_TTL)
    return result


def _call_lab(state: AgentState) -> str:
    symptoms = state.get("symptoms", "")

    key = make_key("tool:lab", symptoms=symptoms)
    cached = cache_get(key)
    if cached is not None:
        return cached

    result = lab_test_suggester.run(**{
        "symptoms": symptoms,
        "severity": "",
        "top_k": 3,
    })
    cache_set(key, result, ttl=TOOL_CACHE_TTL)
    return result


def _call_first_aid(state: AgentState) -> str:
    symptoms = state.get("symptoms", "")
    criticality = state.get("criticality", "mild")
    severity_map = {"critical": "life_threatening", "moderate": "urgent", "mild": "mild"}
    severity = severity_map.get(criticality, "")

    key = make_key("tool:first_aid", symptoms=symptoms, severity=severity)
    cached = cache_get(key)
    if cached is not None:
        return cached

    result = first_aid_guide.run(**{
        "condition": symptoms,
        "severity": severity,
        "top_k": 2,
    })
    cache_set(key, result, ttl=TOOL_CACHE_TTL)
    return result


def _call_doctor(state: AgentState) -> str:
    symptoms = state.get("symptoms", "")
    user_state = state.get("user_state", "")
    user_city = state.get("user_city", "")
    criticality = state.get("criticality", "moderate")

    if not user_state:
        return NEED_LOCATION

    department = ""
    s = symptoms.lower()
    if "chest pain" in s:
        department = "Cardiology"
    elif "fever" in s:
        department = "General Physician"
    elif "skin" in s:
        department = "Dermatology"
    elif "child" in s or "kid" in s:
        department = "Pediatrics"
    elif "pregnant" in s or "pregnancy" in s:
        department = "Gynecology"

    emergency = criticality == "critical"
    key = make_key(
        "tool:doctor", symptoms=symptoms, state=user_state, city=user_city,
        department=department, emergency=emergency,
    )
    cached = cache_get(key)
    if cached is not None:
        return cached

    result = doctor_finder.run(**{
        "symptoms": symptoms,
        "state": user_state,
        "city": user_city,
        "department": department,
        "emergency": emergency,
        "top_k": 3,
    })
    cache_set(key, result, ttl=TOOL_CACHE_TTL)
    return result


_TOOL_DISPATCH = {
    "medicine": _call_medicine,
    "lab": _call_lab,
    "first_aid": _call_first_aid,
    "doctor": _call_doctor,
}


@traceable(name="executor_node", run_type="chain")
def executor_node(state: AgentState) -> dict:
    """
    Walks state['plan'] in order and calls each tool directly — no CrewAI
    Agent/Task/Crew wrapping. The plan already says exactly which tool to
    call; there's no decision left for an agent to make.

    Special case: doctor tool needs a location. If it's missing, execution
    stops and signals the caller (API/UI layer) to ask for state + city,
    rather than silently skipping the doctor step.
    """
    plan = state.get("plan", [])
    logger.info("executor_node — plan: %s", plan)

    tool_outputs = dict(state.get("tool_outputs", {}))
    messages = list(state.get("messages", []))

    for tool_name in plan:
        if tool_name in tool_outputs:
            continue  # already have this one (e.g. re-entry after location was provided)

        handler = _TOOL_DISPATCH.get(tool_name)
        if handler is None:
            logger.warning("executor_node — unknown tool in plan: %s", tool_name)
            continue

        output = handler(state)

        if output == NEED_LOCATION:
            logger.info("executor_node — doctor tool needs location, pausing")
            messages.append("[Executor] Waiting for user_state/user_city to find a doctor.")
            return {
                "tool_outputs": tool_outputs,
                "messages": messages,
                "stage": "doctor_need_location",
            }

        tool_outputs[tool_name] = _clean(output)
        messages.append(f"[{tool_name.upper()}]\n{output}")

    logger.info("executor_node complete — tools run: %s", list(tool_outputs.keys()))

    return {
        "tool_outputs": tool_outputs,
        "messages": messages,
        "stage": "executed",
    }


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHESIS NODE  (replaces response_node — merges N tool outputs into 1 answer)
# ─────────────────────────────────────────────────────────────────────────────

SECTION_TITLES = {
    "medicine": "MEDICINE",
    "first_aid": "FIRST AID & HOME CARE",
    "lab": "LAB TEST RECOMMENDATIONS",
    "doctor": "DOCTOR / HOSPITAL OPTIONS",
}
_SECTION_ORDER = ["medicine", "first_aid", "lab", "doctor"]


def _merge_tool_outputs_template(tool_outputs: dict) -> str:
    """
    Deterministic merge — no LLM call. The tool outputs are already clean,
    well-formatted text (each with its own headers/dosages/disclaimers);
    an LLM rewrite was costing 130-190s per call to do something a plain
    string join achieves in milliseconds with zero information loss (the
    old prompt explicitly forbade inventing anything anyway — it was only
    ever reorganizing text that was already correctly formatted).
    """
    ordered_names = [n for n in _SECTION_ORDER if n in tool_outputs]
    ordered_names += [n for n in tool_outputs if n not in _SECTION_ORDER]

    sections = []
    for name in ordered_names:
        title = SECTION_TITLES.get(name, name.upper())
        sections.append(f"── {title} ──\n{tool_outputs[name].strip()}")

    return "\n\n".join(sections)


@traceable(name="synthesis_node", run_type="chain")
def synthesis_node(state: AgentState) -> dict:
    forced_tool = state.get("forced_tool")
    tool_outputs = state.get("tool_outputs", {})
    criticality = state.get("criticality", "mild")
    full_cache_key = _full_cache_key(state)
    delivered = set(state.get("delivered_tools", []))

    # Button mode: return exactly that tool's output, nothing added, nothing
    # merged, and no follow-up offer — "just give me medicine" means exactly that.
    if forced_tool:
        final = tool_outputs.get(forced_tool, "No result available.")
        logger.info("synthesis_node — forced_tool mode, raw passthrough for '%s'", forced_tool)
        cache_set(full_cache_key, json.dumps({"final_response": final}), ttl=FULL_RESPONSE_CACHE_TTL)
        return {"final_response": final, "stage": "done", "delivered_tools": [forced_tool]}

    # No tool applied — general/greeting query. Not cached — a generic
    # fallback prompt isn't worth serving back for a future different query.
    if not tool_outputs:
        final = state.get("agent_output") or (
            "Could you tell me a bit more about your symptoms so I can help?"
        )
        return {"final_response": final, "stage": "done"}

    header = {
        "critical": "URGENT — please act immediately.",
        "moderate": "Please consult a doctor soon.",
        "mild": "Here is the information based on your query.",
    }.get(criticality, "")

    def _render(outputs: dict) -> str:
        if len(outputs) == 1:
            return next(iter(outputs.values()))
        return _merge_tool_outputs_template(outputs)

    # What's NEW since the last time synthesis ran (i.e. not already shown
    # in an earlier chat message) — this is what actually gets displayed.
    new_names = [n for n in tool_outputs if n not in delivered] or list(tool_outputs.keys())
    display_core = _render({n: tool_outputs[n] for n in new_names})
    final_display = f"{header}\n\n{display_core}" if header else display_core

    # The FULL picture (all tools so far) — this is what gets cached, so a
    # future fresh query for the same symptoms gets the complete answer,
    # not just whatever happened to be the last delivered slice.
    full_core = _render(tool_outputs)
    final_full = f"{header}\n\n{full_core}" if header else full_core

    messages = list(state.get("messages", []))
    messages.append(f"[Synthesis]\n{final_display}")

    update = {
        "final_response": final_display,
        "messages": messages,
        "stage": "done",
        "delivered_tools": list(tool_outputs.keys()),
    }

    # Offer lab test / doctor as a follow-up when this was a symptom-based
    # query (medicine or first_aid ran) that didn't already cover them.
    # Skipped for critical cases — doctor referral is already mandatory there.
    if criticality != "critical":
        missing = [t for t in ("lab", "doctor") if t not in tool_outputs]
        symptom_based = any(t in tool_outputs for t in ("medicine", "first_aid"))
        if missing and symptom_based:
            offer_text = " and ".join(missing)
            offer_line = (
                f"\n\nWould you also like {offer_text} suggestions? "
                f"(reply with 'lab', 'doctor', 'both', or 'no')"
            )
            update["final_response"] += offer_line
            final_full += offer_line
            update["follow_up_offered"] = missing

    cache_set(
        full_cache_key,
        json.dumps({
            "final_response": final_full,
            "criticality": criticality,
            "plan": state.get("plan", []),
        }),
        ttl=FULL_RESPONSE_CACHE_TTL,
    )

    logger.info("synthesis_node complete — tools: %s, new this turn: %s, follow_up_offered: %s",
                list(tool_outputs.keys()), new_names, update.get("follow_up_offered"))

    return update