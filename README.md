🏥 Agentic Doctor System

An AI-powered medical assistant built using a graph-based workflow (LangGraph) that helps users with:

🧠 Symptom understanding
⚠️ Criticality detection (Mild / Moderate / Critical)
💊 Medicine suggestions
🧪 Lab test recommendations
👨‍⚕️ Doctor discovery based on location
🩹 First-aid guidance (with Indian home remedies)
🚀 Features
1. 🧾 Patient Intake
Collects user details:
Name
Age
Gender
Symptoms
Structures the input using an LLM agent
2. ⚠️ Criticality Detection
Classifies condition into:
Critical
Moderate
Mild
Provides immediate triage response
3. 🔀 Intelligent Workflow Routing
If Critical:

➡️ Ask for location (State + City)
➡️ Recommend nearby doctors/hospitals immediately

If Not Critical:

➡️ Ask user intent:

Medicine 💊
Lab Test 🧪
Doctor 👨‍⚕️
4. 💊 Medicine Recommendation
Uses medicine_finder tool
Provides:
Medicines
Dosage (age-aware)
Side effects
Warnings
5. 🧪 Lab Test Suggestions
Suggests relevant diagnostic tests based on symptoms
6. 👨‍⚕️ Doctor Finder
Uses doctor_finder tool
Requires:
State (mandatory)
City (optional but recommended)
Returns:
Doctor name
Hospital
Contact details
Availability
7. 🩹 First Aid Guidance
Immediate steps for handling symptoms
Includes Indian home remedies (Tulsi, Haldi doodh, etc.)
🧠 Architecture

Built using a graph-based workflow:

welcome → intake → criticality
                ↓
        ┌───────────────┐
        │               │
   critical         not critical
        │               │
    doctor         intent node
                       ↓
            medicine / lab / doctor
📁 Project Structure
src/
├── graph/
│   ├── graphs.py        # Graph builder
│   ├── nodes.py         # All nodes (welcome, intake, etc.)
│   ├── router.py        # Intent & criticality detection
│
├── agents/
│   ├── agents.py        # LLM agents
│
├── tools/
│   ├── medicine_finder
│   ├── doctor_finder
│   ├── first_aid_guide
│


→ Medicine suggestions displayed
⚙️ Tech Stack
LangGraph – Workflow orchestration
LLMs (Ollama / LLaMA3) – Reasoning & generation
CrewAI – Agent execution
RAG (Retrieval-Augmented Generation) – Medical knowledge
Python – Core backend
🐢 Known Issues
⏳ Slow response time (due to local LLM - Ollama)
🔁 Re-routing issues if state not maintained properly
🧩 Tool argument validation errors (fixed via structured prompts)
🔧 Future Improvements
⚡ Replace local LLM with faster API (OpenAI / Groq)
🧠 Better symptom extraction (avoid “Cold from yesterday” issue)
💬 Add conversation memory
🌐 Build UI (Streamlit / React)
📍 Auto-detect location (instead of manual input)