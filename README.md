# Agentic AI Framework – Research & Test Automation

## 📌 Overview
This project is an **Agentic AI Framework** built as an office POC.  
It uses **LangGraph, LangChain, and Ollama** to coordinate multiple specialized agents for:
- Log analysis
- Test case generation
- Test script generation
- Long-term memory retrieval & storage

The system provides a **Streamlit-based UI** for interaction and coordination of research workflows.

---

## 🧩 Features
- 🤖 **Research Supervisor Agent** – delegates tasks to sub-agents.
- 📜 **Log Analysis Agent** – analyzes logs & PCAPs for errors.
- 🧪 **Test Case Agent** – generates structured test cases.
- 📝 **Test Script Agent** – produces executable automation scripts.
- 🧠 **Memory Agent** – saves & retrieves findings across sessions.
- 🔗 **Long-term memory (Chroma + HuggingFace embeddings)** for efficient retrieval.
- 📂 **Dynamic tool discovery & registry** for extensibility.
- 🎛️ **Streamlit UI** for running workflows interactively.

---

## ⚙️ Project Structure

├── agents/ # Agent definitions (Supervisor, Sub-agents)
│ ├── a2a_factory.py
│ ├── a2a_system.py
│ ├── agent_executor.py
│ └── agent_cards.json
│
├── config/ # Configurations & prompts
│ ├── prompts.yml
│ ├── memory_config.py
│ ├── config_paths.py
│ └── log_patterns.py
│
├── graph/ # Research graph orchestration
│ ├── research_graph.py
│ └── research_state.py
│
├── tools/ # Tooling (parser, memory, registry)
│ ├── parser.py
│ ├── memory_tools.py
│ └── registry.py
│
├── main2.py # Streamlit entrypoint
├── requirements.txt # Dependencies
└── README.md


---

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
