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

```bash
📦 Project Root
├── agents/                  # Agent definitions (Supervisor & Sub-agents)
│   ├── a2a_factory.py
│   ├── a2a_system.py
│   ├── agent_executor.py
│   └── agent_cards.json
│
├── config/                  # Configuration files & prompt settings
│   ├── prompts.yml
│   ├── memory_config.py
│   ├── config_paths.py
│   └── log_patterns.py
│
├── graph/                   # Research graph orchestration modules
│   ├── research_graph.py
│   └── research_state.py
│
├── tools/                   # Utilities (parser, memory, registry, etc.)
│   ├── parser.py
│   ├── memory_tools.py
│   └── registry.py
│
├── main2.py                 # Streamlit application entry point
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
```

---

Here’s an updated version of your `README.md` with all the requested additions: OS-specific setup instructions, Docker installation, Python version specification, and first-time run guidance. I’ve preserved your original content and appended the new sections with clear segmentation.

---

## 🛠️ System Requirements

- **Python**: 3.10 or higher (recommended: Python 3.10.12)
- **Operating Systems**: Windows, Linux, macOS
- **Memory**: Minimum 8GB RAM
- **Disk**: At least 2GB free space
- **Ollama**: Required for LLM backend (see [Ollama installation](https://ollama.com))

---

## 🧰 Installation Instructions

### 🔧 2. Python Environment Setup

#### Windows / Linux / macOS

1. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   ```

2. **Activate the environment**

   - **Windows**
     ```bash
     .\venv\Scripts\activate
     ```

   - **Linux / macOS**
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Ollama is running**
   - Download and install Ollama from [https://ollama.com](https://ollama.com)
   - Ensure the model `llama3.1:8b` is available:
     ```bash
     ollama run llama3.1:8b
     ```

---

### 🐳 3. Installation via Docker

> Ensure Docker is installed and running on your system.

1. **Build the Docker image**
   ```bash
   docker build -t agentic-ai .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 agentic-ai
   ```

3. **Access the UI**
   - Open your browser and go to: [http://localhost:8501](http://localhost:8501)

---

### 🚦 4. First-Time Setup & Run

1. **Activate your virtual environment** (see above)
2. **Start the Streamlit UI**
   ```bash
   streamlit run main2.py
   ```
3. **Upload your log or PCAP file**
4. **Select agents and run coordination**

---

## 📦 Release Notes

### 🔄 v1.1.0 – October 2025

- ✅ Refactored `agent_executor.py` for better modularity
- 🧠 Added `review_agent_work()` method to supervisor agent
- 🧪 Improved tool execution flow and error handling
- 🧰 Enhanced agent card definitions with richer metadata
- 🐛 Fixed edge cases in JSON extraction logic
- 📦 Updated `requirements.txt` for compatibility with LangGraph v0.2+

---

Let me know if you’d like help tagging this release in GitHub or generating a changelog file!


