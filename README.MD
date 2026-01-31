# Simple Math Agent (OpenAI Function Calling)

A foundational project exploring **Agentic Workflows**. This project demonstrates how to build an AI Agent from scratch that can recognize when it needs external tools, execute Python functions, and return the results to the LLM.

## 🚀 About the Project
Large Language Models (LLMs) are great at language but can struggle with precise calculations or specific logic. This project solves that by giving the LLM "hands" (Tools).

When the user asks a complex math question, the Agent:
1.  **Reasons** that it cannot answer directly.
2.  **Selects** the appropriate tool (`multiply` or `divide`) from its schema.
3.  **Pauses** to let the Python script execute the logic.
4.  **Integrates** the precise result into a natural language response.

## 🛠️ Tech Stack
* **Python 3.x**
* **OpenAI API** (Function Calling capability)
* **python-dotenv** (Environment security)

## 📂 Project Structure
* `main.py`: The core logic containing the tool definitions, schema, and execution loop.
* `requirements.txt`: Lists all dependencies to ensure reproducibility.
* `.env`: Stores the API key securely (not shared in repo).
* `.gitignore`: Ensures sensitive files are not pushed to GitHub.

## ⚡ Setup & Usage

1.  **Clone the repository:**

2.  **Create a Virtual Environment:**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**
    * Create a file named `.env` in the root folder.
    * Add your key: `OPENAI_API_KEY=sk-your-key-here`

5.  **Run the Agent:**
    ```bash
    python main.py
    ```

## 🧠 What I Learned
* How to define JSON Schemas for OpenAI Tools.
* How to parse `tool_calls` from an API response.
* Implementing a complete execution loop (LLM -> Code -> LLM).