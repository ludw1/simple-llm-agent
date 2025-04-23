# Simple LLM Agent with File Reading Tool

This project demonstrates a simple AI agent built using LangChain and LangGraph. The agent can interact with the user, list files in the current directory structure, and read the content of specified files (text and PDF) using a tool.

This project serves as a portfolio piece showcasing:
*   Implementation of a stateful agent using LangGraph.
*   Integration of LLM tools (`file_read`).
*   Support for multiple LLM providers (Ollama, OpenRouter, OpenAI).
*   Configuration management using a `.env` file for API keys.
*   Basic PDF text extraction using `pymupdf4llm`.

## Features

*   Interactive chat loop.
*   Automatic listing of files in the project directory (excluding `.git`, `.venv`, etc.).
*   `file_read` tool to fetch content from text or PDF files.
*   Configurable LLM provider and model name.
*   API key management via `.env` file.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <repository-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # Activate the environment (Windows)
    .\.venv\Scripts\activate
    # Activate the environment (macOS/Linux)
    source .venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Create a `.env` file:**
    Create a file named `.env` in the root directory of the project.
5.  **Add API Keys to `.env`:**
    Add the necessary API keys to your `.env` file. You only need the key for the provider you intend to use (`openrouter` or `openai`).
    ```env
    OPENROUTER_API_KEY=your_openrouter_api_key_here
    OPENAI_API_KEY=your_openai_api_key_here
    ```
    **Important:** Add `.env` to your `.gitignore` file to avoid committing secrets.

## Usage

1.  **Configure the Agent (Optional):**
    Open `main.py` and modify the `PROVIDER` and `MODEL` variables near the bottom of the file to select your desired LLM provider and model name.
    ```python
    # Example usage:
    PROVIDER = "openrouter" # or "ollama" or "openai"
    MODEL = "google/gemini-2.0-flash-exp:free" # Adjust model name as needed
    ```
2.  **Run the script:**
    ```bash
    python main.py
    ```
3.  **Interact with the agent:**
    Enter prompts at the command line. The agent will first list available files and then respond to your requests, potentially using the `file_read` tool if needed.
    Type `quit` to exit.

## Project Structure

*   `main.py`: The main script containing the agent logic, LangGraph definition, and interaction loop.
*   `requirements.txt`: Lists project dependencies.
*   `.env`: Stores API keys (should be gitignored).
*   `README.md`: This file.
*   (Optional) `my_writing/`: Example directory containing files for the agent to read. 