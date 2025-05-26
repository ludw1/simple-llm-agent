# LLM Agent with Multi-Tool Capabilities

This project demonstrates an AI agent built using LangGraph with comprehensive tool integration. The agent combines file reading, web scraping, and web search capabilities to provide intelligent responses for various use cases, particularly **automated job application creation and career assistance**. The goal is to pass a LinkedIn job posting to the agent, have it read the website, ingest various webpages and documents of my own writing and output material which I can use to apply to the job. However, it is mostly an exercise to get acquainted with LangGraph and agentic LLM usage.

*   Implementation of a stateful agent using LangGraph with 2 different LLMs
*   Integration of multiple LLM tools (`file_read`, `web_scrape`, `web_search`)
*   Support for multiple LLM providers (Ollama, OpenRouter, OpenAI)
*   PDF text extraction using `pymupdf4llm`
*   Web search integration via Serper.dev API

## Features

### Core Capabilities
*   **Job Application Automation**: Complete workflow from research to document creation
*   **Interactive Chat Interface**: Persistent conversation with context management
*   **File Operations**: Read text files and extract content from PDFs
*   **Web Scraping**: Extract clean text content from any URL
*   **Web Search**: Google search integration for current information
*   **Smart Tool Selection**: Automatic tool selection based on query context

### Architecture Highlights
*   **Dual LLM System**: Separate models for tool orchestration and content generation
*   **Specialized Prompts**: Job application-focused system prompts for both LLMs
*   **State Management**: LangGraph-based state handling with conversation history
*   **Error Handling**: Robust error handling with graceful fallbacks
*   **Extensible Design**: Easy to add new tools and capabilities

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd simple-llm-agent
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

4.  **Configure API Keys:**
    Create a `.env` file in the root directory with the following keys:
    ```env
    # LLM Provider API Keys (choose one or more)
    OPENROUTER_API_KEY=your_openrouter_api_key_here
    OPENAI_API_KEY=your_openai_api_key_here

    # Web Search API Key (required for search functionality)
    SERPER_API_KEY=your_serper_api_key_here
    ```

5.  **Get API Keys:**
    *   **OpenRouter**: Visit [openrouter.ai](https://openrouter.ai/) for access to multiple models
    *   **OpenAI**: Get your key from [platform.openai.com](https://platform.openai.com/)
    *   **Serper**: Get 2,500 free searches/month at [serper.dev](https://serper.dev/)

## Usage

### Job Application Automation Examples

**Complete Job Application Workflow:**
```
"I'm applying for this job [LinkedIn Url / Company job posting]. Please research their company culture, analyze typical requirements for this role, read my resume from resume.pdf, and create a tailored cover letter that matches my writing style from my_writing_samples.txt"
```

**Company Research:**
```
"Research Spotify's engineering culture, recent technology initiatives, and what they value in software engineers"
```

**Resume Optimization:**
```
"Read my current resume from resume.pdf and research Machine Learning Engineer requirements at Google. Suggest specific improvements to better match their expectations."
```

**Writing Style Analysis:**
```
"Analyze my writing style from cover_letter_samples.txt and create a new cover letter for a DevOps role at AWS that demonstrates cultural fit"
```

## Tools Available

### 1. `file_read(file_path: str)`
*   Reads resumes, cover letters, writing samples, and portfolios
*   Supports text files and PDF extraction
*   Analyzes writing style and professional voice

### 2. `web_scrape(url: str)`
*   Extracts detailed information from job postings
*   Scrapes company about pages and team information
*   Processes career pages and company blogs

### 3. `web_search(query: str)`
*   Google search via Serper.dev API for company research
*   Industry trend analysis and salary data
*   Competitive intelligence and market insights

## Enhanced AI Prompts

This project features **specialized prompts** optimized for job application automation:

### Tool Orchestrator (Research Agent)
*   **Job Application Research Specialist**: Systematically gathers company intel, role requirements, industry context, and candidate profiles
*   **Strategic Tool Selection**: Knows when to use web search vs web scraping vs file reading
*   **Comprehensive Data Collection**: Ensures all relevant information is gathered before creating materials

### Writer Agent (Application Specialist)  
*   **Expert Job Application Writer**: Specialized in resumes, cover letters, and professional communication
*   **Writing Style Analysis**: Analyzes candidate's voice and adapts to maintain authenticity
*   **ATS Optimization**: Creates documents that pass Applicant Tracking Systems
*   **Industry Knowledge**: Understands modern hiring practices and recruiter preferences

## Configuration

### LLM Configuration
Current optimized setup for job applications:
```python
# Tool LLM (Research & Analysis)
TOOL_PROVIDER = "openrouter"
TOOL_MODEL = "google/gemini-2.5-flash-preview"

# Writer LLM (Content Creation)
WRITER_PROVIDER = "openrouter" 
WRITER_MODEL = "openai/gpt-4o-mini"
```

### Tool Limits
*   Maximum 10 tool calls per user turn (configurable)
*   Web search results limited to 8,000 characters
*   Web scraping limited to 10,000 characters
