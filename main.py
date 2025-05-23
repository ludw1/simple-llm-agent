import logging
import pymupdf4llm
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import requests
from bs4 import BeautifulSoup
from langchain_community.utilities import GoogleSerperAPIWrapper


class Config:
    """Configuration management for the QwenAgent"""
    
    # Tool execution limits
    MAX_TOOL_CALLS_PER_TURN = 10
    MAX_CHAT_HISTORY = 20
    
    # File size limits
    MAX_FILE_SIZE_MB = 50
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    
    # Output limits
    MAX_SEARCH_RESULT_LENGTH = 8000
    MAX_SCRAPE_CONTENT_LENGTH = 10000
    
    # Encoding options
    DEFAULT_ENCODING = "utf-8"
    FALLBACK_ENCODING = "latin-1"
    
    # Request timeouts
    WEB_REQUEST_TIMEOUT = 10
    
    # Log configuration
    LOG_FILE = 'agent.log'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure Tool LLM (e.g., Gemini Flash for tool calls)
    TOOL_PROVIDER = "openrouter"
    TOOL_MODEL = "google/gemini-2.5-flash-preview"

    # Configure Writer LLM (e.g., Arliai Qwen for final response)
    WRITER_PROVIDER = "openrouter"
    WRITER_MODEL = "openai/gpt-4o-mini"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format=Config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


def validate_environment():
    """Validate that required environment variables are set"""
    required_vars = {
        'OPENROUTER_API_KEY': 'Required for OpenRouter LLM access',
        'SERPER_API_KEY': 'Required for web search functionality'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var}: {description}")
    
    if missing_vars:
        logger.warning("Missing environment variables:")
        for var in missing_vars:
            logger.warning(f"  - {var}")
        print("Warning: Some features may not work without required environment variables.")
        print("Please check your .env file and ensure all required variables are set.")
    else:
        logger.info("All required environment variables are set")


# Validate environment on startup
validate_environment()


# --- Web Search Configuration ---
def get_search_tool():
    """Initialize and return the web search tool."""
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not serper_api_key:
        print("Warning: SERPER_API_KEY not found. Web search will not be available.")
        return None

    try:
        search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
        return search
    except Exception as e:
        print(f"Error initializing web search: {e}")
        return None


# Initialize the search tool
_search_tool = get_search_tool()


@tool
def web_search(query: str) -> str:
    """Search the web for information using Google search. Provide a clear, specific search query."""
    logger.info(f"Tool: Searching web for: {query}")

    if _search_tool is None:
        error_msg = "Error: Web search is not available. Please check that SERPER_API_KEY is set in your environment."
        logger.error(error_msg)
        return error_msg

    try:
        # Use the search tool to get results
        results = _search_tool.run(query)

        # Limit output length to avoid overwhelming context
        max_length = Config.MAX_SEARCH_RESULT_LENGTH
        if len(results) > max_length:
            logger.warning(f"Search results truncated from {len(results)} to {max_length} characters")
            return results[:max_length] + "\n... (results truncated)"

        logger.info(f"Search completed successfully, {len(results)} characters returned")
        return results

    except Exception as e:
        error_msg = f"Error performing web search: {e}"
        logger.error(error_msg)
        return error_msg


def load_file(file_path: str) -> str:
    """Load a file and return its content. Depending on the file type, different methods can be used to read it.
    Args:
        file_path (str): Path to the file.
    Returns:
        str: Content of the file.
    """
    logger.info(f"Attempting to load file: {file_path}")
    
    if not os.path.exists(file_path):
        error_msg = f"Error: File '{file_path}' does not exist."
        logger.error(error_msg)
        return error_msg

    try:
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size} bytes")
        
        # Check for very large files
        if file_size > Config.MAX_FILE_SIZE_BYTES:
            error_msg = f"Error: File '{file_path}' is too large ({file_size} bytes). Maximum size is {Config.MAX_FILE_SIZE_MB}MB."
            logger.error(error_msg)
            return error_msg

        if file_path.lower().endswith(".pdf"):
            try:
                logger.info("Processing PDF file")
                pdf_text = pymupdf4llm.to_markdown(file_path)
                logger.info(f"PDF processed successfully, {len(pdf_text)} characters extracted")
                return pdf_text
            except Exception as e:
                error_msg = f"Error reading PDF {file_path}: {str(e)}"
                logger.error(error_msg)
                return error_msg
        else:
            try:
                with open(file_path, "r", encoding=Config.DEFAULT_ENCODING) as file:
                    content = file.read()
                    logger.info(f"Text file processed successfully, {len(content)} characters read")
                    return content
            except UnicodeDecodeError:
                try:
                    # Try with different encoding
                    with open(file_path, "r", encoding=Config.FALLBACK_ENCODING) as file:
                        content = file.read()
                        logger.info(f"Text file processed with {Config.FALLBACK_ENCODING} encoding, {len(content)} characters read")
                        return content
                except Exception as e:
                    error_msg = f"Error reading {file_path} with multiple encodings: {str(e)}"
                    logger.error(error_msg)
                    return error_msg
            except Exception as e:
                error_msg = f"Error reading {file_path}: {str(e)}"
                logger.error(error_msg)
                return error_msg
    except Exception as e:
        error_msg = f"Unexpected error processing {file_path}: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def file_read(file_path: str) -> str:
    """Fetch content from a specified file. Supports reading text files and extracting text from PDF files. Provide the full file path."""
    print(f"Tool: Reading file: {file_path}")
    return load_file(file_path)


@tool
def web_scrape(url: str) -> str:
    """Fetches and extracts text content from a given URL. Provide the full URL including http/https."""
    print(f"Tool: Scraping web page: {url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }  # Some sites block default requests user-agent
        response = requests.get(url, headers=headers, timeout=Config.WEB_REQUEST_TIMEOUT)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text, strip leading/trailing whitespace, reduce multiple newlines/spaces
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        # Further process lines to handle extra spaces robustly before joining
        chunks = (
            phrase.strip()
            for line in lines
            for phrase in line.split("  ")
            if phrase.strip()
        )  # Filter empty strings after splitting by double space
        text = "\n".join(
            chunk for chunk in chunks if chunk
        )  # Join non-empty chunks with newline

        # Limit the output length to avoid overwhelming the context
        max_length = Config.MAX_SCRAPE_CONTENT_LENGTH
        if len(text) > max_length:
            return (
                text[:max_length] + "\n... (content truncated)"
            )  # Ensure newline in truncation message
        return text

    except requests.exceptions.RequestException as e:
        return f"Error fetching URL {url}: {e}"
    except Exception as e:
        return f"Error processing URL {url}: {e}"


# Make request_files a regular helper function, no longer a tool
def request_files(directory: str = ".") -> str:
    """Recursively lists full paths of files within a directory."""
    print(f"Helper: Recursively listing full file paths in: {directory}")
    file_paths = []

    try:
        abs_directory = os.path.abspath(directory)
        if not os.path.isdir(abs_directory):
            return (
                f"Error: Directory '{directory}' does not exist or is not accessible."
            )

        for root, dirs, files in os.walk(abs_directory, topdown=True):
            # Filter directories to avoid traversing into them
            dirs[:] = [
                d for d in dirs if d not in (".git", "__pycache__", ".vscode", ".venv")
            ]

            # Collect full paths for files in the current directory
            for filename in files:
                # Optional: Filter specific files like .gitignore if needed
                # if filename == '.gitignore':
                #     continue
                if "thesis" in filename or ".env" in filename:
                    continue
                full_path = os.path.join(root, filename)
                file_paths.append(full_path)

        if not file_paths:
            return "No files found in the specified directory."

        # Format the output as a simple list
        return "\nAvailable files:\n" + "\n".join(sorted(file_paths))

    except Exception as e:
        print(f"Error during os.walk in {directory}: {str(e)}")
        return f"Error listing files in directory {directory}: {str(e)}"


# QwenAgent class
class QwenAgent:
    """An agent that uses LangGraph to orchestrate LLM calls and tool usage.

    This agent can:
    - Maintain conversation history.
    - List files in the workspace.
    - Use a `file_read` tool to ingest file content (text/PDF).
    - Use a `web_scrape` tool to fetch content from URLs.
    - Interact with configurable LLM providers (Ollama, OpenRouter, OpenAI).
    - Optionally use separate LLMs for tool usage and final writing.
    """

    # Define the state for the graph
    class AgentState(TypedDict):
        """Represents the state of the agent graph, primarily the message history."""

        messages: Annotated[list, add_messages]
        # Add counter for tool calls within a turn
        tool_calls_this_turn: int

    # Update __init__ for two LLMs
    def __init__(
        self,
        tool_llm_provider: str = "openrouter",
        tool_llm_model: str = "google/gemini-2.0-flash-exp:free",
        writer_llm_provider: str = "openrouter",
        writer_llm_model: str = "arliai/qwq-32b-arliai-rpr-v1:free",
    ):
        """Initializes the QwenAgent with potentially separate tool and writer LLMs.

        Args:
            tool_llm_provider: The provider for the LLM handling tool calls.
            tool_llm_model: The model name for the tool LLM.
            writer_llm_provider: The provider for the LLM handling final writing/synthesis.
            writer_llm_model: The model name for the writer LLM.
        """
        self.chat_history = []
        _tools_list = [file_read, web_scrape, web_search]
        self.tool_map = {tool.name: tool for tool in _tools_list}

        # --- Initialize Tool LLM ---
        print(
            f"Initializing Tool LLM: Provider={tool_llm_provider}, Model={tool_llm_model}"
        )
        self.tool_llm = self._initialize_llm(tool_llm_provider, tool_llm_model)
        # Bind tools ONLY to the tool_llm
        self.tool_llm_with_tools = self.tool_llm.bind_tools(_tools_list)

        # --- Initialize Writer LLM ---
        print(
            f"Initializing Writer LLM: Provider={writer_llm_provider}, Model={writer_llm_model}"
        )
        # Initialize writer LLM WITHOUT binding tools
        self.writer_llm = self._initialize_llm(writer_llm_provider, writer_llm_model)

        # Define the persistent system prompt message content (primarily for tool LLM)
        self.system_prompt_content = (
            "You are an AI Job Application Research Orchestrator. Your mission is to systematically gather and organize information needed for creating tailored job applications, cover letters, and resumes.\n\n"
            
            "**YOUR TOOLS:**\n"
            "1. `file_read(file_path: str)`: Read resumes, cover letters, writing samples, and other documents\n"
            "2. `web_scrape(url: str)`: Extract detailed information from job postings and company pages\n"
            "3. `web_search(query: str)`: Research companies, roles, industry trends, and requirements\n\n"
            
            "**JOB APPLICATION RESEARCH WORKFLOW:**\n"
            "When helping with job applications, systematically gather:\n"
            "• COMPANY INTEL: Culture, values, recent news, mission, team structure\n"
            "• ROLE REQUIREMENTS: Technical skills, soft skills, experience levels, responsibilities\n"
            "• INDUSTRY CONTEXT: Current trends, salary ranges, market conditions\n"
            "• CANDIDATE PROFILE: Resume content, writing style, experience, skills\n\n"
            
            "**TOOL SELECTION STRATEGY:**\n"
            "• Use `web_scrape` for specific URLs (job postings, company about pages, team pages)\n"
            "• Use `web_search` for general research (company culture, salary data, industry trends)\n"
            "• Use `file_read` for candidate materials (resumes, portfolios, writing samples)\n"
            "• Always gather comprehensive information before stopping\n\n"
            
            "**CRITICAL INSTRUCTIONS:**\n"
            "• Prioritize information that helps tailor applications to specific roles and companies\n"
            "• When researching companies, focus on culture, values, and what they value in employees\n"
            "• For job postings, extract both explicit requirements and implicit preferences\n"
            "• Read ALL relevant candidate files to understand their background and writing style\n"
            "• Research current market trends and expectations for the specific role/industry\n"
            "• Your role is information gathering - a specialist writer will create the final materials\n"
            "• Continue gathering until you have comprehensive insights for application tailoring\n"
        )

        # Build and compile the state graph
        self.graph = self._build_graph()

    def _initialize_llm(self, provider: str, model_name: str):
        """Helper function to initialize an LLM client based on provider and model name."""
        if provider == "ollama":
            return ChatOllama(model=model_name, temperature=0.2)
        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY not found in environment or .env file."
                )
            return ChatOpenAI(
                model=model_name,
                temperature=0.2,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
            )
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment or .env file."
                )
            return ChatOpenAI(model=model_name, temperature=0.2, openai_api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    # Define the function that lists the directory content first
    def _list_directory(self, state: AgentState) -> dict:
        """Graph Node: Calls the `request_files` helper and adds the result as a SystemMessage."""
        print("--- Listing Directory (Helper Call) ---")
        directory_to_list = "."  # Always list the current directory

        # Directly call the helper function
        try:
            response = request_files(directory_to_list)
            # Update the context message format
            message_content = f"Context: File listing results - {response}"
        except Exception as e:
            print(f"Error calling request_files helper: {e}")
            message_content = f"Error listing directory: {e}"

        # Create a SystemMessage to add context to the state.
        list_message = SystemMessage(content=message_content)
        print(f"Directory Listing System Message: {list_message}")

        # Return the message to be added to the state
        return {"messages": [list_message]}

    # Define the function that calls the model
    def _call_model(self, state: AgentState) -> dict:
        """Graph Node: Prepares messages and invokes the LLM with tools."""
        print("--- Calling Model ---")
        messages = state["messages"]
        # Create the system message
        system_message = SystemMessage(content=self.system_prompt_content)
        # Prepend the system message to the current messages
        messages_with_system = [system_message] + messages
        print(f"Messages sent to LLM: {messages_with_system}")
        response = self.tool_llm_with_tools.invoke(messages_with_system)
        print(f"Model response: {response}")
        # We return a list, because this will get added to the existing list via add_messages
        # Note: We only return the LLM's response, not the system prompt we added
        return {"messages": [response]}

    # --- New Node: Call Writer LLM ---
    def _call_writer_model(self, state: AgentState) -> dict:
        """Graph Node: Invokes the writer LLM to synthesize the final response."""
        print("--- Calling Writer Model ---")
        # Get all messages except the initial system prompt for the tool_llm
        messages = state["messages"]

        writer_system_prompt = SystemMessage(
            content=(
                "You are an Expert Job Application Writer and Career Strategist. You specialize in creating compelling, tailored resumes, cover letters, and application materials that get results.\n\n"
                
                "**YOUR EXPERTISE:**\n"
                "• Professional writing across all industries and career levels\n"
                "• ATS-optimized resume formatting and keyword integration\n"
                "• Persuasive cover letter structures that tell compelling stories\n"
                "• Personal branding and value proposition development\n"
                "• Writing style analysis and adaptive mimicry\n"
                "• Modern hiring practices and recruiter preferences\n\n"
                
                "**WHEN WRITING JOB APPLICATION MATERIALS:**\n"
                "\n1. **STYLE ANALYSIS**: If writing samples are provided, analyze:\n"
                "   • Tone (formal, conversational, technical, creative)\n"
                "   • Sentence structure and length preferences\n"
                "   • Vocabulary choices and technical language use\n"
                "   • Personality indicators and communication style\n"
                "   • Professional voice and personal brand\n"
                
                "\n2. **CONTENT STRATEGY**: Based on research data, create:\n"
                "   • Value propositions aligned with company needs\n"
                "   • Achievement stories using STAR/CAR methodology\n"
                "   • Keyword optimization for ATS systems\n"
                "   • Cultural fit demonstrations\n"
                "   • Technical skill presentations relevant to role\n"
                
                "\n3. **DOCUMENT FORMATS**:\n"
                "   • RESUMES: Clean, scannable, achievement-focused, keyword-rich\n"
                "   • COVER LETTERS: Hook, story, value, call-to-action structure\n"
                "   • EMAILS: Professional, concise, purpose-driven\n"
                "   • LINKEDIN MESSAGES: Personal, relevant, action-oriented\n\n"
                
                "**QUALITY STANDARDS:**\n"
                "• Every claim must be substantiated with specific examples\n"
                "• Quantify achievements with metrics when possible\n"
                "• Match language and terminology used by the target company\n"
                "• Demonstrate clear understanding of role requirements\n"
                "• Show genuine interest and cultural alignment\n"
                "• Maintain professional tone while reflecting candidate's personality\n"
                "• Ensure ATS compatibility and recruiter appeal\n\n"
                
                "**OUTPUT REQUIREMENTS:**\n"
                "• Provide clear, ready-to-use documents\n"
                "• Include strategic explanations for key choices\n"
                "• Offer specific customization suggestions\n"
                "• Flag any gaps or areas needing candidate input\n"
                "• Suggest follow-up actions and next steps\n\n"
                
                "Use all provided research to create materials that position the candidate as the ideal fit for the specific role and company."
            )
        )
        messages_for_writer = [writer_system_prompt] + messages

        print(f"Messages sent to Writer LLM: {messages_for_writer}")
        # Invoke the writer LLM (without tools)
        response = self.writer_llm.invoke(messages_for_writer)
        print(f"Writer Model response: {response}")
        # Return the final response to be added to the state
        return {"messages": [response]}

    # Define the function to execute tools
    def _call_tool(self, state: AgentState) -> dict:
        """Graph Node: Executes tools called by the LLM in the previous step."""
        logger.info("Executing tool calls")
        last_message = state["messages"][-1]

        # Ensure last_message is an AIMessage with tool_calls
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            logger.warning("No valid tool calls found in the last AI message")
            return {
                "messages": [],
                "tool_calls_this_turn": state.get("tool_calls_this_turn", 0),
            }

        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            logger.info(f"Executing tool: {tool_name} with args {tool_args} (ID: {tool_id})")

            # Look up the tool function in the map
            if tool_name not in self.tool_map:
                error_msg = f"Error: Tool '{tool_name}' not found."
                logger.error(error_msg)
                response_content = error_msg
            else:
                tool_to_call = self.tool_map[tool_name]
                try:
                    # Invoke the tool using the retrieved function object
                    response = tool_to_call.invoke(tool_args)
                    response_content = str(response)
                    logger.info(f"Tool {tool_name} executed successfully")
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {e}"
                    logger.error(error_msg)
                    response_content = error_msg

            # Include the tool name in the ToolMessage for clarity
            tool_messages.append(
                ToolMessage(
                    content=response_content, name=tool_name, tool_call_id=tool_id
                )
            )

        # Increment the counter for this turn
        current_count = state.get("tool_calls_this_turn", 0)
        new_count = current_count + 1
        logger.info(f"Tool call iteration {new_count}/{Config.MAX_TOOL_CALLS_PER_TURN} for this turn")
        
        # We return a list, because this will get added to the existing list (via add_messages)
        # Also return the updated counter
        return {"messages": tool_messages, "tool_calls_this_turn": new_count}

    # Define the function that determines whether to continue or not
    def _should_continue(self, state: AgentState) -> str:
        """Graph Node: Determines the next step based on the last message and tool call limit."""
        print("--- Checking if should continue ---")
        last_message = state["messages"][-1]
        tool_calls_made = state.get("tool_calls_this_turn", 0)

        # Check if the LLM requested tool calls
        has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls

        if has_tool_calls:
            # Check if we've exceeded the limit for this turn
            if tool_calls_made >= Config.MAX_TOOL_CALLS_PER_TURN:
                print(
                    f"Decision: Tool call limit ({Config.MAX_TOOL_CALLS_PER_TURN}) reached. Forcing end of tool loop."
                )
                return "end"
            else:
                # Limit not reached, continue tool execution
                print(
                    f"Decision: Continue tool execution (Iteration {tool_calls_made + 1})"
                )
                return "continue"
        else:
            # No tool calls requested by LLM
            print("Decision: End (No tool calls requested)")
            return "end"

    def _build_graph(self) -> StateGraph:
        """Builds and compiles the LangGraph StateGraph for the agent."""
        # Graph building logic
        graph = StateGraph(self.AgentState)

        # Define the nodes
        graph.add_node("list_directory", self._list_directory)
        graph.add_node("agent", self._call_model)
        graph.add_node("action", self._call_tool)
        graph.add_node("writer", self._call_writer_model)

        # Set the entrypoint
        graph.set_entry_point("list_directory")

        # Add edges
        graph.add_edge("list_directory", "agent")

        # Conditional edges from agent
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "action",  # If tool call needed, go to action
                "end": "writer",  # If no tool call needed, go to writer
            },
        )

        # Edge from action back to agent (removed summarization)
        graph.add_edge("action", "agent")

        # Edge from writer to END
        graph.add_edge("writer", END)

        # Finally, we compile the graph
        print("Compiling graph...")
        compiled_graph = graph.compile()
        print("Graph compiled successfully.")
        return compiled_graph

    def generate_response(self, user_input: str) -> str:
        """Processes user input, runs it through the LangGraph engine, and returns the final response.

        Manages chat history between invocations.

        Args:
            user_input: The user's prompt.

        Returns:
            The agent's final response string.
        """
        start_time = time.time()
        logger.info(f"Processing user input: {user_input[:100]}...")
        
        # Convert user input to HumanMessage
        human_message = HumanMessage(content=user_input)

        # Append user message to internal chat history for context in next turn
        # Note: LangGraph state `messages` handles history *within* a single run.
        # We need self.chat_history to maintain context *between* runs.
        self.chat_history.append(human_message)

        # Prepare the initial state for this invocation using the history
        # Initialize the tool call counter for this turn
        initial_state = {"messages": self.chat_history, "tool_calls_this_turn": 0}
        logger.info(f"Invoking graph with {len(self.chat_history)} messages in history")

        try:
            # Invoke the graph
            # The `+` operator in AgentState definition handles message accumulation
            graph_start = time.time()
            final_state = self.graph.invoke(initial_state)
            graph_time = time.time() - graph_start
            logger.info(f"Graph execution completed in {graph_time:.2f} seconds")

            # Extract the final AI response from the graph state
            # The last message should be the AI's response after potentially multiple tool calls/responses
            final_ai_message = final_state["messages"][-1]
            
            if isinstance(final_ai_message, AIMessage):
                final_response = final_ai_message.content
                
                # Validation: Check if response seems complete
                if final_response and len(final_response) > 10:
                    total_time = time.time() - start_time
                    logger.info(f"Generated response: {len(final_response)} characters in {total_time:.2f} seconds")
                else:
                    logger.warning("Generated response seems incomplete or empty")
            else:
                error_msg = "Error: Expected AIMessage at the end of processing."
                logger.error(error_msg)
                final_response = error_msg

            # Append the final AI response to the history for the next turn
            self.chat_history.append(final_ai_message)

            # Limit history size to prevent context overflow
            if len(self.chat_history) > Config.MAX_CHAT_HISTORY:
                logger.info("Trimming chat history to maintain context size")
                self.chat_history = self.chat_history[-Config.MAX_CHAT_HISTORY:]

            logger.info("Response generation completed successfully")
            return final_response
            
        except Exception as e:
            error_msg = f"Error during graph execution: {str(e)}"
            logger.error(error_msg)
            return error_msg


agent = QwenAgent(
    tool_llm_provider=Config.TOOL_PROVIDER,
    tool_llm_model=Config.TOOL_MODEL,
    writer_llm_provider=Config.WRITER_PROVIDER,
    writer_llm_model=Config.WRITER_MODEL,
)

# Example loop for interactive chat
while True:
    try:
        prompt = input("Enter prompt (or 'quit' to exit): ")
        if prompt.lower() in ['quit', 'exit', 'q']:
            logger.info("User requested exit")
            break
        
        # Input validation
        if not prompt.strip():
            print("Please enter a non-empty prompt.")
            continue
            
        if len(prompt) > 10000:
            print("Prompt too long. Please limit to 10,000 characters.")
            continue
            
        logger.info("Processing user request")
        response = agent.generate_response(prompt)
        print(f"\nAgent Response:\n{response}\n")
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        print("\nExiting...")
        break
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        print(f"An error occurred: {e}")
        print("Continuing...")
# agent.generate_response(prompt) # Remove single execution line
