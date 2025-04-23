import pymupdf4llm
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate


# --- Configuration --- 
# Maximum number of tool call iterations per user turn
MAX_TOOL_CALLS_PER_TURN = 3 

# Load environment variables from .env file
load_dotenv()

def load_file(file_path: str) -> str:
    """Load a file and return its content. Depending on the file type, different methods can be used to read it.
    Args:
        file_path (str): Path to the file.
    Returns:
        str: Content of the file.
    """
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    
    if file_path.lower().endswith(".pdf"):
        try:
            pdf_text = pymupdf4llm.to_markdown(file_path)
        except Exception as e:
            pdf_text = f"Error reading {file_path}: {str(e)}"
        return pdf_text
    else:
        try:
            with open(file_path, "r") as file:
                return file.read()
        except Exception as e:
            return f"Error reading {file_path}: {str(e)}"

@tool
def file_read(file_path: str) -> str:
    """Fetch content from a specified file. Supports reading text files and extracting text from PDF files. Provide the full file path."""
    print(f"Tool: Reading file: {file_path}")
    return load_file(file_path)

# Make request_files a regular helper function, no longer a tool
def request_files(directory: str = ".") -> str:
    """Recursively lists full paths of files within a directory."""
    print(f"Helper: Recursively listing full file paths in: {directory}")
    file_paths = []

    try:
        abs_directory = os.path.abspath(directory)
        if not os.path.isdir(abs_directory):
            return f"Error: Directory '{directory}' does not exist or is not accessible."

        for root, dirs, files in os.walk(abs_directory, topdown=True):
            # Filter directories to avoid traversing into them
            dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', '.vscode', '.venv')]
            
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
    def __init__(self, 
                 tool_llm_provider: str = "openrouter", 
                 tool_llm_model: str = "google/gemini-2.0-flash-exp:free",
                 writer_llm_provider: str = "openrouter",
                 writer_llm_model: str = "arliai/qwq-32b-arliai-rpr-v1:free"
                 ):
        """Initializes the QwenAgent with potentially separate tool and writer LLMs.

        Args:
            tool_llm_provider: The provider for the LLM handling tool calls.
            tool_llm_model: The model name for the tool LLM.
            writer_llm_provider: The provider for the LLM handling final writing/synthesis.
            writer_llm_model: The model name for the writer LLM.
        """
        self.chat_history = []
        _tools_list = [file_read]
        self.tool_map = {tool.name: tool for tool in _tools_list}

        # --- Initialize Tool LLM --- 
        print(f"Initializing Tool LLM: Provider={tool_llm_provider}, Model={tool_llm_model}")
        self.tool_llm = self._initialize_llm(tool_llm_provider, tool_llm_model)
        # Bind tools ONLY to the tool_llm
        self.tool_llm_with_tools = self.tool_llm.bind_tools(_tools_list)

        # --- Initialize Writer LLM --- 
        print(f"Initializing Writer LLM: Provider={writer_llm_provider}, Model={writer_llm_model}")
        # Initialize writer LLM WITHOUT binding tools
        self.writer_llm = self._initialize_llm(writer_llm_provider, writer_llm_model)

        # Define the persistent system prompt message content (primarily for tool LLM)
        self.system_prompt_content = (
            "You are an orchestrator assistant. Your primary goal is to use tools effectively to gather information needed to answer the user's query."
            "You have access to ONE tool: `file_read(file_path: str)` which allows you to read the content of a file specified by its full path."
            "\n**CRITICAL INSTRUCTION:** If the user asks a question that requires information potentially contained within files listed in the context, you MUST use the `file_read` tool to fetch the content of EACH relevant file."
            "Do NOT attempt to answer the question directly yourself, even if you think you know the answer or have the information from tool calls already."
            "Your job is to call the tool(s). Another assistant will synthesize the final answer."
            "Identify the full paths of the files you need to read from the context."
            "Then, invoke the `file_read` tool for each file path sequentially."
            "If no tool call is needed, or after you have made all necessary tool calls, stop and let the other assistant respond."
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
                raise ValueError("OPENROUTER_API_KEY not found in environment or .env file.")
            return ChatOpenAI(
                model=model_name,
                temperature=0.2,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1"
            )
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment or .env file.")
            return ChatOpenAI(model=model_name, temperature=0.2, openai_api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    # Define the function that lists the directory content first
    def _list_directory(self, state: AgentState) -> dict:
        """Graph Node: Calls the `request_files` helper and adds the result as a SystemMessage."""
        print("--- Listing Directory (Helper Call) ---")
        directory_to_list = "." # Always list the current directory
        
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
        
        writer_system_prompt = SystemMessage(content="You are a helpful AI assistant. Synthesize the information provided in the previous messages (including tool results) to answer the user's final query comprehensively. If asked to write in a specific style, analyze the provided text samples and mimic the style in your response.")
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
        print("--- Calling Tool ---")
        last_message = state["messages"][-1]
        
        # Ensure last_message is an AIMessage with tool_calls
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
             print("No valid tool calls found in the last AI message.")
             return {"messages": [], "tool_calls_this_turn": state.get('tool_calls_this_turn', 0)} 

        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            print(f"Executing tool: {tool_name} with args {tool_args} (ID: {tool_id})")
            
            # Look up the tool function in the map
            if tool_name not in self.tool_map:
                print(f"Error: Tool '{tool_name}' not found.")
                response_content = f"Error: Tool '{tool_name}' not found."
            else:
                tool_to_call = self.tool_map[tool_name]
                try:
                    # Invoke the tool using the retrieved function object
                    response = tool_to_call.invoke(tool_args)
                    response_content = str(response)
                except Exception as e:
                    print(f"Error executing tool {tool_name}: {e}")
                    response_content = f"Error executing tool {tool_name}: {e}"

            # Include the tool name in the ToolMessage for clarity
            tool_messages.append(ToolMessage(content=response_content, name=tool_name, tool_call_id=tool_id))

        print(f"Tool responses: {tool_messages}")
        # Increment the counter for this turn
        current_count = state.get('tool_calls_this_turn', 0)
        new_count = current_count + 1
        print(f"Tool call iteration {new_count}/{MAX_TOOL_CALLS_PER_TURN} for this turn.")
        # We return a list, because this will get added to the existing list (via add_messages)
        # Also return the updated counter
        return {"messages": tool_messages, "tool_calls_this_turn": new_count}

    # Define the function that determines whether to continue or not
    def _should_continue(self, state: AgentState) -> str:
        """Graph Node: Determines the next step based on the last message and tool call limit."""
        print("--- Checking if should continue ---")
        last_message = state["messages"][-1]
        tool_calls_made = state.get('tool_calls_this_turn', 0)
        
        # Check if the LLM requested tool calls
        has_tool_calls = hasattr(last_message, 'tool_calls') and last_message.tool_calls

        if has_tool_calls:
            # Check if we've exceeded the limit for this turn
            if tool_calls_made >= MAX_TOOL_CALLS_PER_TURN:
                print(f"Decision: Tool call limit ({MAX_TOOL_CALLS_PER_TURN}) reached. Forcing end of tool loop.")
                return "end"
            else:
                # Limit not reached, continue tool execution
                print(f"Decision: Continue tool execution (Iteration {tool_calls_made + 1})")
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
                "continue": "action", # If tool call needed, go to action
                "end": "writer",      # If no tool call needed, go to writer
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
        # Convert user input to HumanMessage
        human_message = HumanMessage(content=user_input)
        
        # Append user message to internal chat history for context in next turn
        # Note: LangGraph state `messages` handles history *within* a single run.
        # We need self.chat_history to maintain context *between* runs.
        self.chat_history.append(human_message)

        # Prepare the initial state for this invocation using the history
        # Initialize the tool call counter for this turn
        initial_state = {"messages": self.chat_history, "tool_calls_this_turn": 0}
        print(f"Invoking graph with state: {initial_state}")

        # Invoke the graph
        # The `+` operator in AgentState definition handles message accumulation
        final_state = self.graph.invoke(initial_state)
        print(f"Final graph state: {final_state}")

        # Extract the final AI response from the graph state
        # The last message should be the AI's response after potentially multiple tool calls/responses
        final_ai_message = final_state['messages'][-1]
        final_response = final_ai_message.content if isinstance(final_ai_message, AIMessage) else "Error: Expected AIMessage at the end."
        
        # Append the final AI response to the history for the next turn
        self.chat_history.append(final_ai_message)
        
        # Limit history size (optional, but good practice)
        # self.chat_history = self.chat_history[-10:] # Keep last 5 pairs

        print("Final Response:", final_response)
        # Return the response content string
        return final_response

# Example usage:
# Configure Tool LLM (e.g., Gemini Flash for tool calls)
TOOL_PROVIDER = "openrouter"
TOOL_MODEL = "google/gemini-2.0-flash-exp:free"

# Configure Writer LLM (e.g., Arliai Qwen for final response)
WRITER_PROVIDER = "openrouter"
WRITER_MODEL = "arliai/qwq-32b-arliai-rpr-v1:free"

# Ensure the corresponding API key(s) are set in your .env file

agent = QwenAgent(
    tool_llm_provider=TOOL_PROVIDER, 
    tool_llm_model=TOOL_MODEL,
    writer_llm_provider=WRITER_PROVIDER,
    writer_llm_model=WRITER_MODEL
)

# Example loop for interactive chat
while True:
    try:
        prompt = input("Enter prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        agent.generate_response(prompt)
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
# agent.generate_response(prompt) # Remove single execution line