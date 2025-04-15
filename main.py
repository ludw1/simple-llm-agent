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

# Remove unused system_prompt
# system_prompt = """[SYSTEM] You are an AI assistant capable of using various tools...
#            """

# Define summarization threshold (e.g., characters)
SUMMARIZATION_THRESHOLD = 2000

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
                if "thesis" in filename:
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
    # Define the state for the graph
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    def __init__(self, llm_provider: str = "ollama", model_name: str = "qwen2.5:7b"):
        self.chat_history = [] 
        # Define tools list - only file_read now
        _tools_list = [file_read]
        # Create tool map - only file_read
        self.tool_map = {tool.name: tool for tool in _tools_list}

        # Set up Langchain model based on provider
        if llm_provider == "ollama":
            self.llm = ChatOllama(model=model_name, temperature=0.2)
            print(f"Using Ollama model: {model_name}")
        elif llm_provider == "openrouter":
            # Ensure OPENROUTER_API_KEY is set in environment (loaded from .env)
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment or .env file.")
            # Use ChatOpenAI configured for OpenRouter
            self.llm = ChatOpenAI(
                model=model_name, 
                temperature=0.2, 
                openai_api_key=api_key, 
                openai_api_base="https://openrouter.ai/api/v1"
            )
            print(f"Using OpenRouter via ChatOpenAI: {model_name}")
        elif llm_provider == "openai":
            # Ensure OPENAI_API_KEY is set in environment (loaded from .env)
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment or .env file.")
            self.llm = ChatOpenAI(model=model_name, temperature=0.2, openai_api_key=api_key)
            print(f"Using OpenAI model: {model_name}")
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        # Bind the remaining tool (file_read) to the LLM
        self.llm_with_tools = self.llm.bind_tools(_tools_list)
        
        # Create summarization prompt and chain
        self.summarize_prompt = ChatPromptTemplate.from_template(
            "Summarize the following text concisely to maintain context in an ongoing conversation. Make sure to also include specific style points with examples from this text in your summary:\n\n{text}"
        )
        self.summarize_chain = self.summarize_prompt | self.llm
        
        # Define the persistent system prompt message content
        self.system_prompt_content = (
            "You are a helpful AI assistant. You have access to a tool for reading files (`file_read`). "
            "When asked to analyze information that might be contained in files listed in the context, "
            "you MUST use the `file_read` tool to read EACH relevant file before providing your final answer. "
            "Do not rely solely on filenames; fetch the content. Use the tool sequentially for multiple files if necessary."
        )
        
        # --- LangGraph setup --- 
        self.graph = self._build_graph()

    # Define the function that lists the directory content first
    def _list_directory(self, state):
        """Node that calls the request_files helper to get directory content."""
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
    def _call_model(self, state):
        print("--- Calling Model ---")
        messages = state["messages"]
        # Create the system message
        system_message = SystemMessage(content=self.system_prompt_content)
        # Prepend the system message to the current messages
        messages_with_system = [system_message] + messages
        print(f"Messages sent to LLM: {messages_with_system}")
        response = self.llm_with_tools.invoke(messages_with_system)
        print(f"Model response: {response}")
        # We return a list, because this will get added to the existing list via add_messages
        # Note: We only return the LLM's response, not the system prompt we added
        return {"messages": [response]}

    # Define the function to execute tools
    def _call_tool(self, state):
        print("--- Calling Tool ---")
        last_message = state["messages"][-1]
        
        # Ensure last_message is an AIMessage with tool_calls
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
             print("No valid tool calls found in the last AI message.")
             return {"messages": []} 

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
        # We return a list, because this will get added to the existing list (via add_messages)
        return {"messages": tool_messages}

    # Define the function that determines whether to continue or not
    def _should_continue(self, state):
        print("--- Checking if should continue ---")
        last_message = state["messages"][-1]
        # If there are no tool calls, then we finish
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            print("Decision: End")
            return "end"
        # Otherwise if there are tool calls, we continue
        else:
            print("Decision: Continue")
            return "continue"

    def _build_graph(self):
        # Graph building logic
        graph = StateGraph(self.AgentState)

        # Define the nodes
        graph.add_node("list_directory", self._list_directory)
        graph.add_node("agent", self._call_model)
        graph.add_node("action", self._call_tool)

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
                "end": END,
            },
        )

        # Edge from action back to agent (removed summarization)
        graph.add_edge("action", "agent")

        # Finally, we compile the graph
        print("Compiling graph...")
        compiled_graph = graph.compile()
        print("Graph compiled successfully.")
        return compiled_graph

    def generate_response(self, user_input: str):
        """Generate a response using the LangGraph engine."""
        # Convert user input to HumanMessage
        human_message = HumanMessage(content=user_input)
        
        # Append user message to internal chat history for context in next turn
        # Note: LangGraph state `messages` handles history *within* a single run.
        # We need self.chat_history to maintain context *between* runs.
        self.chat_history.append(human_message)

        # Prepare the initial state for this invocation using the history
        initial_state = {"messages": self.chat_history}
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
# Choose provider ('ollama', 'openrouter', or 'openai') and model name
# Ensure the corresponding API key (OPENROUTER_API_KEY or OPENAI_API_KEY) is set in your .env file
PROVIDER = "openrouter" # "ollama" or "openrouter" or "openai"
# Update model to Mistral Small via OpenRouter
MODEL = "mistralai/mistral-small-3.1-24b-instruct:free" # Adjust model name as needed

agent = QwenAgent(llm_provider=PROVIDER, model_name=MODEL)

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