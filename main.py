import pymupdf4llm
import os
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
    """Recursively lists files and directories, showing the structure.
    If directory is unspecified, lists the current directory."""
    print(f"Helper: Recursively listing directory structure: {directory}")
    output_lines = []

    try:
        abs_directory = os.path.abspath(directory)
        if not os.path.isdir(abs_directory):
            return f"Error: Directory '{directory}' does not exist or is not accessible."

        # Get the starting path's base name for the root display
        start_dir_name = os.path.basename(abs_directory) or '.'
        output_lines.append(f"[{start_dir_name}/]")
        start_level = abs_directory.count(os.path.sep)

        for root, dirs, files in os.walk(abs_directory, topdown=True):
            # Filter directories to avoid traversing into them
            dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', '.vscode', '.venv')]
            # Filter files (optional)
            # files = [f for f in files if f not in ('.gitignore')]

            level = os.path.abspath(root).count(os.path.sep) - start_level
            indent = "    " * (level + 1) # Indent subitems further

            # List directories
            for d in sorted(dirs):
                output_lines.append(f"{indent}[{d}/]")

            # List files
            for f in sorted(files):
                output_lines.append(f"{indent}{f}")

        # Add a newline at the start for better formatting in context
        return "\n" + "\n".join(output_lines)

    except Exception as e:
        # Print the exception for debugging
        print(f"Error during os.walk in {directory}: {str(e)}") 
        return f"Error listing directory {directory}: {str(e)}"

# QwenAgent class
class QwenAgent:
    # Define the state for the graph
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    def __init__(self):
        self.chat_history = [] 
        # Define tools list - only file_read now
        _tools_list = [file_read]
        # Create tool map - only file_read
        self.tool_map = {tool.name: tool for tool in _tools_list}

        # Set up Langchain model
        self.llm = ChatOllama(model="qwen2.5:7b", temperature=0.2)

        # Bind the remaining tool (file_read) to the LLM
        self.llm_with_tools = self.llm.bind_tools(_tools_list)
        
        # Create summarization prompt and chain
        self.summarize_prompt = ChatPromptTemplate.from_template(
            "Summarize the following text concisely to maintain context in an ongoing conversation:\n\n{text}"
        )
        self.summarize_chain = self.summarize_prompt | self.llm
        
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
            message_content = f"Context: Current directory listing is - {response}"
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
        response = self.llm_with_tools.invoke(messages)
        print(f"Model response: {response}")
        # We return a list, because this will get added to the existing list
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

    # Define the function to summarize tool results if needed
    def _summarize_tool_results(self, state):
        """Checks the last ToolMessage(s) and summarizes content if it exceeds the threshold."""
        print("--- Summarizing Tool Results (If Needed) ---")
        messages = state["messages"]
        last_message = messages[-1]
        updated_messages = []
        
        # Check if the last message is a ToolMessage (or potentially multiple if _call_tool returns a list)
        # For simplicity, let's focus on summarizing the *last* message if it's a tool result.
        # A more robust approach might handle multiple tool messages from a single _call_tool step.
        
        # Let's iterate backwards to find the most recent tool messages to summarize
        messages_to_keep = []
        summarized_something = False
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                if len(msg.content) > SUMMARIZATION_THRESHOLD:
                    print(f"Summarizing ToolMessage (ID: {msg.tool_call_id}, Length: {len(msg.content)} chars)")
                    try:
                        summary_response = self.summarize_chain.invoke({"text": msg.content})
                        summary_content = summary_response.content
                        print(f"Summary: {summary_content}")
                        # Create a new ToolMessage with summarized content, keeping original ID
                        summarized_msg = ToolMessage(
                            content=f"Summary of previous tool call: {summary_content}",
                            tool_call_id=msg.tool_call_id,
                            name=msg.name # Keep original name if present
                        )
                        messages_to_keep.append(summarized_msg)
                        summarized_something = True
                    except Exception as e:
                        print(f"Error summarizing ToolMessage ID {msg.tool_call_id}: {e}")
                        # Keep original message if summarization fails
                        messages_to_keep.append(msg) 
                else:
                    # Keep ToolMessage as is (below threshold)
                    messages_to_keep.append(msg)
            else:
                # Stop when we hit a non-ToolMessage, keep it and everything before it
                messages_to_keep.append(msg)
                break
                
        if summarized_something:
            # Reconstruct the messages list in the original order
            final_messages = list(reversed(messages_to_keep)) + messages[:-len(messages_to_keep)]
            # Return the updated state dictionary
            return {"messages": final_messages}
        else:
            # No summarization occurred, return state unchanged (or signal no change)
            print("No summarization needed.")
            # Returning the original state dictionary to avoid modifying state unnecessarily
            return state 

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
        graph.add_node("summarize_results", self._summarize_tool_results) # New summarization node

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

        # Edge from action to summarization
        graph.add_edge("action", "summarize_results")

        # Edge from summarization back to agent
        graph.add_edge("summarize_results", "agent")

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

agent = QwenAgent()
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