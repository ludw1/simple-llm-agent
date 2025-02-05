import ollama
import re

system_prompt = """[SYSTEM] You are an AI assistant capable of using various tools which you can use either on the users request or when you think they are needed.
        When calling these tools please be extra careful to use the precise syntax and format as shown below.
        You have access to these tools:
            - [FILE_READ]: Fetch content from any file. Usage: [FILE_READ]path/to/file When the user does not provide a file path, assume it is in the same directory as the script."""
def load_file(file_path: str) -> str:
    """Load a file and return its content."""
    with open(file_path, "r") as file:
        return file.read()
class QwenAgent:
    def __init__(self):
        self.function_pattern = re.compile(r"\[.*?\]")  # Regex to detect function requests
        self.file_pattern = re.compile(r"(?i)\[FILE_READ\](.*?\.\w+)")  # Regex to detect file requests
        self.chat_history = []

    def _handle_function_request(self, response: str) -> str:
        """Check if the model requested a function and execute it.
        Args:
            response (str): The model response.
        Returns:
            str: The content of the file if a file read function was requested.
        """
        search_result = ""
        match = self.function_pattern.findall(response)
        for func_call in set(match):
            if "[FILE_READ]" in func_call.upper():
                match = self.file_pattern.findall(response)
                if match:
                    for file_path in match:
                        print(f"Reading file: {file_path}")
                        try:
                            search_result += f"\nFile Content ({file_path}):\n{load_file(file_path)}"
                        except Exception as e:
                            search_result += f"Error reading {file_path}: {str(e)}"
        return search_result

    def generate_response(self, prompt: str) -> str:
        # Initial model response
        response = ollama.generate(
            model="qwen2.5:7b",
            prompt=prompt,
            system=system_prompt,
            options={"temperature": 0.2}  # Lower temp for precise tool use
        )['response']
        print("Model reponse:", response)
        # Check if the model wants to execute a function
        self.chat_history.append({"role": "assistant", "content": response})
        function_calls = self._handle_function_request(response)
        if function_calls:
            self.chat_history.append({"role": "function", "content": function_calls})
            # Regenerate the answer with function content
            chat_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.chat_history])
            new_prompt = f"{chat_context}\n\nUser: {prompt}\nAssistant:"
            print(new_prompt)
            self.generate_response(new_prompt)
agent = QwenAgent()
agent.generate_response("Please read and summarize the file dummy.txt as well as the file dummy2.txt. ")