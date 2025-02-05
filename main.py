import ollama
import re
import pymupdf4llm
import os
system_prompt = """[SYSTEM] You are an AI assistant capable of using various tools which you can use either on the users request or when you think they are needed.
        When calling these tools please be extra careful to use the precise syntax and format as shown below.
        You have access to these tools:
            - [FILE_READ]: Fetch content from any file. Usage: [FILE_READ]path/to/file When the user does not provide a file path, assume it is in the same directory as the script."""
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
                        search_result += f"\nFile Content ({file_path}):\n{load_file(file_path)}"
        print(search_result)
        return search_result

    def generate_response(self, prompt: str):
        """Recursive function that will generate a response based on the prompt given to it. 
        If the model requests a function, it will execute it and regenerate the response.

        Args:
            prompt (str): Prompt to generate a response for.
        """
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
            self.generate_response(new_prompt)
agent = QwenAgent()
agent.generate_response("Please read my CV stored in kraemer_cv.pdf and give me suggestions on how to improve it.")