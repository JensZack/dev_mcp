"""
Google genai client.
"""

import os
import subprocess
from typing import Callable
from pathlib import Path
from google import genai
from google.genai import types
from pylspclient.lsp_client import LspClient
from pylspclient.lsp_endpoint import LspEndpoint
from pylspclient.json_rpc_endpoint import JsonRpcEndpoint
from pylspclient.lsp_pydantic_strcuts import TextDocumentIdentifier, Position


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def another_function(a: float, b: int):
    return (a ** 2) + (a * b)


def my_function(a: int, b: float):
    if a == 0:
        x = b ** 2
        y = 1
    else:
        x = b / a
        y = -1

    z = another_function(x, y)  # 32:9
    return z
    

tools: dict[str, Callable] = {}
def register_tool(func):
    tools[func.__name__] = func
    return func

@register_tool
def get_function_definition(function_name: str):
    """
    given the name of a function, returns the function definition
    """
    # functions = {"another_function": another_function_text}
    functions = {}
    if function_name not in functions:
        raise ValueError(f"{function_name} not available")
    return functions[function_name]

function = types.FunctionDeclaration(
    name='get_function_definition',
    description='Get the definition of a function by name',
    parameters=types.Schema(
        type='OBJECT',
        properties={
            'function_name': types.Schema(
                type='STRING',
                description='the name of the function',
            ),
        },
        required=['function_name'],
    ),
)

tool = types.Tool(function_declarations=[function])
    


class GenAIClient:
    """
    A simple client for interacting with the Google Generative AI API.
    """

    def __init__(self):
        """
        Initializes the GenAIClient with a specified model.

        Args:
            model_name (str): The name of the generative model to use.
                               Defaults to "gemini-1.0-pro".
        """
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = "gemini-2.0-flash"
        # self.model_name = "gemini-2.5-pro-preview-05-06"


    def generate_text(self, prompt):
        """
        Generates text based on the given prompt.

        Args:
            prompt (str): The input prompt for text generation.

        Returns:
            str: The generated text, or None if an error occurred.
        """
        config=types.GenerateContentConfig(
            tools=[tool],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=2,
            ),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode='ANY', allowed_function_names=["get_function_definition"]),
            ),
        )
        max_iterations = 2
        content = [prompt]
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                # system_instruction=["you are an expert coding assistant that thinks deeply about fixing and writing code"],
                contents=content,
                config=config,
            )

            tool_results = []
            for _ in range(max_iterations):
                if response.function_calls is None:
                    break

                for function_call in response.function_calls:
                    if function_call.name not in tools:
                        raise KeyError(f"{function_call.name} not registered as a tool")
                    
                    tool_result = tools[function_call.name](**function_call.args)
                    tool_results.append(tool_result)
            
            return response.text
        except Exception as e:
            print(f"Error generating text: {e}")
            return None

def test_mcp():
    # Example usage:
    client = GenAIClient()
    prompt = f"""
    If I pass a = 1 and b = 10 to:
    my_function()
    what is the result?
    """
    generated_text = client.generate_text(prompt)

    if generated_text:
        print("Generated Text:")
        print(generated_text)

class Pylsp:
    def __enter__(self):
        self.proc = subprocess.Popen("uv run pylsp -vvv --log-file test_pylsp.log".split(" "), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return self.proc

    def __exit__(self, exc_type, exc_value, traceback):
        self.proc.kill()
        self.proc.communicate()


def test_lsp():
    with Pylsp() as pylsp:
        assert pylsp.stdin is not None and pylsp.stdout is not None
        endpoint = JsonRpcEndpoint(pylsp.stdin, pylsp.stdout)
        root_uri = Path(__file__).parents[2].as_uri()
        lsp_endpoint = LspEndpoint(endpoint)
        lsp_client = LspClient(lsp_endpoint=lsp_endpoint)
        lsp_client.initialize(rootUri=root_uri, capabilities={})
        uri = Path(__file__).as_uri().__str__()
        document = TextDocumentIdentifier(uri=uri)

        # lsp counts lines/characters from 0, vim counts from 1
        position = Position(line=31, character=9)
        res = lsp_client.definition(document, position)

    return res


if __name__ == "__main__":
    test_lsp()
    
