

# tools/test_script_tool.py

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from tools.registry import register_tool
from typing import Dict, Any
import re
from langchain_community.llms.ollama import Ollama
from langchain.callbacks.base import BaseCallbackHandler

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.partial_output = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.partial_output += token


@tool
def generate_test_script(text_input: str,prompts: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Generates a basic test script based on a single, specific test case description.
    The text_input should be the test case selected by a human for script generation.
    """
    prompt_text = (prompts or {}).get("custom_prompt") or (prompts or {}).get("Default")
   
    
    if not prompt_text:
        return {
            "success": False,
            "message": "Tool Error: generate_test_script was called without a valid prompt.",
            "results": [],
            "response": "Could not generate a response because no prompt was provided to the generate_test_script tool.",
            "used_prompt": "None"
        }
    
    if not text_input or not isinstance(text_input, str):
        return {
            "success": False,
            "message": "Tool Error: Invalid or empty text_input provided.",
            "test_script": ""
        }

    try:
        # This prompt is highly specific to generating a script from a single test case.

        prompt = prompt_text.format(context=text_input)
        print(">>>>>>>>",prompt)
        streaming_handler = StreamingCallbackHandler()
        # Initialize the LLM as requested
        llm = Ollama(model="gemma3:12b", callbacks=[streaming_handler])
        
        # Invoke the LLM to get the response
        response = llm.invoke(prompt)
        print("RESPONSE",response)
        
        script_content = response.content if hasattr(response, 'content') else str(response)

        # Clean up the response to ensure it's just the script
        # Often, models wrap code in ```python ... ```
        # if "```" in script_content:
        #     # Find the first code block and extract its content
        #     match = re.search(r'```(?:python\n)?(.*?)```', script_content, re.DOTALL)
        #     if match:
        #         script_content = match.group(1).strip()
        
       

        return {
            "success": True,
            "message": "Test script generated successfully from the selected test case.",
            "test_script":script_content
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"An error occurred during test script generation: {str(e)}",
            "test_script": ""
        }

# Register the tool so the system can discover and use it
register_tool("generate_test_script", generate_test_script)