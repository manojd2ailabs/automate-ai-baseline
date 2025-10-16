
"""
Description:
Tool for AgenticAI workflows that generates structured test scripts from user-selected test cases. 
It formats retrieved agent memories into a contextual prompt and uses an LLM to produce Python-based test scripts.

Key Features:
- Accepts a test case input and optional retrieved memories to provide context-aware script generation.
- Integrates past agent findings to enrich the prompt for the LLM.
- Returns structured output including success status, message, and generated test script.

Created By: Pediredla Sai Ram, Parise Hari Sai
Date: 
Modified By:
Reason:
Example:
Used by the test_script_agent in agent_executor workflows to automatically generate test scripts from selected test cases, leveraging context from past memories and RAG-enhanced insights.
"""


from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from tools.registry import register_tool
from typing import Dict, Any, Optional
import re
from langchain_community.llms.ollama import Ollama
from langchain.callbacks.base import BaseCallbackHandler

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.partial_output = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.partial_output += token

@tool
def generate_test_script(text_input: str, prompts: Dict[str, str] = None, retrieved_memories: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generates a test script based on a test case, using a structured context from retrieved memories.
    - text_input: The test case selected by the user.
    - retrieved_memories: A dictionary containing memories loaded for this agent.
    """
    prompt_text = (prompts or {}).get("custom_prompt") or (prompts or {}).get("Default")
    print(text_input,"PROMPT TEXT")

    if not prompt_text:
        return {
            "success": False, "message": "Tool Error: generate_test_script was called without a valid prompt.", "test_script": ""
        }
    
    if not text_input or not isinstance(text_input, str):
        return {
            "success": False, "message": "Tool Error: Invalid or empty text_input provided.", "test_script": ""
        }

    try:
        # --- START: Structured Context Building (as per your example) ---
        
        # 1. Format the retrieved memories by iterating through all agents found.
        memory_context = "No relevant memories found." # Default value
        if retrieved_memories and isinstance(retrieved_memories, dict):
            formatted_memories = []
            for agent_id, findings in retrieved_memories.items():
                if findings and findings.get('memories'):
                    mem_str = "\n".join(f"- {mem}" for mem in findings['memories'])
                    formatted_memories.append(f"Memories from {agent_id}:\n{mem_str}")
            
            if formatted_memories:
                memory_context = "### Relevant Past Memories\n" + "\n\n".join(formatted_memories)

        # 2. Format the final prompt using TWO separate placeholders: {context} and {test_case}.
        prompt = prompt_text.format( testscript=memory_context,context=text_input)
        print("<<<<<<<<<<<<<,,")
        print("Final Prompt for Test Script Generation:\n", prompt)
        # print(text_input,"<<<<<<<<<<<<<<<<<<<<<<<<<<")
        
        # --- END: Structured Context Building ---
        
        streaming_handler = StreamingCallbackHandler()
        llm = Ollama(model="gemma3:12b", callbacks=[streaming_handler])
        
        response = llm.invoke(prompt)
        
        script_content = response.content if hasattr(response, 'content') else str(response)

        # Clean up potential markdown code fences from the LLM response.
        if "```" in script_content:
            match = re.search(r'```(?:python\n)?(.*?)```', script_content, re.DOTALL)
            if match:
                script_content = match.group(1).strip()
        
        return {
            "success": True,
            "message": "Test script generated successfully from the selected test case.",
            "test_script": response
        }
    except KeyError as e:
        return {
            "success": False,
            "message": f"Tool Error: Your prompt is missing a required placeholder. Please ensure the prompt for 'test_script_agent' contains both '{{context}}' and '{{test_case}}'. Error: Missing key {e}",
            "test_script": ""
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"An error occurred during test script generation: {str(e)}",
            "test_script": ""
        }

# Register the tool
register_tool("generate_test_script", generate_test_script)