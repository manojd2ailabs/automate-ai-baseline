
from typing import Optional, Dict, Any, List
import json
import re
import traceback
from tools.registry import discover_and_register_tools, get_all_registered_tools
from agents.a2a_factory import A2AAgentFactory
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from agents.a2a_system import _global_registry
from graph.research_state import ResearchState
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import inspect
from tools.tool_executor import ToolExecutor

def get_agent_tools(all_tools: Dict[str, Any], available_tools: Optional[List[Any]]) -> List[Any]:
    """
    Convert agent card's available_tools into tool instances.
    """
    if not available_tools:
        print("⚠️ No available_tools specified for this agent; returning empty list.")
        return []

    agent_tools = []
    for tool_spec in available_tools:
        tool_name = tool_spec.get("name") if isinstance(tool_spec, dict) else tool_spec
        tool = all_tools.get(tool_name)
        if not tool:
            print(f"⚠️ Tool '{tool_name}' not found in registry")
            continue
        agent_tools.append(tool)
    return agent_tools


class BaseAgent:
    def __init__(self, card, prompts: Optional[Dict[str, Dict[str, str]]] = None):
        self.card = card
        self.agent_id = card.id
        self.system_prompt = card.system_prompt
        self.llm = ChatOllama(model="llama3.1:8b", temperature=0)
        # Register global tools once
        discover_and_register_tools()
        all_tools = get_all_registered_tools()
        self.tools = get_agent_tools(all_tools, card.available_tools or [])
        self.tool_names = [tool.name for tool in self.tools]
        self.prompts = prompts or {self.agent_id: {"system_prompt": self.system_prompt}}

        print(f"🧠 Agent {self.agent_id} initialized with tools: {self.tool_names}")

    def get_prompts(self) -> Dict[str, str]:
        """
        Return the prompt(s) for this agent.
        If no specific prompts for this agent are found, returns an empty dict.
        """
        return self.prompts.get(self.agent_id, {})

    def run(self, state: Optional[ResearchState] = None) -> Dict[str, Any]:
        """
        Run the agent's iterative multi-tool LLM workflow.
        Collects all tool outputs in tool_results and returns a full findings dict.
        """
        try:
            if state is not None:
                self.shared_state = dict(state)  # make shallow copy of state
            else:
                self.shared_state = {}

            tool_executor = ToolExecutor(self.tools, self.shared_state)

            # Descriptions for all tools provided to LLM
            tool_descriptions = self._build_tool_descriptions()

            max_iterations = 8
            iteration = 0
            tools_called = []
            tool_results = {}

            # Initialize LLM conversation with system prompt and instructions
            messages = [
                HumanMessage(content=f"""
{self.system_prompt}

AVAILABLE TOOLS:
{tool_descriptions}

CURRENT CONTEXT:
{self._build_context()}

INSTRUCTIONS:
- Use the available tools to complete your analysis step by step
- Call tools by responding with: CALL_TOOL: tool_name [with parameters]
- After completing your tool usage, provide FINAL_ANSWER: with your conclusion
- Be concise and focused on the task

What tool should you use first?
""")
            ]

            while iteration < max_iterations:
                iteration += 1
                print(f"🔄 Iteration {iteration}")

                response = self.llm.invoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                messages.append(AIMessage(content=content))
                print(f"🤖 LLM Response: {content[:200]}...")

                tool_call = self._extract_tool_call(content)

                if tool_call:
                    tool_name = tool_call['name']
                    params = tool_call.get('params', {})

                    try:
                        result = tool_executor.execute_tool(tool_name, **params)
                        tools_called.append(tool_name)
                        tool_results[tool_name] = result

                        result_msg = f"TOOL_RESULT: {tool_name} completed successfully."
                        if isinstance(result, str) and len(result) < 200:
                            result_msg += f" Summary: {result}"
                        messages.append(HumanMessage(content=result_msg + "\n\nWhat's next, or provide FINAL_ANSWER if complete?"))
                        print(f"📤 Tool '{tool_name}' result sent to conversation")

                    except Exception as e:
                        error_msg = f"TOOL_ERROR: {tool_name} failed: {str(e)}\nProvide FINAL_ANSWER if ready."
                        messages.append(HumanMessage(content=error_msg))
                        print(f"❌ Tool error added to conversation")
                elif "FINAL_ANSWER:" in content.upper():
                    print(f"🏁 Final answer provided at iteration {iteration}")
                    break
                elif iteration >= max_iterations or len(tools_called) >= len(self.tools):
                    messages.append(HumanMessage(content="Please provide your FINAL_ANSWER based on available information."))
                    response = self.llm.invoke(messages)
                    final_content = response.content if hasattr(response, 'content') else str(response)
                    messages.append(AIMessage(content=final_content))
                    print(f"🏁 Forced completion at iteration {iteration}")
                    break
                else:
                    messages.append(HumanMessage(content="Continue with your next tool call or provide FINAL_ANSWER if ready."))

            # Merge updated shared_state back into input state if provided
            if state is not None:
                for k, v in self.shared_state.items():
                    state[k] = v
                print(f"📤 Merged state keys: {list(state.keys())}")

            final_response = self._extract_final_answer(messages)
            print(f"📋 Final response: {final_response[:200]}...")

            return {
                "summary": final_response,
                "tools": list(set(tools_called)),
                "tool_results": tool_results
            }

        except Exception as e:
            traceback.print_exc()
            return {
                "summary": f"[ERROR] Agent '{self.agent_id}' run failed: {str(e)}",
                "tools": [],
                "tool_results": {}
            }

    def _extract_final_answer(self, messages: List[HumanMessage or AIMessage]) -> str:
        """
        Extract the FINAL_ANSWER from the last several messages.
        """
        for message in reversed(messages[-3:]):
            if hasattr(message, 'content') and "FINAL_ANSWER:" in message.content.upper():
                content = message.content
                idx = content.upper().find("FINAL_ANSWER:")
                if idx >= 0:
                    return content[idx + len("FINAL_ANSWER:"):].strip()

        # fallback: return last AI message content
        for message in reversed(messages):
            if isinstance(message, AIMessage) and hasattr(message, 'content'):
                return message.content

        return "No response generated"

    def _build_tool_descriptions(self) -> str:
        return "\n".join(f"- {tool.name}: {getattr(tool, 'description', 'No description')}" for tool in self.tools)

    def _extract_tool_call(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Detect CALL_TOOL pattern and extract tool name and parameters.
        """
        pattern = r'CALL_TOOL:\s*(\w+)(?:\s+(.+))?'
        match = re.search(pattern, content, re.IGNORECASE)
        if not match:
            return None
        tool_name = match.group(1)
        params_text = match.group(2) or ""
        params = {'input': params_text.strip()} if params_text.strip() else {}
        return {'name': tool_name, 'params': params}

    def _build_context(self) -> str:
        """
        Convert shared state into a string summary for LLM input context.
        Filter out empty values and error entries.
        """
        if not self.shared_state:
            return "No context available"

        parts = []
        for key, value in self.shared_state.items():
            if value and key != "messages" and not key.endswith("_error"):
                parts.append(f"{key}: {str(value)[:100]}")
        return "\n".join(parts) if parts else "No relevant context"






class ResearchSupervisorAgent:
    def __init__(self, card, prompts: Optional[Dict[str, Dict[str, str]]] = None):
        self.card = card
        self.agent_id = card.id
        self.system_prompt = card.system_prompt
        self.prompts = prompts or {self.agent_id: {"system_prompt": self.system_prompt}}
        self.llm = ChatOllama(model="llama3.1:8b", temperature=0)

        # Register tools once globally
        discover_and_register_tools()
        all_tools = get_all_registered_tools()

        # Load and create sub-agents for coordinates_agents
        self.agents = []
        all_agent_ids = set(A2AAgentFactory.get_all_agent_ids())
        sub_agent_ids = card.coordinates_agents or []

        for agent_id in sub_agent_ids:
            if agent_id == self.agent_id:
                continue  # skip self
            if agent_id not in all_agent_ids:
                print(f"⚠️ Agent '{agent_id}' in coordinates_agents but not registered, skipping.")
                continue

            agent_card = _global_registry.get_agent_card(agent_id)
            if not agent_card:
                print(f"⚠️ Agent card for '{agent_id}' not found, skipping.")
                continue

            if not agent_card.system_prompt:
                print(f"⚠️ Skipping {agent_id}: missing system prompt.")
                continue

            # Get tools for sub-agent
            agent_tools = get_agent_tools(all_tools, agent_card.available_tools)

            react_agent = create_react_agent(
                model=self.llm,
                tools=agent_tools,
                name=agent_id,
            )
            self.agents.append(react_agent)

        # Build supervisor prompt with agent list injected
        raw_template = self.prompts.get("research_supervisor", {}).get("system_prompt_template", "")
        if not raw_template:
            raise ValueError("Missing 'system_prompt_template' for supervisor.")

        agent_list_text = "\n".join(f"- {agent.name}" for agent in self.agents)
        supervisor_prompt = raw_template.format(available_agents_list=agent_list_text)

        print(f"[INFO]: supervisor prompt is {supervisor_prompt}")
        self.supervisor = create_supervisor(
            agents=self.agents,
            model=self.llm,
            prompt=supervisor_prompt,
            output_mode="last_message"
        ).compile()

    def run(self, state: ResearchState) -> str:
        """Run supervisor agent"""
        file_path = state.get("file_path", None)
        context_input = f"Research State: {str(state)}\nFile Path: {file_path or 'No file provided.'}"
        
        # Clean state of any ellipsis objects
        clean_state = self._clean_state(dict(state))
        
        try:
            clean_state["messages"] = [{"role": "user", "content": context_input}]
            config = {"recursion_limit": 50}
            
            result = self.supervisor.invoke(clean_state,config=config)
            
            print("[INFO] context SUCCESS!")
            last_message = result["messages"][-1]
            content = getattr(last_message, 'content', str(last_message))
            return self._extract_json_from_response(content)
            
        except Exception as e:
            print(f"[ERROR] Supervisor failed with exception: {e}")
            import traceback
            traceback.print_exc()
            
            return json.dumps({
                "required_agents": [],
                "topics": {},
                "coordination_plan": f"Supervisor failed to analyze brief: {str(e)}"
            }, indent=2)

    def _clean_state(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Remove ellipsis and other non-serializable objects from state"""
        clean_dict = {}
        for key, value in state_dict.items():
            if value is ... or value is Ellipsis:
                continue  # Skip ellipsis
            elif isinstance(value, dict):
                clean_dict[key] = self._clean_state(value)
            elif isinstance(value, list):
                clean_dict[key] = [self._clean_value(item) for item in value if item is not ... and item is not Ellipsis]
            else:
                clean_dict[key] = self._clean_value(value)
        return clean_dict

    def _clean_value(self, value):
        """Clean individual values"""
        if value is ... or value is Ellipsis:
            return None
        return value

    def synthesize_findings(self, all_findings: Dict[str, str]) -> str:
        synthesis_prompt = self.prompts.get("coordination", {}).get("findings_synthesis", "")
        if not synthesis_prompt:
            raise ValueError("Missing 'findings_synthesis' prompt in prompts.yml")

        findings_text = "\n".join([f"- {agent}: {text}" for agent, text in all_findings.items()])
        response = self.llm.invoke(synthesis_prompt.format(findings=findings_text))
        return response.content if hasattr(response, "content") else str(response)
    
    def _extract_json_from_response(self, text: str) -> Dict[str, Any]:
        print(f"[DEBUG] Extracting JSON from text: {text[:200]}...")  # Show first 200 chars
        
        try:
            # Try to find a JSON block - improved regex
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',  # JSON code block
                r'```\s*(\{.*?\})\s*```',      # Generic code block
                r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',  # Basic JSON object
                r'(\{.*\})',                    # Fallback - any braces content
            ]
            
            for i, pattern in enumerate(json_patterns):
                print(f"[DEBUG] Trying pattern {i+1}: {pattern}")
                json_match = re.search(pattern, text, re.DOTALL)
                
                if json_match:
                    raw_json = json_match.group(1)
                    print(f"[DEBUG] Found JSON with pattern {i+1}: {raw_json}")
                    
                    try:
                        parsed = json.loads(raw_json)
                        print(f"[DEBUG] Successfully parsed JSON: {parsed}")
                        
                        # Check for ellipsis placeholders
                        parsed = self._clean_state(parsed)
                        
                        # Validate required fields
                        if not isinstance(parsed.get('required_agents'), list):
                            print("[WARNING] required_agents is not a list, fixing...")
                            parsed['required_agents'] = []
                        
                        if not isinstance(parsed.get('topics'), dict):
                            print("[WARNING] topics is not a dict, fixing...")
                            parsed['topics'] = {}
                            
                        if 'coordination_plan' not in parsed:
                            print("[WARNING] coordination_plan missing, adding default...")
                            parsed['coordination_plan'] = "No coordination plan provided"
                        
                        return parsed
                        
                    except json.JSONDecodeError as json_err:
                        print(f"[DEBUG] JSON decode failed for pattern {i+1}: {json_err}")
                        continue

            # If no JSON found, try to extract key information manually
            print("[DEBUG] No valid JSON found, attempting manual extraction...")
            
            # Look for agent mentions
            agents_match = re.search(r'(?:agents?|required)[:\s]*\[([^\]]+)\]', text, re.IGNORECASE)
            agents = []
            if agents_match:
                agent_text = agents_match.group(1)
                agents = [agent.strip(' "\'') for agent in agent_text.split(',')]
                print(f"[DEBUG] Manually extracted agents: {agents}")

            return {
                "required_agents": agents,
                "topics": {},
                "coordination_plan": f"Manual extraction from: {text[:100]}..."
            }

        except Exception as e:
            print(f"[ERROR] JSON parsing error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "required_agents": [],
                "topics": {},
                "coordination_plan": f"JSON parsing failed: {e}"
            }
        
    # Add this method to your supervisor agent class

    def review_agent_work(self, review_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review completed agent work and decide on next steps

        Args:
            review_context: Contains completed_agent, agent_findings, remaining_agents, etc.

        Returns:
            Dict with continue flag, additional_agents list, and feedback
        """

        completed_agent = review_context.get("completed_agent", "")
        findings = review_context.get("agent_findings", {}).get(completed_agent, {})
        remaining_agents = review_context.get("remaining_agents", [])
        accumulated_context = review_context.get("accumulated_context", "")

        # Create review prompt for the supervisor
        review_prompt = f"""
        You are reviewing the work completed by agent '{completed_agent}'.

        AGENT FINDINGS:
        {findings}

        ACCUMULATED CONTEXT FROM ALL AGENTS:
        {accumulated_context}

        REMAINING AGENTS IN PLAN: {remaining_agents}

        ORIGINAL BRIEF: {review_context.get('original_brief', '')}

        Based on the agent's work, determine:
        1. Should we continue with the current plan?
        2. Do we need additional agents beyond what's planned?
        3. Any feedback or adjustments needed?

        Respond in JSON format:
        {{
            "continue": true/false,
            "additional_agents": ["agent1", "agent2"] or [],
            "feedback": "Your assessment and recommendations",
            "updated_plan": "Any updates to coordination plan if needed"
        }}
        """

        try:
            # Use your supervisor's LLM to analyze
            response = self.llm.invoke([{"role": "user", "content": review_prompt}])

            # Parse response
            import json
            result = json.loads(response.content)

            return result

        except Exception as e:
            # Fallback response
            return {
                "continue": True,
                "additional_agents": [],
                "feedback": f"Review completed for {completed_agent}. Continuing with plan.",
                "updated_plan": review_context.get("coordination_plan", "")
            }




class LogAnalysisAgent(BaseAgent):
    pass

class TestCaseAgent(BaseAgent):
    pass


class TestScriptAgent(BaseAgent):
    def run(self, state: Optional[ResearchState] = None) -> Dict[str, Any]:
        """
        Runs the test script generation. It specifically looks for 'selected_test_case'
        in the provided state.
        """
        try:
            if state is None:
                raise ValueError("TestScriptAgent requires a state object with 'selected_test_case'.")

            self.shared_state = dict(state)
            
            # --- IMPORTANT CHANGE ---
            # Get the input for the tool directly from the state.
            selected_test_case = self.shared_state.get("selected_test_case")
            if not selected_test_case:
                return {
                    "summary": "[ERROR] No test case was selected for script generation.",
                    "tools": [],
                    "tool_results": {}
                }

            # Prepare tool execution
            tool_executor = ToolExecutor(self.tools, self.shared_state)
            tool_name = "generate_test_script"
            
            # Execute the tool with the selected test case as input
            result = tool_executor.execute_tool(tool_name, text_input=selected_test_case)

            return {
                "summary": "Successfully generated the test script.",
                "tools": [tool_name],
                "tool_results": {tool_name: result}
            }

        except Exception as e:
            traceback.print_exc()
            return {
                "summary": f"[ERROR] Agent '{self.agent_id}' run failed: {str(e)}",
                "tools": [],
                "tool_results": {}
            }
