# graph/research_graph.py
"""
Description:
This module implements a graph-based orchestration system for managing multi-agent 
research workflows based on LangGraph. It coordinates the execution of agents via the Agent Executor, 
handling dynamic routing based on agent outputs, user inputs, and supervisor-defined 
execution plans. The graph manages state across agents, findings, and metadata, 
supports memory integration (load/save), and synthesizes final reports from agent outputs. 
It also implements pausing points for user input, email approvals, and memory-saving decisions, 
ensuring smooth handoff and context propagation between agents.

Created By: Tanvi Dongaonkar , Pediredla Sai Ram, Parise Hari Sai, Jayant Sharma
Date: 
Modified By:
Reason:
Example:
This module orchestrates LLM-driven agents such as log analysis, test case generation, 
email handling, and test script creation. It invokes each agent through the Agent Executor, 
manages agent queues, propagates retrieved memories, and compiles a final report while 
maintaining persistent context across all steps.
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
import os
from tools.memory_tools import save_memories, load_memories
import json
import traceback
from graph.research_state import ResearchState
import logging
from tools.memory_tools import AGENT_DB_MAP
from agents.a2a_factory import A2AAgentFactory

logger = logging.getLogger(__name__)

class ResearchGraph:
    def __init__(self, prompts: Dict[str, Dict[str, str]], supervisor_agent, agent_factory: A2AAgentFactory):
        self.prompts = prompts
        self.supervisor = supervisor_agent
        self.agent_factory = agent_factory
        self.graph = self._build_graph()

    def _create_agent_node(self, agent_id: str):
        def agent_node(state: ResearchState) -> ResearchState:
            try:
                agent_prompts = self.prompts.get(agent_id, {}).copy()
                selection_info = (state.get("selected_prompts") or {}).get(agent_id)
                if selection_info:
                    prompt_text = selection_info.get("prompt_text")
                    agent_prompts["system_prompt"] = prompt_text
                    agent_prompts["custom_prompt"] = prompt_text

                state_with_context = { **state, "prompts": agent_prompts }
                
                # --- Pass a mutable copy to the agent ---
                mutable_state_copy = dict(state_with_context)

                logger.info(f"🤖 Starting agent: {agent_id}")
                agent = self.agent_factory.create_agent(agent_id, agent_prompts)
                
                # Load memories just before running
                memory_findings = load_memories.invoke({
                    "agent_id": agent_id,
                    "query": state.get("brief", ""),
                    "file_path": state.get("file_path", "unknown_file")
                })
                mutable_state_copy["retrieved_memories"] = {agent_id: memory_findings}
                
                findings = agent.run(state=mutable_state_copy) # Agent might modify this dict
                logger.info(f"✅ Agent {agent_id} completed.")

                return_updates = {
                    "completed_agents": state.get("completed_agents", []) + [agent_id],
                    "agent_findings": {**state.get("agent_findings", {}), agent_id: findings},
                    "last_completed_agent": agent_id,
                }
                
                # --- Propagate pause flag from agent run to graph state ---
                if mutable_state_copy.get("is_paused_for_email"):
                    return_updates["is_paused_for_email"] = True
                
                return return_updates
            except Exception as e:
                logger.error(f"❌ Agent {agent_id} failed: {e}")
                traceback.print_exc()
                return { 
                    "completed_agents": state.get("completed_agents", []) + [agent_id], 
                    "agent_findings": {**state.get("agent_findings", {}), agent_id: f"Error: {e}"} 
                }
        return agent_node
    
    def _planner_node(self, state: ResearchState) -> ResearchState:
        if state.get("analysis_result") and state.get("agent_queue"):
            logger.info("✅ Execution plan already exists. Resuming workflow by skipping planner.")
            return {}
        logger.info("📋 Supervisor is creating the execution plan for a new workflow...")
        analysis_result = self.supervisor.run(state)
        parsed = json.loads(analysis_result) if isinstance(analysis_result, str) else analysis_result
        agent_queue = parsed.get("required_agents", [])
        logger.info(f"Plan created. Agent queue: {agent_queue}")
        return {"analysis_result": parsed, "agent_queue": agent_queue.copy()}

    def _supervisor_review_node(self, state: ResearchState) -> ResearchState:
        last_agent = state.get("last_completed_agent")
        if last_agent == 'test_case_agent' and 'test_script_agent' in state.get("agent_queue", []):
            logger.info("⏸️ PAUSING workflow for user to select a test case.")
            return {"is_paused_for_input": True}
        return {}

    def _route_after_agent_run(self, state: ResearchState) -> str:
        return "pause_for_input" if state.get("is_paused_for_input") else "continue"
    
    # --- Routing logic after the email agent runs ---
    def _route_after_email_agent(self, state: ResearchState) -> str:
        if state.get("is_paused_for_email"):
            logger.info("Email agent requires user input. Pausing.")
            return "pause"
        else:
            logger.info("Email agent finished. Continuing workflow.")
            return "continue"

    def _route_to_next_agent(self, state: ResearchState) -> str:
        agent_queue = state.get("agent_queue", [])
        if not agent_queue: 
            return "ask_to_save"
        next_agent = agent_queue.pop(0)
        return f"agent_{next_agent}"

    def _ask_to_save_memory_node(self, state: ResearchState) -> ResearchState:
        logger.info("⏸️ PAUSING workflow to ask user about saving memories.")
        return {"is_paused_for_memory_save": True, "findings_to_save": state.get("agent_findings")}

    def _route_after_save_prompt(self, state: ResearchState) -> str:
        return "save_memory" if state.get("user_memory_save_decision") else "synthesize"

    def _memory_saver_node(self, state: ResearchState) -> ResearchState:
        findings_to_save = state.get("findings_to_save", {})
        file_path = state.get("file_path", "unknown_file")
        query = state.get("brief", "No query provided")

        if not findings_to_save: return {}

        logger.info(f"💾 Dynamically preparing findings to save to memory...")
        for agent_id, findings in findings_to_save.items():
            if not findings or agent_id not in AGENT_DB_MAP: continue
            content_to_save = None
            if isinstance(findings, str):
                content_to_save = findings
            elif isinstance(findings, dict):
                if agent_id == 'test_case_agent' or agent_id == 'log_analysis_agent':
                    content_to_save = findings.get('response') or findings.get('tool_results', {}).get('query_rag_store', {}).get('response')
                elif agent_id == 'test_script_agent':
                    content_to_save = findings.get('tool_results', {}).get('generate_test_script', {}).get('test_script') or findings.get('response')
                    
                if not content_to_save:
                    content_to_save = findings.get('response') or findings.get('summary')
            
            if content_to_save:
                try:
                    save_memories.invoke({
                        "agent_id": agent_id, "file_path": file_path,
                        "content": content_to_save, "query": query
                    })
                except Exception as e:
                    logger.error(f"❌ Failed to save memory for {agent_id}: {e}")
            else:
                logger.warning(f"⚠️ No suitable content found to save for agent '{agent_id}'.")
        
        logger.info("✅ Memory saving process complete.")
        return {}

    def _synthesizer_node(self, state: ResearchState) -> ResearchState:
        logger.info("✍️ Synthesizing final report...")
        final_report = self.supervisor.synthesize_findings(state.get("agent_findings", {}))
        return {"final_report": final_report}

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(ResearchState)
        
        # --- NODES ---
        graph.add_node("planner", self._planner_node)
        graph.add_node("agent_router", lambda state: state)
        graph.add_node("supervisor_review", self._supervisor_review_node)
        graph.add_node("ask_to_save_memory", self._ask_to_save_memory_node)
        graph.add_node("memory_saver", self._memory_saver_node)
        graph.add_node("synthesizer", self._synthesizer_node)
        graph.add_node("pause_for_input", lambda state: state)
        graph.add_node("pause_for_email", lambda state: state) # --- NEW NODE ---

        available_agents = [a for a in self.agent_factory.get_all_agent_ids() if a not in ["research_supervisor", "memory_agent"]]
        for agent_id in available_agents:
            graph.add_node(f"agent_{agent_id}", self._create_agent_node(agent_id))
            # --- Create specific edges instead of a generic one ---
            if agent_id == 'log_analysis_agent':
                graph.add_edge(f"agent_{agent_id}", "agent_email_agent")
            else:
                graph.add_edge(f"agent_{agent_id}", "supervisor_review")

        # --- EDGES ---
        graph.set_entry_point("planner")
        graph.add_edge("planner", "agent_router")
        graph.add_edge("memory_saver", "synthesizer")
        graph.add_edge("synthesizer", END)
        graph.add_edge("pause_for_input", END)
        graph.add_edge("pause_for_email", END) # --- NEW EDGE TO END ---

        # --- Conditional edge after email agent runs ---
        graph.add_conditional_edges(
            "agent_email_agent",
            self._route_after_email_agent,
            {
                "pause": "pause_for_email",
                "continue": "supervisor_review",
            }
        )

        graph.add_conditional_edges("supervisor_review", self._route_after_agent_run, 
                                    {"pause_for_input": "pause_for_input", "continue": "agent_router"})
        
        agent_routes = {f"agent_{a}": f"agent_{a}" for a in available_agents}
        agent_routes["ask_to_save"] = "ask_to_save_memory"
        graph.add_conditional_edges("agent_router", self._route_to_next_agent, agent_routes)

        graph.add_conditional_edges("ask_to_save_memory", self._route_after_save_prompt, 
                                    {"save_memory": "memory_saver", "synthesize": "synthesizer"})

        return graph.compile()

    def run_research(self, brief: str, file_path: str, selected_prompts: Dict, analysis_result: Dict, agent_queue: list) -> Dict:
        initial_state = ResearchState(
            messages=[HumanMessage(content=brief)], brief=brief, file_path=file_path or "",
            selected_prompts=selected_prompts or {}, analysis_result=analysis_result,
            agent_queue=agent_queue, completed_agents=[], agent_findings={},
            is_paused_for_email=False # Initialize the new flag
        )
        final_state = self.graph.invoke(initial_state)
        return {"success": True, "final_state": final_state}

    def resume_research(self, paused_state: Dict) -> Dict:
        logger.info("🚀 Resuming research workflow...")
        paused_state["is_paused_for_input"] = False
        paused_state["is_paused_for_memory_save"] = False
        paused_state["is_paused_for_email"] = False
        final_state = self.graph.invoke(paused_state)
        return {"success": True, "final_state": final_state}