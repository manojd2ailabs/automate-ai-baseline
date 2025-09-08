
# graph/research_graph.py

from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from agents.a2a_factory import A2AAgentFactory
import json
import traceback
from graph.research_state import ResearchState
import logging

logger = logging.getLogger(__name__)

class ResearchGraph:
    def __init__(self, prompts: Dict[str, Dict[str, str]], supervisor_agent, agent_factory):
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
                    match_type = selection_info.get("match_type")
                    agent_prompts["system_prompt"] = prompt_text
                    if match_type == "keyword":
                        logger.info(f"[Agent {agent_id}] ✅ Using KEYWORD-matched prompt.")
                        agent_prompts["custom_prompt"] = prompt_text
                    elif match_type == "default":
                        logger.info(f"[Agent {agent_id}] ✅ Using DEFAULT fallback prompt.")
                else:
                    logger.info(f"[Agent {agent_id}] ⚠️ No matching prompt found, using main system_prompt.")

                # This dictionary correctly includes the prompts
                state_with_context = {
                    **state,
                    "prompts": agent_prompts,
                    "previous_findings": state.get("agent_findings", {}),
                    "accumulated_context": state.get("accumulated_context", ""),
                    "coordination_plan": state.get("coordination_plan", "")
                }
                
                logger.info(f"🤖 Starting agent: {agent_id}")
                agent = A2AAgentFactory.create_agent(agent_id, agent_prompts)
                
                # --- THIS IS THE FIX ---
                # Pass the enriched state_with_context to the agent, not the old state.
                findings = agent.run(state=state_with_context)
                
                logger.info(f"✅ Agent {agent_id} completed with findings")

                updated_context = self._update_accumulated_context(state.get("accumulated_context", ""), agent_id, findings)

                return {
                    **state,
                    "completed_agents": state["completed_agents"] + [agent_id],
                    "agent_findings": {**state["agent_findings"], agent_id: findings},
                    "accumulated_context": updated_context,
                    "last_completed_agent": agent_id,
                }
            except Exception as e:
                logger.error(f"❌ Agent {agent_id} failed: {e}")
                return {
                    **state,
                    "completed_agents": state["completed_agents"] + [agent_id],
                    "agent_findings": {**state["agent_findings"], agent_id: f"Error: {e}"},
                    "last_completed_agent": agent_id
                }
        return agent_node

    def _update_accumulated_context(self, current_context: str, agent_id: str, findings) -> str:
        findings_summary = str(findings)[:300] + "..." if len(str(findings)) > 300 else str(findings)
        new_entry = f"\n\n=== {agent_id.upper()} RESULTS ===\n{findings_summary}"
        return (current_context + new_entry).strip()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(ResearchState)
        
        graph.add_node("coordinator", self._coordinator_node)
        graph.add_node("supervisor_review", self._supervisor_review_node)
        graph.add_node("synthesizer", self._synthesizer_node)
        
        available_agents = [a for a in A2AAgentFactory.get_all_agent_ids() if a != "research_supervisor"]
        for agent_id in available_agents:
            graph.add_node(f"agent_{agent_id}", self._create_agent_node(agent_id))
            graph.add_edge(f"agent_{agent_id}", "supervisor_review")
        
        graph.add_edge(START, "coordinator")
        graph.add_edge("synthesizer", END)
        
        graph.add_conditional_edges(
            "supervisor_review",
            self.after_supervisor_review_router,
            {"continue": "coordinator", "stop": END}
        )
        
        routing_options = {f"agent_{a}": f"agent_{a}" for a in available_agents}
        routing_options.update({"synthesizer": "synthesizer", "END": END})
        
        graph.add_conditional_edges("coordinator", self._route_to_next, routing_options)
        
        return graph.compile()

    def after_supervisor_review_router(self, state: ResearchState) -> str:
        if state.get("is_paused_for_input"):
            return "stop"
        else:
            return "continue"

    def _supervisor_review_node(self, state: ResearchState) -> ResearchState:
        last_agent = state.get("last_completed_agent", "")
        completed_agents = state.get("completed_agents", [])
        remaining_agents = [a for a in state.get("required_agents", []) if a not in completed_agents]
        
        logger.info(f"👁️ Supervisor reviewing work from agent: {last_agent}")

        if last_agent == 'test_case_agent' and 'test_script_agent' in remaining_agents:
            logger.info("⏸️ PAUSING workflow for user to select a test case.")
            return {**state, "is_paused_for_input": True}

        review_context = {"completed_agent": last_agent, "agent_findings": state.get("agent_findings", {}), "remaining_agents": remaining_agents}
        review_result = self.supervisor.review_agent_work(review_context)
        review_parsed = json.loads(review_result) if isinstance(review_result, str) else review_result

        return {**state, "next_action": "coordinate"}

    def _coordinator_node(self, state: ResearchState) -> ResearchState:
        agent_queue = state.get("agent_queue", [])
        if not agent_queue:
            return {**state, "next_action": "synthesizer"}
        
        next_agent = agent_queue[0]
        remaining_queue = agent_queue[1:]
        
        return {**state, "current_agent": next_agent, "agent_queue": remaining_queue, "next_action": f"agent_{next_agent}"}

    def _synthesizer_node(self, state: ResearchState) -> ResearchState:
        synthesis_input = {"agent_findings": state.get("agent_findings", {}), "accumulated_context": state.get("accumulated_context", ""), "original_brief": state.get("brief", "")}
        final_report = self.supervisor.synthesize_findings(synthesis_input)
        return {**state, "final_report": final_report, "next_action": "END"}

    def _route_to_next(self, state: ResearchState) -> str:
        return state.get("next_action", "END")

    def run_research(self, brief: str, file_path: str, selected_prompts: Dict[str, str], analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("🚀 Starting research workflow with pre-computed plan...")
        parsed = analysis_result
        initial_state = ResearchState(
            messages=[HumanMessage(content=brief)], brief=brief, file_path=file_path or "",
            selected_prompts=selected_prompts or {}, analysis_result=parsed,
            required_agents=parsed.get("required_agents", []), topics=parsed.get("topics", {}),
            coordination_plan=parsed.get("coordination_plan", ""),
            agent_queue=parsed.get("required_agents", []).copy(),
            completed_agents=[], agent_findings={}, final_report="", is_paused_for_input=False,
            selected_test_case=None
        )
        final_state = self.graph.invoke(initial_state)
        logger.info("🏁 Research workflow completed")
        return {"success": True, "final_state": final_state}

    def resume_research(self, paused_state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("🚀 Resuming research workflow...")
        paused_state["is_paused_for_input"] = False
        final_state = self.graph.invoke(paused_state)
        logger.info("🏁 Resumed research workflow completed")
        return {"success": True, "final_state": final_state}