"""
Description:
A2AAgentFactory is a centralized factory for dynamically creating and managing agents 
within the global A2A system. It retrieves agent metadata from the global registry, 
validates their status, and instantiates agent classes based on definitions in `agent_card.json`. 
The factory also supports agent discovery and ID retrieval for orchestration by supervisor agents.

Created By: Tanvi Dongaonkar.
Date: 
Modified By:
Reason:
Example:
Used by the ResearchSupervisorAgent to dynamically instantiate sub-agents such as 
LogAnalysisAgent, TestCaseAgent, or EmailAgent using metadata from `agent_card.json`.
"""

from agents.a2a_system import _global_registry

class A2AAgentFactory:
    @staticmethod
    def create_agent(agent_id: str, prompts: dict = None):
        card = _global_registry.get_agent_card(agent_id)
        if not card or card.status != "active":
            raise ValueError(f"Agent '{agent_id}' is not active or found")
        cls = _global_registry.load_agent_class(card)
        return cls(card, prompts)

    @staticmethod
    def get_all_agent_ids():
        return list(_global_registry.discover_agents().keys())

    @staticmethod
    def discover_agents():
        return {k: v.to_dict() for k, v in _global_registry.discover_agents().items()}
 