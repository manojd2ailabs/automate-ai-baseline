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
 