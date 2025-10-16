"""
Description:
This module provides centralized management of agents and their metadata for the Agentic AI system. 
It loads agent configurations from `agent_cards.json`, injects system prompts from `prompts.yml`.

The global registry (`_global_registry`) allows supervisor agents and factories to:
- Discover active agents
- Retrieve metadata via AgentCard
- Load corresponding classes dynamically for orchestration

Created By: Tanvi Dongaonkar
Date: 
Modified By:
Reason:
Example:
Used by A2AAgentFactory and supervisor agents to fetch agent metadata and instantiate agents 
like LogAnalysisAgent, TestCaseAgent, or EmailAgent based on definitions in `agent_cards.json`.
"""

import json
import os
import importlib
from typing import Dict, Any, Optional
from dataclasses import dataclass
from config import load_prompts

@dataclass
class AgentCard:
    id: str
    name: str
    description: str
    class_path: str
    system_prompt: str
    capabilities: list
    supported_file_types: list
    version: str
    status: str
    available_tools: Optional[list] = None
    coordinates_agents: Optional[list] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentCard':
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

class A2ARegistry:
    def __init__(self, cards_file: str = None):
        self.cards_file = cards_file or os.path.join(os.path.dirname(__file__), 'agent_cards.json')
        self.agent_cards: Dict[str, AgentCard] = {}
        self.agent_classes: Dict[str, type] = {}
        self._load_agent_cards()

    def _load_agent_cards(self):
        with open(self.cards_file, 'r') as f:
            data = json.load(f)

        prompts = load_prompts()  # Load all prompts from prompts.yml

        for agent_id, agent_data in data['agents'].items():
            # Inject system prompt from YAML
            agent_data['system_prompt'] = prompts.get(agent_id, {}).get('system_prompt', '')
            card = AgentCard.from_dict(agent_data)
            self.agent_cards[agent_id] = card

    def discover_agents(self) -> Dict[str, AgentCard]:
        return {k: v for k, v in self.agent_cards.items() if v.status == "active"}

    def get_agent_card(self, agent_id: str) -> Optional[AgentCard]:
        return self.agent_cards.get(agent_id)

    def load_agent_class(self, agent_card: AgentCard):
        if agent_card.id in self.agent_classes:
            return self.agent_classes[agent_card.id]
        module_path, class_name = agent_card.class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        self.agent_classes[agent_card.id] = cls
        return cls

# Global instance
_global_registry = A2ARegistry()
 