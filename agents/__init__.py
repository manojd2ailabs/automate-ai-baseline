import os
import sys
# Import key A2A components
from agents.a2a_factory import A2AAgentFactory
from agents.a2a_system import _global_registry
from agents.agent_executor import ResearchSupervisorAgent

# 🔁 Add parent directory to sys.path (for top-level imports)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Discover and register tools at startup
from tools.registry import discover_and_register_tools, get_all_registered_tools
discover_and_register_tools()

# Public API: A2A supervisor + factory + tool registry
__all__ = [
    "ResearchSupervisorAgent",
    "A2AAgentFactory",
    "get_all_registered_tools",
    "_global_registry"
]
 