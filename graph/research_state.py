
from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage

class ResearchState(TypedDict):
    # Existing core fields
    messages: List[BaseMessage]
    brief: str
    file_path: str
    selected_prompts: Dict[str, Any]
    required_agents: List[str]
    topics: Dict[str, Any]
    coordination_plan: str
    agent_findings: Dict[str, Any]
    current_agent: str
    completed_agents: List[str]
    final_report: str
    next_action: str
    analysis_result: Optional[Dict[str, Any]]

    # New fields for proper agent handoff
    accumulated_context: str                    # Context passed between agents
    agent_queue: List[str]                     # Queue of remaining agents to execute
    last_completed_agent: str                  # Track the last agent that completed
    supervisor_feedback: str                   # Feedback from supervisor reviews
    previous_findings: Optional[Dict[str, Any]] # Previous agent results for context

    # New fields for pausing and resuming
    is_paused_for_input: bool                  # Flag to signal the graph should pause for UI input
    selected_test_case: Optional[str]          # To store the test case selected by the user