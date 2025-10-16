
"""
Description:
This module defines the `ResearchState` TypedDict, which represents the shared state 
object used across all agents and workflows in the Agentic AI system. It maintains 
conversation history, agent coordination data, intermediate findings, and control flags 
for pausing, resuming, and handoff between agents.

It ensures consistent context tracking for supervisor-driven orchestration, 
multi-agent coordination, and memory-aware reasoning during execution.

Created By: Tanvi Dongaonkar , Pediredla Sai Ram, Parise Hari Sai, Jayant Sharma
Date:
Modified By:
Reason:
Example:
Used by `research_graph.py` and all sub-agents to maintain and update the global 
workflow state, including messages, agent queue, analysis results, and user feedback.
"""

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

    # --- MODIFIED: Added is_paused_for_email ---
    is_paused_for_input: bool                  
    selected_test_case: Optional[str] 
    is_paused_for_email: bool

    # New fields for memory agent workflow
    is_paused_for_memory_save: bool            
    user_memory_save_decision: Optional[bool]  
    findings_to_save: Optional[Dict[str, Any]] 
    retrieved_memories: Optional[Dict[str, Any]] # UPDATED: To store memories from DB
    findings_for_current_agent: Optional[Dict[str, Any]]