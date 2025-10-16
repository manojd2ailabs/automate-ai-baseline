# main2.py

"""
Description:
AgenticAIApp is a Streamlit-based interface for orchestrating multi-agent AI workflows 
built around ResearchSupervisorAgent, ResearchGraph, and A2AAgentFactory. It allows users 
to upload files (PCAP, TXT, PDF, CSV, JSON, LOG), submit queries, and have the AI agents 
analyze the data, generate test scripts, and optionally send email reports. The app supports 
pausing and resuming workflows for user decisions such as selecting test cases, approving 
generated responses, and saving memories for context-aware reasoning.

Key Features:
- Upload and process files with user-defined queries.
- Dynamic prompt selection based on user query keywords and agent capabilities.
- Async execution of agent workflows with ResearchGraph.
- Integrates feedback loops for memory saving and context-aware decision making.
- Supports email notifications with pre-filled configurations.
- Customizable Streamlit UI with professional styling and interactive components.
- Maintains session state for workflow pausing, resuming, and iterative user interactions.

Created By: Pediredla Sai Ram, Parise Hari Sai, Tanvi Dongaonkar ,Jayant Sharma
Date: 
Modified By:
Reason:
Example:
1. User uploads a pcap file and request analysis.
2. Agents run asynchronously, producing findings and selects Sub agents
4. AI responses are displayed; user can save memory or send an email report.
"""


import asyncio
import streamlit as st
from typing import Dict, Any
import os
import json
import re
from pathlib import Path
# Assume these imports are correctly pointing to your project structure
from agents.agent_executor import ResearchSupervisorAgent
from agents.a2a_factory import A2AAgentFactory
from agents.a2a_system import _global_registry
from graph.research_graph import ResearchGraph
from config import load_prompts, load_email_config
from tools.registry import get_all_registered_tools
 
 
class AgenticAIApp:
    def __init__(self):
        self.prompts = load_prompts()
        supervisor_card = _global_registry.get_agent_card("research_supervisor")
        self.supervisor_agent = ResearchSupervisorAgent(supervisor_card, self.prompts)
        self.agent_factory = A2AAgentFactory()
        self.graph = ResearchGraph(self.prompts, self.supervisor_agent, self.agent_factory)
 
    def _tokenize_key(self, key: str) -> set:
        words = re.split(r'[_\-]+', key)
        final_tokens = set()
        for word in words:
            tokens_from_word = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\s|$)', word)
            for token in tokens_from_word:
                final_tokens.add(token.lower())
        final_tokens.discard("prompt")
        return final_tokens
 
    def _split_test_cases(self,text: str) -> list[str]:
        """
        Splits a string into a list of test cases.
        This version handles test case headers both with and without a hyphen,
        e.g., "Test Case - 1" and "Test Case 1".
        """
        pattern = r'(Test\s*Case\s+(?:-\s*)?\d+.*?)(?=Test\s*Case\s+(?:-\s*)?\d+|$)'
        return [m.strip() for m in re.findall(pattern, text, flags=re.DOTALL)]
   
    def match_prompts_to_query(self, user_query: str, required_agents: list) -> Dict[str, Dict[str, str]]:
        user_query_lower = user_query.lower()
        final_selections = {}
 
        for agent_id in required_agents:
            agent_prompts = self.prompts.get(agent_id, {})
            if not agent_prompts:
                continue
 
            matched_info = None
            for key, text in agent_prompts.items():
                if key.strip().lower() in ["system_prompt", "default"]:
                    continue
                key_tokens = self._tokenize_key(key)
                if any(token in user_query_lower for token in key_tokens):
                    matched_info = { "prompt_text": text, "match_type": "keyword" }
                    break
 
            if not matched_info and "default" in (k.lower() for k in agent_prompts.keys()):
                matched_info = { "prompt_text": agent_prompts.get("Default", agent_prompts.get("default")), "match_type": "default" }
 
            if matched_info:
                final_selections[agent_id] = matched_info
 
        return final_selections
 
    async def process_query(self, file_path: str, user_query: str) -> Dict[str, Any]:
        """Initiates a new research task."""
        try:
            with st.spinner("Analyzing request and creating execution plan..."):
                analysis_json = self.supervisor_agent.run({"brief": user_query, "file_path": file_path})
                analysis_result = json.loads(analysis_json) if isinstance(analysis_json, str) else analysis_json
            st.session_state['execution_plan'] = analysis_result.get('required_agents', [])
            st.info(f"Execution Plan: Running agents -> {analysis_result.get('required_agents', [])}")
           
            required_agents = analysis_result.get("required_agents", [])
            selected_prompts = self.match_prompts_to_query(user_query, required_agents)
 
            return await asyncio.to_thread(
                self.graph.run_research,
                brief=user_query,
                file_path=file_path,
                selected_prompts=selected_prompts,
                analysis_result=analysis_result,
                agent_queue=required_agents
            )
        except Exception as e:
            st.error(f"Error during initial processing: {e}")
            return {"success": False, "error": str(e)}
 
    async def resume_with_test_case(self, paused_state: Dict[str, Any], selected_case: str) -> Dict[str, Any]:
        """Resumes the graph after a test case has been selected."""
        if not paused_state:
            return {"success": False, "error": "Cannot resume, no paused state found."}
        paused_state['selected_test_case'] = selected_case
        return await asyncio.to_thread(self.graph.resume_research, paused_state)
 
    async def resume_with_memory_decision(self, paused_state: Dict[str, Any], decision: bool) -> Dict[str, Any]:
        """Resumes the graph after the user decides whether to save memories."""
        if not paused_state:
            return {"success": False, "error": "Cannot resume, no paused state found."}
        paused_state['user_memory_save_decision'] = decision
        return await asyncio.to_thread(self.graph.resume_research, paused_state)
 
    # --- NEW: Function to resume after email decision ---
    async def resume_after_email_prompt(self, paused_state: Dict[str, Any]) -> Dict[str, Any]:
        """Resumes the graph after the user makes a decision about sending an email."""
        if not paused_state:
            return {"success": False, "error": "Cannot resume, no paused state found."}
        return await asyncio.to_thread(self.graph.resume_research, paused_state)

    def _apply_custom_styling(self):
        st.markdown("""<style>
                .stApp {
                    background: linear-gradient(to right bottom, #0d1b2a, #1b263b, #415a77);
                    background-attachment: fixed; color: #FFFFFF !important;
                }
                div[data-testid="stHeader"] { background-color: #0d1b2a !important; }
                div[data-testid="stAlert"] {
                    background-color: rgba(119, 141, 169, 0.2) !important;
                    border: 1px solid #778da9 !important; border-radius: 8px !important;
                }
                div[data-testid="stAlert"] div { color: #EAEAEA !important; }
                h1, h2, h3, h4, h5, h6 { color: #FFFFFF !important; }
                div[data-testid="stMarkdown"] p, div[data-testid="stMarkdown"] li,
                label[data-testid="stWidgetLabel"], div[data-testid="stText"] { color: #FFFFFF !important; }
                hr { margin-top: 0.75rem !important; margin-bottom: 1rem !important; }
                h3 { margin-top: 0 !important; }
                .stTextInput input, .stTextArea textarea {
                    background-color: #1b263b; border: 1px solid #778da9;
                    color: #FFFFFF; border-radius: 5px;
                }
                ::placeholder { color: #bdc3c7 !important; opacity: 1; }
                .stButton > button {
                    background-color: #415A77; color: #FFFFFF !important;
                    border: 2px solid #778DA9; border-radius: 8px;
                    font-weight: bold; transition: all 0.3s ease;
                }
                .stButton > button:hover { background-color: #778DA9; border-color: #FFFFFF; }
                
                /* 1. Make spinner more visible */
                div[data-testid="stSpinner"] > div {
                    border-top-color: #E67E22 !important;
                    border-left-color: #E67E22 !important;
                }
    
                /* 2. Remove border from drag-and-drop box */
                [data-testid="stFileUploader"] {
                    border: none;
                    background-color: transparent;
                    padding: 0;
                }
    
                /* 3. Style the "Browse files" button to be orange */
                [data-testid="stFileUploader"] section button {
                    background-color: #E67E22 !important;
                    border: 2px solid #D35400 !important;
                    color: #FFFFFF !important;
                    border-radius: 8px;
                    font-weight: bold;
                    transition: all 0.3s ease;
                }
                [data-testid="stFileUploader"] section button:hover {
                    background-color: #F39C12 !important;
                    border-color: #E67E22 !important;
                }
            
                /* Style for the uploaded file name display */
                [data-testid="stFileUploaderFileName"] {
                    color: #FFFFFF;
                }
                /* --- END OF MODIFICATIONS --- */
                    
                /* Professional button styling for primary actions */
                .stButton > button[kind="primary"] {
                    background-color: #2E7D32 !important;
                    border-color: #4CAF50 !important;
                }
                .stButton > button[kind="primary"]:hover {
                    background-color: #4CAF50 !important;
                    border-color: #66BB6A !important;
                }
                    
                </style>""", unsafe_allow_html=True)
 
    def run_streamlit_app(self):
        st.set_page_config(layout="wide")
        self._apply_custom_styling()
 
        if 'rag_response' not in st.session_state: st.session_state['rag_response'] = None
        if 'split_test_cases' not in st.session_state: st.session_state['split_test_cases'] = []
        if 'selected_testcase' not in st.session_state: st.session_state['selected_testcase'] = None
        if 'generated_script' not in st.session_state: st.session_state['generated_script'] = None
        if 'paused_graph_state' not in st.session_state: st.session_state['paused_graph_state'] = None
        if 'final_report' not in st.session_state: st.session_state['final_report'] = None
        if 'uploaded_filename' not in st.session_state: st.session_state['uploaded_filename'] = ""
        if 'original_query' not in st.session_state: st.session_state['original_query'] = ""
       
        st.title("Agentic AiTest")
        st.markdown("Upload a file, describe your goal, and let the AI agents handle the rest.")
        st.divider()
 
        with st.container():
            st.subheader("1. Provide your file and instructions")
            user_query = st.text_input("What do you want to do with the uploaded file?", placeholder="e.g., analyze this pcap for suspicious traffic")
            uploaded_file = st.file_uploader("Upload your file", type=['pcap', 'txt', 'log', 'json', 'csv', 'pdf'])
           
            if st.button("▶️ Run Analysis", use_container_width=True):
                st.session_state.clear()
                if uploaded_file and user_query:
                    st.session_state['original_query'] = user_query
                    st.session_state['uploaded_filename'] = uploaded_file.name
                    temp_dir = "temp_uploads"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    temp_path = f"temp_{uploaded_file.name}"
                    try:
                        with open(temp_path, "wb") as f: f.write(uploaded_file.getvalue())
                        run_output = asyncio.run(self.process_query(temp_path, user_query))
                       
                        if run_output and run_output.get("success"):
                            st.session_state['paused_graph_state'] = run_output.get("final_state", {})
                            st.rerun()
                        else:
                            st.error(f"Processing failed: {run_output.get('error', 'Unknown error')}")
                   
                    except Exception as e:
                        st.error(f"Application error: {str(e)}")
                    
        paused_state = st.session_state.get('paused_graph_state')
        if paused_state:
            findings = paused_state.get("agent_findings", {})
            response_found = False
            for agent_data in findings.values():
                if isinstance(agent_data, dict):
                    tool_results = agent_data.get("tool_results", {})
                    for tool_data in tool_results.values():
                        if isinstance(tool_data, dict) and "response" in tool_data:
                            st.session_state['rag_response'] = tool_data["response"]
                            response_found = True
                            break
                if response_found: break

        if st.session_state.get('rag_response'):
            with st.container(border=True):
                st.subheader("🤖 Agent Response")
                st.markdown(st.session_state['rag_response'])

        # --- NEW: Auto Email Status ---
        if paused_state:
            email_findings = paused_state.get("agent_findings", {}).get("email_agent", {})
            if isinstance(email_findings, dict):
                email_summary = email_findings.get("summary", "")
                if email_summary and "Email sent" in email_summary:
                    with st.container(border=True):
                        st.subheader("📧 Email Notification")
                        st.success(f"✅ {email_summary}")

        # --- NEW: UI for Email Decision Pause State ---
        if paused_state and paused_state.get('is_paused_for_email'):
            with st.container(border=True):
                st.subheader("📧 Email Notification")
                st.info("No critical Events were found in the analysis. Do you want to send the report via email anyway?")

                # Pre-fill values from config
                email_cfg = load_email_config()
                default_recipients = ", ".join(email_cfg.get("default_recipients", []))

                recipients_text = st.text_input(
                    "Recipients (comma-separated)", 
                    value=default_recipients
                )
                subject_text = st.text_input(
                    "Subject", 
                    value=f"AgenticAI Analysis Report - {st.session_state.get('uploaded_filename', 'Analysis')}"
                )
                body_text = st.text_area(
                    "Email Body", 
                    value=st.session_state.get("rag_response", ""), 
                    height=200
                )

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("✅ Yes, Send Email", use_container_width=True, type="primary"):
                        with st.spinner("Sending email..."):
                            all_tools = get_all_registered_tools()
                            send_tool = all_tools.get("send_email")
                            if send_tool:
                                payload = {
                                    "recipients": [r.strip() for r in recipients_text.split(",") if r.strip()],
                                    "subject": subject_text,
                                    "body": body_text,
                                    "query": st.session_state.get('original_query', ''),
                                    "filename": st.session_state.get('uploaded_filename', ''),
                                }
                                resp = send_tool.invoke(payload)
                                if isinstance(resp, dict) and resp.get("success"):
                                    st.success("✅ Email sent successfully!")
                                else:
                                    err = resp.get("error") if isinstance(resp, dict) else str(resp)
                                    st.error(f"Failed to send email: {err}")

                            # Resume the graph
                            resume_output = asyncio.run(self.resume_after_email_prompt(paused_state))
                            if resume_output and resume_output.get("success"):
                                st.session_state['paused_graph_state'] = resume_output.get("final_state", {})
                            st.rerun()

                with col2:
                    if st.button("❌ No, Just Continue", use_container_width=True):
                        with st.spinner("Continuing workflow..."):
                            resume_output = asyncio.run(self.resume_after_email_prompt(paused_state))
                            if resume_output and resume_output.get("success"):
                                st.session_state['paused_graph_state'] = resume_output.get("final_state", {})
                            st.rerun()


        if paused_state and paused_state.get('is_paused_for_input'):
            with st.container(border=True):
                st.subheader("🧪 Choose a Test Case to Generate a Script")
                if not st.session_state.get('split_test_cases') and st.session_state.get('rag_response'):
                    st.session_state['split_test_cases'] = self._split_test_cases(st.session_state['rag_response'])
               
                test_cases = st.session_state.get('split_test_cases', [])
                if test_cases:
                    cols = st.columns(min(5, len(test_cases)))
                    for i, case in enumerate(test_cases):
                        case_id = f"Test Case {i + 1}"
                        if cols[i % len(cols)].button(case_id, key=f"button_{i}", use_container_width=True):
                            st.session_state['selected_testcase'] = case
                            st.rerun()
                else:
                    st.warning("No test cases were found in the agent's response.")
 
        if st.session_state.get('selected_testcase') and not st.session_state.get('generated_script'):
             with st.container(border=True):
                st.subheader("📝 Edit Selected Test Case")
                edited_testcase = st.text_area("You can edit the test case below before generating the script:", st.session_state['selected_testcase'], height=250)
                if st.button("Generate Test Script", use_container_width=True):
                    with st.spinner("Resuming workflow and generating script..."):
                        resume_output = asyncio.run(self.resume_with_test_case(paused_state, edited_testcase))
                        if resume_output and resume_output.get("success"):
                            st.session_state['paused_graph_state'] = resume_output.get("final_state", {})
                            agent_findings = st.session_state['paused_graph_state'].get("agent_findings", {})
                            script_findings = agent_findings.get("test_script_agent", {})
 
                            if isinstance(script_findings, dict):
                                tool_results = script_findings.get("tool_results", {}).get("generate_test_script", {})
                                if isinstance(tool_results, dict) and tool_results.get("test_script"):
                                    st.session_state['generated_script'] = tool_results.get("test_script")
                                else:
                                     st.warning("Could not find the generated test script in the agent's response.")
                            st.rerun()
 
        if st.session_state.get('generated_script'):
            with st.container(border=True):
                st.subheader("🐍 Generated Test Script")
                st.code(st.session_state['generated_script'], language="python")
 
        if paused_state and paused_state.get('is_paused_for_memory_save'):
            with st.container(border=True):
                st.subheader("💭 Response Feedback")
                
                st.markdown("""
                **Would you prefer this response?**
                
                Your feedback helps us improve AI agent performance and determines whether 
                these findings should be saved for future reference.
                """)
                
                # Professional thumbs layout
                col1, col2, col3, col4 = st.columns([1, 3, 3, 1])
                
                with col2:
                    if st.button("👍 Yes, I prefer this response", 
                               key="save_findings", 
                               help="This response was helpful - save findings and complete",
                               type="primary",
                               use_container_width=True):
                        with st.spinner("💾 Saving preferred response to knowledge base..."):
                            resume_output = asyncio.run(self.resume_with_memory_decision(paused_state, True))
                            if resume_output and resume_output.get("success"):
                                st.session_state['final_report'] = resume_output.get("final_state", {}).get('final_report')
                                st.session_state['paused_graph_state'] = None
                                st.success("✅ Thank you! Response saved for future improvements.")
                                st.rerun()
                
                with col3:
                    if st.button("👎 No, I don't prefer this response", 
                               key="skip_save", 
                               help="This response needs improvement - complete without saving",
                               type="primary",
                               use_container_width=True):
                        with st.spinner("📋 Completing analysis without saving..."):
                            resume_output = asyncio.run(self.resume_with_memory_decision(paused_state, False))
                            if resume_output and resume_output.get("success"):
                                st.session_state['final_report'] = resume_output.get("final_state", {}).get('final_report')
                                st.session_state['paused_graph_state'] = None
                                st.info("📝 Feedback noted. We'll work on improving similar responses.")
                                st.rerun()
                
                st.markdown("---")
                st.caption("💡 Preferred responses are saved to help AI agents learn and provide better analysis in the future.")
 

 
if __name__ == "__main__":
    app = AgenticAIApp()
    app.run_streamlit_app()