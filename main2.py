import asyncio
import streamlit as st
from typing import Dict, Any
import os
import json
import re

# Assume these imports are correctly pointing to your project structure
from agents.agent_executor import ResearchSupervisorAgent
from agents.a2a_factory import A2AAgentFactory
from agents.a2a_system import _global_registry
from graph import ResearchGraph
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

    def _split_test_cases(self, text: str) -> list[str]:
        pattern = r'(Test\s*Case\s*-\s*\d+.*?)(?=Test\s*Case\s*-\s*\d+|$)'
        return [m.strip() for m in re.findall(pattern, text, flags=re.DOTALL)]

    def match_prompts_to_query(self, user_query: str, required_agents: list) -> Dict[str, Dict[str, str]]:
        user_query_lower = user_query.lower()
        final_selections = {}

        for agent_id in required_agents:
            agent_prompts = self.prompts.get(agent_id, {})
            if not agent_prompts:
                continue

            matched_info = None
            matched_keywords = []

            for key, text in agent_prompts.items():
                if key.strip().lower() in ["system_prompt", "default"]:
                    continue

                key_tokens = self._tokenize_key(key)

                for token in key_tokens:
                    if token in user_query_lower:
                        matched_keywords.append(token)

                if matched_keywords:
                    matched_info = {
                        "prompt_text": text,
                        "match_type": "keyword",
                        "matched_keywords": matched_keywords
                    }
                    break

            if not matched_info and "default" in (k.lower() for k in agent_prompts.keys()):
                matched_info = {
                    "prompt_text": agent_prompts.get("Default", agent_prompts.get("default")),
                    "match_type": "default",
                    "matched_keywords": []
                }

            if matched_info:
                final_selections[agent_id] = matched_info

        return final_selections

    async def process_query(self, file_path: str, user_query: str) -> Dict[str, Any]:
        try:
            with st.spinner("Analyzing request and creating execution plan..."):
                analysis_json = self.supervisor_agent.run({"brief": user_query, "file_path": file_path})
                analysis_result = json.loads(analysis_json) if isinstance(analysis_json, str) else analysis_json

            st.info(f"Execution Plan: Running agents -> {analysis_result.get('required_agents', [])}")
            required_agents = analysis_result.get("required_agents", [])
            selected_prompts = self.match_prompts_to_query(user_query, required_agents)

            return await asyncio.to_thread(
                self.graph.run_research,
                brief=user_query,
                file_path=file_path,
                selected_prompts=selected_prompts,
                analysis_result=analysis_result
            )
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def resume_with_test_case(self, paused_state: Dict[str, Any], selected_case: str) -> Dict[str, Any]:
        if not paused_state:
            return {"success": False, "error": "Cannot resume, no paused state found."}

        paused_state['selected_test_case'] = selected_case
        return await asyncio.to_thread(self.graph.resume_research, paused_state)

    def _apply_custom_styling(self):
        """
        Injects custom CSS to style the header, info box, and other components.
        """
        st.markdown("""
            <style>
                /* Top menu bar */
                header, .css-1v3fvcr {
                    background-color: #1c1f26 !important;
                    color: #e0e0e0 !important;
                }
                
                /* Hamburger menu and text */
                .css-1v3fvcr div {
                    color: #e0e0e0 !important;
                }
                /* Set a high-contrast default text color for the entire app */
                .stApp {
                    background: linear-gradient(to right bottom, #0d1b2a, #1b263b, #415a77);
                    background-attachment: fixed;
                    color: #FFFFFF !important;
                }
                
                /* Style the top header bar to match the dark theme */
                div[data-testid="stHeader"] {
                    background-color: #0d1b2a !important;
                }

                /* Style the st.info box for better visibility */
                div[data-testid="stAlert"] {
                    background-color: rgba(119, 141, 169, 0.2) !important;
                    border: 1px solid #778da9 !important;
                    border-radius: 8px !important;
                }
                div[data-testid="stAlert"] div { /* Target the text inside */
                    color: #EAEAEA !important;
                }

                /* Force all headers to be bright white */
                h1, h2, h3, h4, h5, h6 {
                    color: #FFFFFF !important;
                }

                /* Target specific Streamlit elements to ensure their text is white */
                div[data-testid="stMarkdown"] p,
                div[data-testid="stMarkdown"] li,
                label[data-testid="stWidgetLabel"],
                div[data-testid="stText"] {
                    color: #FFFFFF !important;
                }
                
                /* Reduce spacing around the main divider */
                hr {
                    margin-top: 0.75rem !important;
                    margin-bottom: 1rem !important;
                }
                h3 {
                    margin-top: 0 !important;
                }

                /* --- Input & Widget Styling --- */
                .stTextInput input, .stTextArea textarea {
                    background-color: #1b263b;
                    border: 1px solid #778da9;
                    color: #FFFFFF;
                    border-radius: 5px;
                }

                /* Make placeholder text visible */
                ::placeholder {
                    color: #bdc3c7 !important;
                    opacity: 1;
                }
                
                /* This uses the exact class name from your browser's HTML */
                .st-emotion-cache-1gulkj5 {
                    background-color: #1b263b !important;
                    border: 2px dashed #778da9 !important;
                    border-radius: 5px !important;
                }
                /* This makes all text and icons inside the dropzone white */
                .st-emotion-cache-1gulkj5 * {
                    color: #FFFFFF !important;
                }
                /* This styles the 'Browse files' button to be orange */
                .st-emotion-cache-1gulkj5 button {
                    background-color: #F39C12 !important;
                    color: #FFFFFF !important;
                    border: none !important;
                    transition: all 0.3s ease;
                }
                .st-emotion-cache-1gulkj5 button:hover {
                    background-color: #E67E22 !important;
                }
                
                /* This targets the label and any text element inside it */
                div[data-testid="stCheckbox"] label,
                div[data-testid="stCheckbox"] label p,
                div[data-testid="stCheckbox"] label span {
                    color: #FFFFFF !important;
                }

                /* --- Button Styling --- */
                .stButton > button {
                    background-color: #415A77;
                    color: #FFFFFF !important;
                    border: 2px solid #778DA9;
                    border-radius: 8px;
                    font-weight: bold;
                    transition: all 0.3s ease;
                }
                .stButton > button:hover {
                    background-color: #778DA9;
                    border-color: #FFFFFF;
                }
            </style>
        """, unsafe_allow_html=True)

    def run_streamlit_app(self):
        st.set_page_config(layout="wide")
        self._apply_custom_styling()

        # --- SESSION STATE INITIALIZATION ---
        if 'rag_response' not in st.session_state: st.session_state['rag_response'] = None
        if 'split_test_cases' not in st.session_state: st.session_state['split_test_cases'] = []
        if 'selected_testcase' not in st.session_state: st.session_state['selected_testcase'] = None
        if 'generated_script' not in st.session_state: st.session_state['generated_script'] = None
        if 'paused_graph_state' not in st.session_state: st.session_state['paused_graph_state'] = None
        if 'uploaded_filename' not in st.session_state: st.session_state['uploaded_filename'] = ""
        if 'original_query' not in st.session_state: st.session_state['original_query'] = ""
        # vvv --- NEW SESSION STATE KEY --- vvv
        if 'responding_agent_id' not in st.session_state: st.session_state['responding_agent_id'] = None
        # ^^^ --- END OF NEW KEY --- ^^^
        
        # --- HEADER ---
        st.title("🚀 Agentic AI - Intelligent File Processor")
        st.markdown("Upload a file, describe your goal, and let the AI agents handle the rest.")
        st.divider()

        # --- INPUT SECTION ---
        with st.container():
            st.subheader("1. Provide your file and instructions")
            
            user_query = st.text_input(
                "What do you want to do with the uploaded file?",
                placeholder="e.g., analyze this pcap for suspicious traffic"
            )
            uploaded_file = st.file_uploader(
                "Upload your file",
                type=['pcap', 'txt', 'log', 'json', 'csv', 'pdf']
            )
            
            if st.button("▶️ Run Analysis", use_container_width=True):
                st.session_state.clear()
                if uploaded_file and user_query:
                    # Store the original query and filename for email purposes
                    st.session_state['original_query'] = user_query
                    st.session_state['uploaded_filename'] = uploaded_file.name
                    
                    temp_path = f"temp_{uploaded_file.name}"
                    try:
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())

                        run_output = asyncio.run(self.process_query(temp_path, user_query))

                        if run_output and run_output.get("success"):
                            result = run_output.get("final_state", {})

                            if result.get("is_paused_for_input"):
                                st.session_state['paused_graph_state'] = result
                                st.info("✅ Workflow paused. Select a test case to continue.")

                            findings = result.get("agent_findings", {})
                            response_found = False
                            for agent_id, agent_data in findings.items():
                                if isinstance(agent_data, dict):
                                    tool_results = agent_data.get("tool_results", {})
                                    for tool_name, tool_data in tool_results.items():
                                        if isinstance(tool_data, dict) and "response" in tool_data:
                                            st.session_state['rag_response'] = tool_data["response"]
                                            # vvv --- SAVE THE AGENT ID --- vvv
                                            st.session_state['responding_agent_id'] = agent_id
                                            # ^^^ ----------------------- ^^^
                                            response_found = True
                                            break
                                if response_found:
                                    break
                            
                            if not response_found:
                                st.warning("Workflow complete, but no specific response was generated for display.")
                                st.json(findings)
                        else:
                            st.error(f"Processing failed: {run_output.get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Application error: {str(e)}")
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
        
        # --- AGENT RESPONSE SECTION ---
        if st.session_state.get('rag_response'):
            with st.container(border=True):
                # vvv --- DYNAMIC TITLE LOGIC --- vvv
                agent_id = st.session_state.get('responding_agent_id')
                if agent_id:
                    # Format the agent_id into a nice title, e.g., "log_analysis_agent" -> "Log Analysis Response"
                    title_text = agent_id.replace('_', ' ').replace(' agent', '').title() + " Response"
                    st.subheader(f"🤖 {title_text}")
                else:
                    st.subheader("🤖 Agent Response") # Fallback if agent_id isn't found
                # ^^^ --- END OF DYNAMIC TITLE LOGIC --- ^^^
                
                st.markdown(st.session_state['rag_response'])

        # --- TEST CASE UI SECTION ---
        if st.session_state.get('paused_graph_state') and st.session_state['paused_graph_state'].get('is_paused_for_input'):
            with st.container(border=True):
                st.subheader("🧪 Choose a Test Case to Generate a Script")
                if not st.session_state.get('split_test_cases') and st.session_state.get('rag_response'):
                    st.session_state['split_test_cases'] = self._split_test_cases(st.session_state['rag_response'])
                
                test_cases = st.session_state.get('split_test_cases', [])
                if test_cases:
                    num_cases = len(test_cases)
                    cols = st.columns(min(5, num_cases))
                    for i, case in enumerate(test_cases):
                        match = re.search(r'(Test\s*Case\s*-\s*\d+)', case)
                        case_id = match.group(1) if match else f"Test Case {i + 1}"
                        if cols[i % len(cols)].button(case_id, key=f"button_{i}", use_container_width=True):
                            st.session_state['selected_testcase'] = case
                            st.session_state['generated_script'] = None
                            st.rerun()
                else:
                    st.warning("No test cases were found in the agent's response to select from.")

        # --- SELECTED TEST CASE & SCRIPT GENERATION ---
        if st.session_state.get('selected_testcase'):
            with st.container(border=True):
                st.subheader("📝 Edit Selected Test Case")
                edited_testcase = st.text_area(
                    "You can edit the test case below before generating the script:",
                    st.session_state['selected_testcase'],
                    height=250,
                    key="edited_testcase_area"
                )
                if st.button("Generate Test Script", key="generate_script", use_container_width=True):
                    if st.session_state.get('paused_graph_state'):
                        with st.spinner("Resuming workflow and generating script..."):
                            resume_output = asyncio.run(self.resume_with_test_case(st.session_state['paused_graph_state'], edited_testcase))
                        
                        if resume_output and resume_output.get("success"):
                            final_state = resume_output.get("final_state", {})
                            findings = final_state.get("agent_findings", {})
                            script_agent_result = findings.get("test_script_agent", {})
                            tool_output = script_agent_result.get("tool_results", {}).get("generate_test_script", {})

                            if tool_output.get("success"):
                                st.session_state['generated_script'] = tool_output.get("test_script")
                            else:
                                st.error(f"Script generation failed: {tool_output.get('message', 'No message.')}")
                        else:
                            st.error(f"Error resuming workflow: {resume_output.get('error')}")
                    else:
                        st.warning("Could not find a paused workflow state. Please 'Run' the process again.")
        
        # --- DISPLAY GENERATED SCRIPT ---
        if st.session_state.get('generated_script'):
            with st.container(border=True):
                st.subheader("🐍 Generated Test Script")
                st.code(st.session_state['generated_script'], language="python")

        # --- ENHANCED EMAIL SECTION ---
        if st.session_state.get("rag_response"):
            with st.container(border=True):
                st.header("📧 Share Report via Email")
                email_cfg = load_email_config()
                default_recipients = ", ".join(email_cfg.get("default_recipients", []))
                
                recipients_text = st.text_input("Recipients (comma-separated)", value=default_recipients)
                subject_text = st.text_input("Subject", value=f"AgenticAI Analysis Report - {st.session_state.get('uploaded_filename', 'Analysis')}")
                
                # Show preview of what will be in the email
                st.info("📋 Email Preview: A professional report card will be sent with the complete analysis attached as a TXT file.")
                
                # with st.expander("📄 View Email Content Preview"):
                #     st.markdown("**Email will contain:**")
                #     st.markdown("- 🎨 Professional header with LTTS branding")
                #     st.markdown("- 📊 Executive summary of analysis")
                #     st.markdown("- 🔍 Original query and file information")
                #     st.markdown("- 🚦 System status indicator (Green/Yellow/Red)")
                #     st.markdown("- 📎 Complete detailed analysis as TXT attachment")
                #     st.markdown("- 💼 Professional footer with company branding")
                
                if st.checkbox("Send this professional report by email"):
                    if st.button("📤 Send Professional Report", use_container_width=True):
                        all_tools = get_all_registered_tools()
                        send_tool = all_tools.get("send_email")
                        if send_tool is None:
                            st.error("Email tool not found.")
                        else:
                            # Get the original query from session state
                            original_query = st.session_state.get("original_query", "Analysis request")
                            
                            payload = {
                                "recipients": [r.strip() for r in recipients_text.split(",") if r.strip()],
                                "subject": subject_text,
                                "body": st.session_state.get("rag_response", ""),
                                "query": original_query,
                                "filename": st.session_state.get('uploaded_filename', ''),
                            }
                            with st.spinner("📧 Sending professional report with attachment..."):
                                try:
                                    resp = send_tool.invoke(payload)
                                    if isinstance(resp, dict) and resp.get("success"):
                                        st.success(f"✅ Professional report sent successfully!")
                                        st.info(f"📊 Report details: {resp.get('message')}")
                                        
                                    else:
                                        err = resp.get("error") if isinstance(resp, dict) else str(resp)
                                        st.error(f"Failed to send email: {err}")
                                except Exception as e:
                                    st.error(f"Exception while sending email: {e}")

if __name__ == "__main__":
    app = AgenticAIApp()
    app.run_streamlit_app()