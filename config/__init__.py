import yaml
from pathlib import Path
from typing import Dict, Any

def load_email_config() -> dict:
    """
    Load email SMTP configuration from config/email_config.yml.
    Returns an empty dict if file not found.
    """
    cfg_file = Path(__file__).parent / "email_config.yml"
    if not cfg_file.exists():
        return {}
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[WARN] Failed to load email config: {e}")
        return {}

def load_prompts() -> Dict[str, Dict[str, str]]:
    """Load prompts from YAML configuration file"""
    prompts_file = Path(__file__).parent / "prompts.yml"
    
    if not prompts_file.exists():
        # Default prompts if file doesn't exist
        return {
            "router": {
                "system_prompt": "You are a router agent that determines the appropriate processing pipeline."
            },
            "log_analysis": {
                "system_prompt": "You are a log analysis agent specialized in processing log files."
            },
            "test_case_generation": {
                "system_prompt": "You are a test case generation agent."
            },
            "research_supervisor " : {
                "system_prompt": "You are a supervisor, Choose the best sub agent."
            }
        }
    
    # ✅ Fix: Use UTF-8 encoding to avoid UnicodeDecodeError
    with open(prompts_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
