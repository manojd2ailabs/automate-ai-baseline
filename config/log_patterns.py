"""
Description:
This module defines log pattern configurations and provides utilities to identify 
log types (e.g., ADB, PCAP) based on pattern matching and keyword analysis. It supports 
dynamic addition of new log types, pattern inspection, and keyword retrieval for filtering 
and classification during parsing or RAG workflows.

Created By: Tanvi Dongaonkar
Date: 
Modified By:
Reason:
Example:
Used by Parser Tool to automatically detect log type 
from uploaded files before selecting the appropriate parsing or analysis tool.
"""

import re
from typing import Dict, List, Optional

class LogPatterns:
    """Simple log pattern matching with basic scoring"""
    
    LOG_CONFIGS = {
        'adb': {
            "patterns": [
                r'\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+\s+\d+\s+\d+\s+[A-Z]\s+\w+\s*:\s*.*',
            ],
            "keywords": ["Telephony:", "PhoneGlobals:"]
        },
        'pcap': {
            "patterns": [],  # Binary files - no text patterns
            "keywords": ["ngap"]
        }
    }
    
    @staticmethod
    def identify_log_type(file_path: str) -> Optional[str]:
        """
        Simple log type identification:
        - Read file sample
        - Check each log type's patterns
        - Return first type with at least 50% pattern matches
        """
        try:
            # Read sample content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(4000)  # Read more chars to ensure we get actual content
            
            # Split into lines and filter out empty lines
            all_lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            # Take a representative sample from the available lines
            # If we have many lines, take from different parts of the file
            if len(all_lines) > 30:
                # Take first 15 + last 15 lines for better coverage
                sample_lines = all_lines[:15] + all_lines[-15:]
            else:
                # Use all available lines
                sample_lines = all_lines
            
            sample_text = '\n'.join(sample_lines)
            
            print(f"Checking log type for: {file_path}")
            print(f"Total lines: {len(all_lines)}, Sample lines: {len(sample_lines)}")
            
            # Test each log type
            for log_type, config in LogPatterns.LOG_CONFIGS.items():
                patterns = config.get("patterns", [])
                
                if not patterns:  # Skip types without patterns (like pcap)
                    continue
                
                # Count matches
                matches = 0
                for pattern in patterns:
                    if re.search(pattern, sample_text, re.MULTILINE):
                        matches += 1
                
                # Calculate match percentage
                match_rate = matches / len(patterns)
                print(f"{log_type}: {matches}/{len(patterns)} patterns matched ({match_rate:.1%})")
                
                # Return if meets threshold
                if match_rate >= 0.5:
                    print(f"Identified as: {log_type}")
                    return log_type
            
            print("No log type identified")
            return None
            
        except Exception as e:
            print(f"Error identifying log type: {e}")
            return None
    
    @staticmethod
    def get_keywords_for_type(log_type: str) -> List[str]:
        """Get filter keywords for a log type"""
        return LogPatterns.LOG_CONFIGS.get(log_type, {}).get("keywords", [])
    
    @staticmethod
    def add_log_type(log_type: str, patterns: List[str], keywords: List[str]):
        """Add a new log type configuration"""
        LogPatterns.LOG_CONFIGS[log_type] = {
            "patterns": patterns,
            "keywords": keywords
        }
        print(f"Added log type: {log_type}")
    
    @staticmethod
    def list_log_types() -> List[str]:
        """Get all configured log types"""
        return list(LogPatterns.LOG_CONFIGS.keys())