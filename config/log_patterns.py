# import re
# from typing import Dict, List, Optional

# class LogPatterns:
#     """Centralized log pattern matching and filter configuration with best match scoring"""
    
#     LOG_CONFIGS = {
#         'adb': {
#             "patterns": [
#                 r'\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+  \d+  \d+ [A-Z] Telephony: .*',
#             ],
#             "keywords": [
#                 "Telephony:", "PhoneGlobals:", "registration failed"
#             ]
#         },
#         # Easy to add new log types here:
#         # 'syslog': {
#         #     "patterns": [
#         #         r'Jan \d{2} \d{2}:\d{2}:\d{2}',
#         #         r'kernel:',
#         #         r'systemd\[\d+\]:',
#         #         r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}',
#         #     ],
#         #     "keywords": [
#         #         "kernel:", "systemd:", "NetworkManager:", "sudo:",
#         #         "cron:", "ssh:", "apache2:", "nginx:"
#         #     ]
#         # }
#         'pcap':{
#             "patterns":[],
#             "keywords":["ngap"]
#         }
#     }
    
#     # # PCAP keywords only (no patterns needed for pcap)
#     # PCAP_Filters = ["ngap"]
    
#     @staticmethod
#     def identify_log_type(file_path: str) -> Optional[str]:
#         """Identify log type by checking ALL patterns and returning best match"""
#         try:
#             with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#                 content = f.read(2000)  # Read first 2000 chars for detection
            
#             lines = content.split('\n')[:10]
#             sample_text = '\n'.join(lines)
            
#             scores = {}
            
#             # Check all log types and calculate match scores
#             for log_type, config in LogPatterns.LOG_CONFIGS.items():
#                 matches = 0
#                 patterns = config.get("patterns", [])
#                 if not patterns:
#                     # Skip scoring for patternless types (e.g., pcap)
#                     continue

#                 matches = sum(1 for pattern in patterns if re.search(pattern, sample_text, re.MULTILINE))
#                 total_patterns = len(patterns)
                
#                 # Calculate match percentage
#                 if total_patterns > 0:
#                     scores[log_type] = matches / total_patterns
            
#             # Return log type with highest score (minimum 50% match required)
#             if scores:
#                 best_match = max(scores.items(), key=lambda x: x[1])
#                 if best_match[1] >= 0.5:  # At least 50% patterns must match
#                     return best_match[0]
            
#             return None  # No log type identified with sufficient confidence
            
#         except Exception:
#             return None
    
#     @staticmethod
#     def get_keywords_for_type(log_type: str) -> List[str]:
#         """Get filter keywords for specific log type"""
#         return LogPatterns.LOG_CONFIGS.get(log_type, {}).get("keywords", [])


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