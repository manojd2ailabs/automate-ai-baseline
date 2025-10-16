
"""
Description:
This module provides a generic file parsing tool for the AgenticAI workflow, handling PCAP and 
TXT logs. It standardizes file preprocessing for downstream analysis by agents, using LogPatterns filters
for keyword-based filtering and preserving important context.It Returns structured metadata including parsed file path, original path, and file type

Created By: Tanvi Dongaonkar ,Pediredla Sai Ram
Date: 
Modified By:
Reason:
Example:
Used by ResearchGraph, log analysis agents, and other workflows to preprocess incoming PCAP and 
log files before semantic analysis, memory embedding, or test case generation.
"""

from langchain_core.tools import tool
from typing import Dict, Any, List
import os
import subprocess
from pathlib import Path
from config.config_paths import OUTPUT_DIR 
from tools.registry import register_tool
from shutil import copy2
from config.log_patterns import LogPatterns


class ParserTool:
    """Generic parser tool with support for multiple file formats"""
    
    @staticmethod
    def get_file_type(file_path: str) -> str:
        """Determine file type from extension"""
        return Path(file_path).suffix.lower()
    
    @staticmethod
    def pcap_to_txt(file_path: str) -> str:
        """Convert PCAP file to text format using tshark"""
        try:
            # Create output directory for parsed files
            parsed_files_dir = os.path.join(OUTPUT_DIR, "parsed_files")
            os.makedirs(parsed_files_dir, exist_ok=True)

            # Define the output file path
            output_path = os.path.join(parsed_files_dir, os.path.basename(file_path).replace('.pcap', '_parsed.txt'))

            # Use tshark to convert pcap to text
            cmd = [
                'tshark',
                '-r', file_path,
                '-T', 'text',
                '-V'  # Verbose output
            ]

            # Get pcap keywords from LOG_CONFIGS
            keywords = LogPatterns.get_keywords_for_type('pcap')

            # Apply filter if keywords exist
            if keywords:
                # For example, merge multiple filters with "or"
                display_filter = " or ".join(keywords)
                cmd.extend(['-Y', display_filter])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                with open(output_path, 'w') as f:
                    f.write(result.stdout)
                return output_path
            else:
                raise Exception(f"tshark error: {result.stderr}")

        except Exception as e:
            raise Exception(f"Error converting PCAP to text: {str(e)}")

    @staticmethod
    def filter_adb_logs(file_path: str, log_type: str) -> str:
        """Filter logs based on log type and keywords from LogPatterns"""
        try:
            # Get keywords for the specific log type
            keywords = LogPatterns.get_keywords_for_type(log_type)
            
            # Create output directory for parsed files
            parsed_files_dir = os.path.join(OUTPUT_DIR, "parsed_files")
            os.makedirs(parsed_files_dir, exist_ok=True)
            
            # Define output file path
            base_name = os.path.basename(file_path)
            output_path = os.path.join(parsed_files_dir, f"{Path(base_name).stem}_filtered.txt")
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                log_lines = f.readlines()
            
            total_lines = len(log_lines)
            included_indices = set()
            results = []
            
            # Include the first 5 lines
            for idx in range(min(5, total_lines)):
                if idx not in included_indices:
                    results.append(log_lines[idx])
                    included_indices.add(idx)
            
            # Keyword-based extraction (10 lines after each match)
            for i, line in enumerate(log_lines):
                if any(keyword in line for keyword in keywords):
                    start = i
                    end = min(i + 6, total_lines)  # Current line + 10 after
                    for idx in range(start, end):
                        if idx not in included_indices:
                            results.append(log_lines[idx])
                            included_indices.add(idx)
            
            # Include the last 5 lines
            for idx in range(max(0, total_lines - 5), total_lines):
                if idx not in included_indices:
                    results.append(log_lines[idx])
                    included_indices.add(idx)
            
            # Write filtered results to output file
            with open(output_path, 'w', encoding='utf-8', errors='ignore') as output_file:
                output_file.writelines(results)
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Error filtering {log_type} logs: {str(e)}")


@tool
def parse_file(file_path: str) -> Dict[str, Any]:
    """
    Generic file parser that selects appropriate parsing method based on file type and use case.
    
    Args:
        file_path: Path to the file to parse
    
    Returns:
        Dict containing parsed file path and metadata
    """
    try:
        parser = ParserTool()
        file_type = parser.get_file_type(file_path)

        # Create output directory for parsed files
        parsed_files_dir = os.path.join(OUTPUT_DIR, "parsed_files")
        os.makedirs(parsed_files_dir, exist_ok=True)
     
        # Select parser based on file type
        if file_type == '.pcap':
            parsed_path = parser.pcap_to_txt(file_path)
        elif file_type == '.txt':
            # Identify log type and filter if known type, otherwise copy as-is
            log_type = LogPatterns.identify_log_type(file_path)
            print(f"Log type is :{log_type}")
            
            if log_type == 'adb':
                parsed_path = parser.filter_adb_logs(file_path, log_type)
            else:
                # Copy .txt files as-is for unknown types (existing behavior)
                dest_path = os.path.join(parsed_files_dir, os.path.basename(file_path))
                copy2(file_path, dest_path)
                parsed_path = dest_path
        else:
            # Copy other files as-is
            dest_path = os.path.join(parsed_files_dir, os.path.basename(file_path))
            copy2(file_path, dest_path)
            parsed_path = dest_path
        
        return {
            "success": True,
            "parsed_path": parsed_path,
            "original_path": file_path,
            "file_type": file_type,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "original_path": file_path
        }


register_tool("parse_file", parse_file)