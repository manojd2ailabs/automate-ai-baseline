# config_files/config_paths.py
"""
Description:
This module defines and initializes key directory paths used across the Agentic AI system. 
It sets up the base project structure for outputs, vector stores, and memory storage, ensuring 
required directories exist at runtime.

Created By: Tanvi Dongaonkar
Date: 
Modified By:
Reason:
Example:
Used by various modules (e.g., vector store creation, memory manager, or log analysis tools) 
to reference consistent project paths such as OUTPUT_DIR, VECTOR_STORE_BASE_DIR, and MEMORIES_DIR.
"""

import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).parent.parent.absolute()


# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, "output_files")

# Vector store directories
VECTOR_STORE_BASE_DIR = os.path.join(OUTPUT_DIR, "vector_stores")

# Memory store directories (NEW)
MEMORIES_DIR = os.path.join(BASE_DIR, "memories")

# Create directories if they don't exist
for directory in [ OUTPUT_DIR,
                  VECTOR_STORE_BASE_DIR,
                  MEMORIES_DIR]:
    os.makedirs(directory, exist_ok=True)