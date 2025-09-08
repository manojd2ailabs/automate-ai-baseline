
# config_files/config_paths.py
import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).parent.parent.absolute()


# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, "output_files")

# Vector store directories
VECTOR_STORE_BASE_DIR = os.path.join(OUTPUT_DIR, "vector_stores")

# Create directories if they don't exist
for directory in [ OUTPUT_DIR,
                  VECTOR_STORE_BASE_DIR]:
    os.makedirs(directory, exist_ok=True)