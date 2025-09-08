# # tools/__init__.py
# from .parser import parse_file
# from .rag import create_rag_store, query_rag_store

# __all__ = ['parse_file', 'create_rag_store', 'query_rag_store']


# tools/__init__.py
from tools.registry import discover_and_register_tools, get_all_registered_tools, register_tool

# Automatically discover and register tools when the package is imported
discover_and_register_tools()

# Re-export key functions for convenience if needed elsewhere
__all__ = ["get_all_registered_tools", "register_tool"]
