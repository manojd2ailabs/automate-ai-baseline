# config/memory_config.py

"""
Description:
Define the keys that will be used for metadata in the vector store.
This allows for easy changes to the metadata schema in the future without
having to change the tool's implementation logic.

Created By: Pediredla Sai Ram, Parise Hari Sai
Date: 
Modified By:
Reason:
Example:
Used by the Memory Tool and RAG components to attach and retrieve metadata 
(e.g., file_path, agent_id, query_text) during vector store creation and retrieval.
"""

METADATA_KEYS = {
    "FILE": "file_path",
    "AGENT": "agent_id",
    "QUERY": "query_text"
}