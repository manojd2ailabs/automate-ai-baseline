# Query metadata

"""
Description:
This module provides tools for saving and retrieving agent-specific memories in the 
AgenticAI workflow using a vector store. It standardizes memory storage with metadata, 
enabling context-aware retrieval for different agents. Key features include:

- Saving agent findings or responses with metadata such as file path, agent ID, and query.
- Retrieving memories using semantic similarity for most agents.
- Advanced retrieval for 'test_case_agent', using cosine similarity on stored queries to ensure 
  highly relevant memory retrieval.
- Integration with Chroma vector store and HuggingFace sentence embeddings for consistent 
  semantic searches.
- Automatic database path management per agent using a centralized mapping.

Created By:  Pediredla Sai Ram, Parise Hari Sai, Tanvi Dongaonkar.
Date: 
Modified By:
Reason:
Example:
Used by ResearchGraph and other agent workflows to persist findings and retrieve past results 
(e.g., test cases, log analysis outputs, or generated scripts) for context-aware agent reasoning.
"""


import os
import json
import numpy as np 
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from config.config_paths import MEMORIES_DIR
from config.memory_config import METADATA_KEYS
from tools.registry import register_tool
from typing import Dict, Any

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
EMBEDDER = HuggingFaceEmbeddings(model_name=MODEL_NAME)

AGENT_DB_MAP = {
    "log_analysis_agent": "log_db",
    "test_case_agent": "TC_db",
    "test_script_agent": "TS_db",
}

def _get_memory_db_path(agent_id: str) -> str:
    """Constructs the path for an agent's memory database."""
    db_folder = AGENT_DB_MAP.get(agent_id)
    if not db_folder:
        safe_agent_id = agent_id.replace(" ", "_").replace("-", "_")
        db_folder = f"{safe_agent_id}_db"
    return os.path.join(MEMORIES_DIR, db_folder)

def _cosine_similarity(vec1, vec2):
    """Helper function to calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    
    if vec1_norm == 0 or vec2_norm == 0:
        return 0.0
        
    return np.dot(vec1, vec2) / (vec1_norm * vec2_norm)

@tool
def load_memories(agent_id: str, query: str, file_path: str) -> Dict[str, Any]:
    """
    Loads memories for an agent.
    
    - For 'test_case_agent', it retrieves the most relevant memory by performing a cosine
      similarity search on the queries stored in the metadata, loading the memory only
      if the similarity score is above 90%.
      
    - For all other agents, it performs a standard similarity search on document content.
    """
    db_path = _get_memory_db_path(agent_id)
    if not os.path.exists(db_path):
        return {"success": True, "message": f"No memories found for agent '{agent_id}'.", "memories": []}
    
    try:
        vector_store = Chroma(persist_directory=db_path, embedding_function=EMBEDDER)
        memories = []
        
        if agent_id == 'test_case_agent':
            print("--- Executing advanced memory retrieval for test_case_agent ---")
            
            metadata_filter = {
                "$and": [
                    {METADATA_KEYS["AGENT"]: agent_id},
                    {METADATA_KEYS["FILE"]: file_path}
                ]
            }
            candidate_docs = vector_store.get(where=metadata_filter, include=["metadatas", "documents"])

            if not candidate_docs or not candidate_docs.get('ids'):
                return {"success": True, "memories": []}

            user_query_embedding = EMBEDDER.embed_query(query)

            best_match_content = None
            best_similarity_score = -1.0
            best_match_query = None 

            for i, metadata in enumerate(candidate_docs['metadatas']):
                stored_query = metadata.get(METADATA_KEYS["QUERY"])
                if not stored_query:
                    continue
                
                stored_query_embedding = EMBEDDER.embed_query(stored_query)
                similarity = _cosine_similarity(user_query_embedding, stored_query_embedding)
                
                print(f"Comparing with stored query (Similarity: {similarity:.2f}): '{stored_query}'")

                if similarity > best_similarity_score:
                    best_similarity_score = similarity
                    best_match_content = candidate_docs['documents'][i]
                    best_match_query = stored_query

            if best_similarity_score > 0.9:
                print(f"Found best match with similarity {best_similarity_score:.2f}.")
                # FIXED: Added the 'f' prefix to correctly format the string.
                print(f"Loading memory associated with stored query: '{best_match_query}'")
                memories = [best_match_content]
            else:
                print(f"Best match similarity ({best_similarity_score:.2f}) is below 0.9 threshold. No memory loaded.")

        else:
            print(f"--- Executing standard memory retrieval for {agent_id} ---")
            search_filter = {
                "$and": [
                    {METADATA_KEYS["AGENT"]: agent_id},
                    {METADATA_KEYS["FILE"]: file_path}
                ]
            }
            results = vector_store.similarity_search(query, k=1, filter=search_filter)
            memories = [doc.page_content for doc in results]

        print(f"Loaded {len(memories)} memories.")
        return {"success": True, "message": f"Loaded {len(memories)} memories.", "memories": memories}

    except Exception as e:
        return {"success": False, "error": f"Failed to load memories: {e}", "memories": []}


@tool
def save_memories(agent_id: str, content: str, file_path: str, query: str = None) -> Dict[str, Any]:
    """
    Saves a memory, using configurable keys for the metadata.
    For the test_case_agent, it also saves the query text in the metadata.
    """
    db_path = _get_memory_db_path(agent_id)
    try:
        content_to_save = json.dumps(json.loads(content), indent=2)
    except (json.JSONDecodeError, TypeError):
        content_to_save = str(content)

    metadata = {
        METADATA_KEYS["FILE"]: file_path,
        METADATA_KEYS["AGENT"]: agent_id
    }

    if agent_id == 'test_case_agent' and query:
        metadata[METADATA_KEYS["QUERY"]] = query

    document = Document(
        page_content=content_to_save,
        metadata=metadata
    )

    os.makedirs(db_path, exist_ok=True)
    vector_store = Chroma(persist_directory=db_path, embedding_function=EMBEDDER)
    vector_store.add_documents([document])
    vector_store.persist()
    print(f"Saved memory for agent '{agent_id}' with metadata: {metadata}")

    return {"success": True, "message": f"Successfully saved memory for agent '{agent_id}'."}


register_tool("load_memories", load_memories)
register_tool("save_memories", save_memories)

 
