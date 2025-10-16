# rag_tool.py

"""
Description:
RAG tool for AgenticAI workflows that performs semantic search over parsed PDF/TXT files, 
creates vector stores, retrieves relevant context for queries, and optionally integrates 
long-term agent memories for LLM-enhanced responses.

Key Features:
- Extracts and chunks text from PDFs/TXT for vector storage.
- Performs similarity search to retrieve relevant context.
- Enriches queries with past agent memories.
- Supports file-based query storage.
- Returns structured results including retrieved documents and LLM output.

Created By:  Pediredla Sai Ram, Parise Hari Sai
Date:
Modified By:
Reason:
Example:
Used by ResearchGraph and log/test case agents to semantically query preprocessed files and 
generate AI-augmented insights.

"""


from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import warnings
from config.config_paths import VECTOR_STORE_BASE_DIR
warnings.filterwarnings("ignore", message=".*resume_download*", category=FutureWarning)
import fitz  
from pathlib import Path
from langchain.schema import Document
from io import BytesIO
from tools.registry import register_tool
from typing import Dict, Any, Optional
from langchain_ollama import ChatOllama
import tempfile
from langchain_community.llms.ollama import Ollama


def set_rag_query(query: str):
    """Set the RAG query from the UI using file-based storage."""
    try:
        query_file = os.path.join(tempfile.gettempdir(), "rag_query.txt")
        with open(query_file, 'w', encoding='utf-8') as f:
            f.write(query)
        print(f"RAG query set in file: {query}")
    except Exception as e:
        print(f"Error setting RAG query: {e}")


def get_rag_query():
    """Get the RAG query from file storage."""
    try:
        query_file = os.path.join(tempfile.gettempdir(), "rag_query.txt")
        if os.path.exists(query_file):
            with open(query_file, 'r', encoding='utf-8') as f:
                query = f.read().strip()
            print(f"RAG query retrieved from file: {query}")
            return query
        print("No RAG query file found")
        return ""
    except Exception as e:
        print(f"Error getting RAG query: {e}")
        return ""



class RAG:
    @staticmethod
    def get_file_type(file_path: str) -> str:
        """Determine file type from extension."""
        return Path(file_path).suffix.lower()

    @staticmethod
    def extract_text_from_pdf(pdf_source):
        """
        Extract text from PDF.
        Accepts either file path (str/Path) or PDF bytes.
        """
        text = ""
        try:
            if isinstance(pdf_source, (str, Path)) and os.path.exists(pdf_source):
                
                # From file path
                doc = fitz.open(pdf_source)
            elif isinstance(pdf_source, (bytes, bytearray)):
                # From bytes
                file_stream = BytesIO(pdf_source)
                doc = fitz.open(stream=file_stream, filetype='pdf')
            else:
                raise ValueError("Invalid PDF source: must be file path or bytes.")

            for page in doc:
                text += page.get_text()
            doc.close()

        except Exception as e:
            
            raise RuntimeError(f"Failed to extract text from PDF: {e}")

        return text

    @staticmethod
    def chunk_text(text, chunk_size=2000, overlap=16):
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks



@tool
def create_vector_store(parsed_path: str) -> Dict[str, Any]:
    """
    Create a RAG vector store from the given file (PDF or TXT).
    """
    model_name = 'sentence-transformers/all-mpnet-base-v2'
    embedder = HuggingFaceEmbeddings(model_name=model_name)

    os.makedirs(VECTOR_STORE_BASE_DIR, exist_ok=True)

    base_name = os.path.basename(parsed_path).replace('.', '_')
    vector_store_name = f"vs_{base_name}"
    vector_store_path = os.path.join(VECTOR_STORE_BASE_DIR, vector_store_name)

    # Avoid recreating if already exists
    if os.path.exists(vector_store_path):
        return {
            "success": True,
            "vector_store_name": vector_store_name,
            "message": f"Vector store '{vector_store_name}' already exists.",
            "vector_store_path": vector_store_path,
            "created": False
        }

    try:
        file_type = RAG.get_file_type(parsed_path)

        if file_type == '.txt':
            with open(parsed_path, 'r', encoding='utf-8') as f:
                text_content = f.read()

        elif file_type == '.pdf':
            
            text_content = RAG.extract_text_from_pdf(parsed_path)
            

        else:
            return {
                "success": False,
                "message": f"Unsupported file format: {file_type}",
                "vector_store_name": vector_store_name,
                "vector_store_path": vector_store_path,
                "created": False
            }

        if not text_content.strip():
            return {
                "success": False,
                "message": f"No text extracted from {parsed_path}",
                "vector_store_name": vector_store_name,
                "vector_store_path": vector_store_path,
                "created": False
            }

        chunks = RAG.chunk_text(text_content)
        documents = [Document(page_content=chunk) for chunk in chunks]

        vector_store = Chroma(
            persist_directory=vector_store_path,
            embedding_function=embedder
        )
        vector_store.add_documents(documents)
        vector_store.persist()

        return {
            "success": True,
            "vector_store_name": vector_store_name,
            "message": f"Vector store '{vector_store_name}' created successfully.",
            "vector_store_path": vector_store_path,
            "created": True
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error creating vector store: {e}",
            "vector_store_name": vector_store_name,
            "vector_store_path": vector_store_path,
            "created": False
        }


# UPDATED: Added 'retrieved_memories' to accept memories from the state
@tool
def query_rag_store(brief: str, vector_store_path: str = "", prompts: Dict[str, str] = None, retrieved_memories: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Retrieve similar context for a given query from vector store, enhanced with long-term memories."""
    print("Retrieving context...")
    print(f'[INFO: retrieved_memories]: {retrieved_memories}')
    query = brief.strip() if brief else ""
    prompt_text = (prompts or {}).get("custom_prompt") or (prompts or {}).get("Default")
    print(f'[INFO: prompt_text]: {prompt_text}')
    
    if not prompt_text:
        return {"success": False, "message": "Tool Error: query_rag_store was called without a valid prompt."}
    
    if not query:
        query = get_rag_query()

    if not query:
        return {"success": False, "message": "No query provided. Please enter a RAG query."}

    if not vector_store_path:
        # Logic to find a default vector store...
        try:
            vector_stores = [d for d in os.listdir(VECTOR_STORE_BASE_DIR) if os.path.isdir(os.path.join(VECTOR_STORE_BASE_DIR, d))]
            if not vector_stores:
                return {"success": False, "message": "No vector store found. Please create one first."}
            vector_store_path = os.path.join(VECTOR_STORE_BASE_DIR, vector_stores[0])
        except Exception as e:
            return {"success": False, "message": f"Error finding vector store: {e}"}

    try:
        model_name = 'sentence-transformers/all-mpnet-base-v2'
        embedder = HuggingFaceEmbeddings(model_name=model_name)

        vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embedder)

        results = vector_store.similarity_search(query, k=3)
        context = " ".join(doc.page_content for doc in results)
        print(context,"contextttt")
        

        # UPDATED: Format memories and add them to the prompt context
        memory_context = ""
        if retrieved_memories and isinstance(retrieved_memories, dict):
            formatted_memories = []
            for agent, findings in retrieved_memories.items():
                if findings and findings.get('memories'):
                    mem_str = "\n".join(f"- {mem}" for mem in findings['memories'])
                    formatted_memories.append(f"Memories from {agent}:\n{mem_str}")
            if formatted_memories:
                memory_context = "### Relevant Past Memories\n" + "\n\n".join(formatted_memories) + "\n\n"
        print(f'[INFO: memory_context]: {memory_context}')
        # Combine memories with the RAG context for the final prompt
        final_context = f"### Context from Provided File\n{context} {memory_context}"
        print(f'[INFO:final_context ]: {final_context}')
        
        prompt = prompt_text.format(context=final_context, query=query)
        
        llm = Ollama(model="gemma3:12b")
        response = llm.invoke(prompt)

        return {
            "success": True,
            "query": query,
            "results": [{"content": doc.page_content, "metadata": doc.metadata} for doc in results],
            "response": response,
            "message": f"Found {len(results)} similar documents for query: '{query}'",
            "used_prompt": prompt_text
        }

    except Exception as e:
        return {"success": False, "message": f"Error querying vector store: {e}"}


register_tool("create_vector_store", create_vector_store)
register_tool("query_rag_store", query_rag_store)