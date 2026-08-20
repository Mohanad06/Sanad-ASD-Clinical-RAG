import os
import json
import shutil
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma

import backend.config as config
from .parsers import extract_document_pages
from .schemas import DocumentMetadata

# Save data outside workspace to prevent uvicorn reload loops (especially in OneDrive environments)
DATA_ROOT = Path.home() / ".asd_rag_uploads"
UPLOADS_DIR = DATA_ROOT / "uploads"
INDEX_DIR = DATA_ROOT / "index"
REGISTRY_FILE = DATA_ROOT / "registry.json"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Shared embedding model
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    return _embedding_model

def get_vectorstore() -> Chroma:
    """Gets the isolated Chroma database for uploaded documents."""
    return Chroma(
        persist_directory=str(INDEX_DIR),
        embedding_function=get_embedding_model(),
        collection_name='uploaded_documents_kb',
        collection_metadata={'hnsw:space': 'cosine'}
    )

def read_registry() -> List[Dict[str, Any]]:
    if not REGISTRY_FILE.exists():
        return []
    try:
        with open(REGISTRY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def write_registry(data: List[Dict[str, Any]]) -> None:
    with open(REGISTRY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_next_document_id(registry: List[Dict[str, Any]]) -> str:
    if not registry:
        return "UPLOAD-001"
    ids = []
    for item in registry:
        try:
            val = int(item['document_id'].split('-')[1])
            ids.append(val)
        except Exception:
            pass
    max_id = max(ids) if ids else 0
    return f"UPLOAD-{max_id + 1:03d}"

def save_and_index_document(file_content: bytes, filename: str) -> DocumentMetadata:
    """
    Saves the file to uploads/, extracts text, chunks it, generates embeddings,
    stores chunks in the isolated Chroma index, and registers it.
    """
    registry = read_registry()
    doc_id = generate_next_document_id(registry)
    
    # Save file
    file_path = UPLOADS_DIR / f"{doc_id}_{filename}"
    with open(file_path, "wb") as f:
        f.write(file_content)
        
    try:
        # Extract text pages
        pages = extract_document_pages(str(file_path), filename)
        
        # Convert to LangChain Documents
        documents = []
        for p in pages:
            doc = Document(
                page_content=p['text'],
                metadata={
                    'document_id': doc_id,
                    'document_name': filename,
                    'page_number': p['page_number'],
                    'source': filename
                }
            )
            documents.append(doc)
            
        # Chunk documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=850,
            chunk_overlap=150,
            separators=['\n\n', '\n', '. ', ' ', '']
        )
        chunks = splitter.split_documents(documents)
        
        # Assign stable chunk IDs
        for idx, chunk in enumerate(chunks, start=1):
            chunk.metadata['chunk_id'] = f"{doc_id}-CH-{idx:04d}"
            
        # Embed and save chunks in separate Chroma vector store
        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)
        
        # Save metadata to registry
        meta = DocumentMetadata(
            document_id=doc_id,
            filename=filename,
            file_type=os.path.splitext(filename.lower())[1],
            file_size=len(file_content),
            upload_time=datetime.datetime.now().isoformat(),
            status="indexed"
        )
        
        registry.append(meta.model_dump())
        write_registry(registry)
        return meta
        
    except Exception as e:
        # Cleanup uploaded file on error
        if file_path.exists():
            os.remove(file_path)
        raise e

def delete_document(document_id: str) -> None:
    """
    Deletes the document from the registry, deletes its uploaded file,
    and removes all associated chunks from the separate vector store.
    """
    registry = read_registry()
    target_doc = None
    for item in registry:
        if item['document_id'] == document_id:
            target_doc = item
            break
            
    if not target_doc:
        raise ValueError(f"Document with ID '{document_id}' not found.")
        
    # Remove from Chroma vector store
    vectorstore = get_vectorstore()
    # Get all chunks matching this document_id
    # We query all documents with document_id filter, then delete by their IDs
    results = vectorstore.get(where={"document_id": document_id})
    ids_to_delete = results.get("ids", [])
    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)
        
    # Remove file from uploads/
    filename = target_doc['filename']
    file_path = UPLOADS_DIR / f"{document_id}_{filename}"
    if file_path.exists():
        os.remove(file_path)
        
    # Update registry
    updated_registry = [item for item in registry if item['document_id'] != document_id]
    write_registry(updated_registry)
