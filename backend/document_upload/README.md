# Experimental Document Upload RAG Pipeline

This module allows clinicians/administrators to upload PDF/DOCX documents, parse and index them dynamically in a separate vector store, and perform citation-aware queries grounded on their contents.

## Folder Structure

- `uploads/`: Raw PDF and DOCX files.
- `index/`: Chroma DB persist directory for uploaded documents chunks.
- `registry.json`: JSON metadata registry tracking uploaded documents.
- `parsers.py`: PDF and DOCX parsers and file validator.
- `service.py`: Management of uploads, registry, indexing, and deletion.
- `retrieval.py`: Retrieval and grounded generation logic for uploaded documents.
- `routes.py`: FastAPI endpoints for uploading, listing, and deleting.
- `schemas.py`: Pydantic models for request/response validation.
