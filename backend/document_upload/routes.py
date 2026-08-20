from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os

from .schemas import UploadResponse, DocumentListResponse, DeleteResponse, DocumentMetadata
from .service import save_and_index_document, read_registry, delete_document

router = APIRouter(prefix="/api/documents", tags=["Document Upload"])

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or DOCX file, parse its text, chunk it, embed it,
    index it in the isolated vector store, and save it in the registry.
    """
    # Verify extension
    filename = file.filename
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ('.pdf', '.docx'):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Only .pdf and .docx are supported."
        )

    try:
        content = await file.read()
        meta = save_and_index_document(content, filename)
        return UploadResponse(
            status="success",
            document_id=meta.document_id,
            filename=meta.filename,
            message="Document uploaded and indexed successfully."
        )
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error processing document: {str(e)}"
        )

@router.get("", response_model=DocumentListResponse)
def list_documents():
    """
    List all uploaded and registered documents.
    """
    registry = read_registry()
    docs = [DocumentMetadata(**item) for item in registry]
    return DocumentListResponse(documents=docs)

@router.delete("/{document_id}", response_model=DeleteResponse)
def remove_document(document_id: str):
    """
    Delete the uploaded file, metadata registry, and associated Chroma vector chunks.
    """
    try:
        delete_document(document_id)
        return DeleteResponse(
            status="success",
            message=f"Document '{document_id}' deleted successfully."
        )
    except ValueError as val_err:
        raise HTTPException(status_code=404, detail=str(val_err))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )
