from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentMetadata(BaseModel):
    document_id: str = Field(..., description="Stable ID of the document (e.g. UPLOAD-001)")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File extension (.pdf or .docx)")
    file_size: int = Field(..., description="File size in bytes")
    upload_time: str = Field(..., description="Timestamp when uploaded")
    status: str = Field("indexed", description="Ingestion/indexing status")

class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]

class UploadResponse(BaseModel):
    status: str
    document_id: str
    filename: str
    message: str

class DeleteResponse(BaseModel):
    status: str
    message: str
