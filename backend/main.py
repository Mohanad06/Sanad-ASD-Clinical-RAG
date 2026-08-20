import os
from typing import List, Optional
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
import backend.config as config
from backend.rag_pipeline import run_rag_pipeline, check_safety_pre_check
from backend.document_upload.routes import router as document_upload_router
from backend.document_upload.retrieval import query_uploaded_document

app = FastAPI(
    title="Clinical ASD RAG Decision Support API",
    description="A safe, citation-aware clinical decision support system for Autism Spectrum Disorder (ASD).",
    version="1.0.0"
)

# Register the document upload router
app.include_router(document_upload_router)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for the local hackathon demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static frontend directory
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

class ConversationTurn(BaseModel):
    """A single turn in the conversation history."""
    role: str  # "user" or "assistant"
    content: str

class QueryRequest(BaseModel):
    question: str
    # Optional conversation history for contextual follow-up questions.
    # The backend uses this ONLY to understand the current question — not as medical evidence.
    conversation: Optional[List[ConversationTurn]] = None
    # Optional document ID to target queries to a specific uploaded document instead of the official KB.
    document_id: Optional[str] = None

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "database_initialized": config.DB_DIR.exists(),
        "model": config.MODEL_NAME
    }

@app.post("/api/query")
def execute_query(payload: QueryRequest):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Sanitize and bound the conversation to the last MAX_CONTEXT_TURNS turns.
    # Empty list and None are both treated as "no conversation context".
    raw_conversation = payload.conversation or []
    # Only accept valid roles; drop malformed entries silently.
    clean_conversation = [
        {"role": t.role, "content": t.content}
        for t in raw_conversation
        if t.role in ("user", "assistant") and t.content and t.content.strip()
    ]

    # If document_id is provided, route query to the uploaded document index
    if payload.document_id:
        # Step 1: Run raw-question safety pre-check BEFORE retrieving or generating
        safety_res = check_safety_pre_check(payload.question)
        if safety_res:
            return safety_res
            
        try:
            response = query_uploaded_document(
                payload.question,
                document_id=payload.document_id,
                conversation=clean_conversation
            )
            return response
        except Exception as e:
            return {
                "status": "error",
                "recommendation": "Internal server error occurred while processing uploaded document query.",
                "confidence": "low",
                "supporting_evidence": [],
                "missing_information": [str(e)],
                "safety_note": "System failure."
            }

    try:
        response = run_rag_pipeline(payload.question, conversation=clean_conversation)
        return response
    except Exception as e:
        return {
            "status": "error",
            "recommendation": "Internal server error occurred while processing query.",
            "confidence": "low",
            "supporting_evidence": [],
            "missing_information": [str(e)],
            "safety_note": "System failure."
        }

@app.get("/api/evaluation/summary")
def get_evaluation_summary():
    # Return the static summary parsed from evaluation metrics to avoid LLM calls
    return {
        "metrics": {
            "precision_at_3": 0.8667,
            "precision_at_5": 0.8800,
            "safety_pass_rate": 1.00,
            "citation_validity": 1.00,
            "claim_faithfulness": 1.00,
            "out_of_scope_refusal_rate": 1.00
        },
        "description": "Evaluation metrics computed against the Day 4 benchmark (29 questions).",
        "benchmark_source": "evaluation/day4_benchmark.csv"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=config.HOST, port=config.PORT, reload=True)
