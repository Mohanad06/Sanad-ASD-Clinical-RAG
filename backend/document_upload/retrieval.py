import os
import re
import json
from typing import List, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

import backend.config as config
from backend.rag_pipeline import (
    resolve_question_with_context,
    clean_llm_json,
    parse_and_validate_json,
    extract_claims,
    is_claim_supported,
    run_unsupported_claim_detection
)
from .service import get_vectorstore

SYSTEM_PROMPT = """You are a safe, citation-bound clinical evidence assistant for uploaded Autism Spectrum Disorder documents.

RULES:
1. Answer ONLY using the retrieved evidence supplied below. Do not use outside medical knowledge, assumptions, or external facts.
2. Every claim in your "recommendation" must be directly supported by the "evidence" you cite.
3. You MUST return your response as a valid JSON object matching exactly this structure:
   {
     "status": "answered",
     "recommendation": "...",
     "supporting_evidence": [
       {"claim": "...", "citation": "[Document Name | Page: ... | Chunk: ...]"}
     ],
     "confidence": "high" | "medium" | "low",
     "missing_information": [],
     "safety_note": "Educational information only; not a diagnosis or medical advice."
   }
4. If the retrieved evidence is weak, missing, unrelated, or insufficient to answer the question, you MUST return status as "insufficient_evidence", recommendation as "The retrieved guidelines do not provide sufficient evidence to answer this question reliably.", confidence as "low", list what is missing in missing_information, and empty supporting_evidence.
5. Never invent a citation. Never soften a refusal into a partial guess.
"""

def format_uploaded_evidence_blocks(retrieved_results) -> str:
    evidence_blocks = []
    for rank, (doc, score) in enumerate(retrieved_results, start=1):
        m = doc.metadata
        doc_name = m.get('document_name', 'document')
        page = m.get('page_number', 'N/A')
        chunk_id = m.get('chunk_id', 'N/A')
        
        citation_str = f"[{doc_name} | Page: {page} | Chunk: {chunk_id}]"
        evidence_blocks.append(
            f"Evidence Block {rank}\n"
            f"Citation ID: {citation_str}\n"
            f"Similarity Score: {score:.4f}\n"
            f"Text: {doc.page_content.strip()}"
        )
    return "\n\n".join(evidence_blocks)

def validate_uploaded_citations(answer_dict: dict, retrieved_results) -> dict:
    if answer_dict.get("status") in ["refused", "redirected", "insufficient_evidence"]:
        return {"citation_existence": True, "citation_binding": True, "details": []}
        
    valid_ids = []
    for doc, _ in retrieved_results:
        m = doc.metadata
        doc_name = m.get('document_name', 'document')
        page = m.get('page_number', 'N/A')
        chunk_id = m.get('chunk_id', 'N/A')
        valid_ids.append(f"[{doc_name} | Page: {page} | Chunk: {chunk_id}]")
        
    details = []
    existence = True
    binding = True
    
    evidence_items = answer_dict.get("supporting_evidence", [])
    if not evidence_items and answer_dict.get("status") == "answered":
        binding = False
        details.append("Status is 'answered' but no supporting evidence list provided.")
        
    for item in evidence_items:
        cit = item.get("citation", "")
        if not cit:
            binding = False
            details.append(f"Claim '{item.get('claim')}' lacks citation binding.")
        elif cit not in valid_ids:
            existence = False
            details.append(f"Citation ID '{cit}' does not match any retrieved context blocks.")
            
    return {
        "citation_existence": existence,
        "citation_binding": binding,
        "details": details
    }

def query_uploaded_document(
    question: str,
    document_id: str,
    conversation: Optional[List[dict]] = None
) -> dict:
    """
    Retrieves information from the isolated vector store for a specific document_id,
    and runs the citation-bound RAG pipeline.
    """
    # Initialize the LLM (using same config settings)
    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        base_url=config.GROQ_BASE_URL,
        api_key=config.GROQ_API_KEY,
        temperature=0.0
    )

    # Step 1: Resolve context — produce a self-contained retrieval query.
    retrieval_question = resolve_question_with_context(question, conversation)
    contextualized = retrieval_question != question

    # Step 2: Retrieve Top-K Chunks filtered by document_id
    vectorstore = get_vectorstore()
    
    # We use similarity_search_with_relevance_scores with filter
    results = vectorstore.similarity_search_with_relevance_scores(
        retrieval_question,
        k=config.TOP_K,
        filter={"document_id": document_id}
    )
    
    # Step 3: Evidence Sufficiency Check
    top_score = results[0][1] if results else 0.0
    if not results or top_score < config.SIMILARITY_THRESHOLD:
        return {
            "status": "insufficient_evidence",
            "recommendation": "The retrieved guidelines do not provide sufficient evidence to answer this question reliably.",
            "confidence": "low",
            "supporting_evidence": [],
            "missing_information": [f"Top score {top_score:.4f} is below calibrated threshold of {config.SIMILARITY_THRESHOLD}"],
            "safety_note": "Educational information only; not a diagnosis or medical advice."
        }
        
    # Step 4: Format Evidence
    evidence_text = format_uploaded_evidence_blocks(results)

    # Step 5: Grounded Generation
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Retrieved Evidence:\n{evidence_text}\n\nQuestion:\n{retrieval_question}")
    ]
    
    response_content = ""
    try:
        response = llm.invoke(messages)
        response_content = response.content
        json_output = parse_and_validate_json(response_content)
    except Exception as e:
        # Retry once
        try:
            retry_messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Retrieved Evidence:\n{evidence_text}\n\nQuestion:\n{question}"),
                SystemMessage(content=response_content if response_content else "Error"),
                HumanMessage(content="Your response was not valid JSON or was empty. Return ONLY a valid JSON object matching the required schema. No markdown fences. No explanation.")
            ]
            retry_response = llm.invoke(retry_messages)
            json_output = parse_and_validate_json(retry_response.content)
        except Exception as retry_err:
            return {
                "status": "error",
                "recommendation": "The system failed to generate a valid structured response due to formatting or parsing issues.",
                "confidence": "low",
                "supporting_evidence": [],
                "missing_information": [f"LLM parsing error: {str(retry_err)}"],
                "safety_note": "System Generation Failure."
            }

    # Step 6: Post-Generation Citation Validation
    val_report = validate_uploaded_citations(json_output, results)
    json_output["citation_validation_report"] = val_report

    # Step 7: Post-Generation Unsupported Claim Overlap Detection
    flagged = run_unsupported_claim_detection(json_output, evidence_text)
    if flagged:
        json_output["status"] = "error"
        json_output["recommendation"] = "The system output failed grounded accuracy verification checks."
        json_output["supporting_evidence"] = []
        json_output["confidence"] = "low"
        json_output["missing_information"] = [f"Hallucination Warning: Claim '{c}' is unsupported by the context." for c in flagged]

    if contextualized:
        json_output["_context_resolved"] = True

    return json_output
