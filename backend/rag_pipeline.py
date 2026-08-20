import os
import re
import json
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
import backend.config as config

# Validate DB exists at startup, otherwise raise error
if not config.DB_DIR.exists():
    raise RuntimeError(
        f"Database directory {config.DB_DIR} does not exist. "
        "Please run offline indexing first: python -m backend.scripts.build_index"
    )

print(f"Loading persistent vector store from {config.DB_DIR}...")
embedding_model = FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
vectorstore = Chroma(
    persist_directory=str(config.DB_DIR),
    embedding_function=embedding_model,
    collection_name='autism_clinical_kb'
)

# Groq Client Initialization (strictly using env variable, NO default API key fallbacks)
if not config.GROQ_API_KEY:
    # Do not hardcode/expose, fail if missing
    raise RuntimeError("Security Alert: GROQ_API_KEY environment variable is not set!")

llm = ChatOpenAI(
    model=config.MODEL_NAME,
    base_url=config.GROQ_BASE_URL,
    api_key=config.GROQ_API_KEY,
    temperature=0.0
)

# Maximum number of recent conversation turns to include as context.
# Keeps prompts small and avoids sending unbounded history to the LLM.
MAX_CONTEXT_TURNS = 3  # last 3 turns = up to 3 user + 3 assistant messages

DAY3_SYSTEM_PROMPT = """You are a safe, citation-bound clinical evidence assistant for Autism Spectrum Disorder clinical guidelines.

RULES:
1. Answer ONLY using the retrieved evidence supplied below. Do not use outside medical knowledge, assumptions, or external facts.
2. Every claim in your "recommendation" must be directly supported by the "evidence" you cite.
3. You MUST return your response as a valid JSON object matching exactly this structure:
   {
     "status": "answered",
     "recommendation": "...",
     "supporting_evidence": [
       {"claim": "...", "citation": "[Document Name | Section: ... | Page: ... | Chunk: ...]"}
     ],
     "confidence": "high" | "medium" | "low",
     "missing_information": [],
     "safety_note": "Educational information only; not a diagnosis or medical advice."
   }
4. If the retrieved evidence is weak, missing, unrelated, or insufficient to answer the question, you MUST return status as "insufficient_evidence", recommendation as "The retrieved guidelines do not provide sufficient evidence to answer this question reliably.", confidence as "low", list what is missing in missing_information, and empty supporting_evidence.
5. Never invent a citation. Never soften a refusal into a partial guess.
"""

def clean_llm_json(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\n", "", content)
        content = re.sub(r"\n```$", "", content)
    return content.strip()

def parse_and_validate_json(raw_content: str) -> dict:
    cleaned = clean_llm_json(raw_content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError("Failed to parse response as valid JSON.")

def check_safety_pre_check(question: str) -> dict:
    q_lower = question.lower()
    
    # 1. Emergency Redirect Guardrail
    emergency_terms = [
        # Acute system emergencies
        "emergency", "911", "call 911", "call an ambulance",
        # Breathing / airway
        "stopped breathing", "not breathing", "trouble breathing",
        "difficulty breathing", "can't breathe", "cannot breathe",
        "choking", "airway",
        # Cardiac / consciousness
        "unconscious", "unresponsive", "heart attack", "stroke",
        "chest pain", "collapsed",
        # Toxicological
        "poison", "overdose", "ingested", "swallowed something",
        # Self-harm / violence
        "suicide", "suicidal", "self-harm", "kill myself",
        "harm myself", "harm",
        # Severe injury / seizure
        "seizure", "severe bleeding", "bleeding out", "severe injury",
    ]
    if any(term in q_lower for term in emergency_terms):
        return {
            "status": "redirected",
            "recommendation": "This sounds like an emergency. Please contact your local emergency medical services immediately.",
            "confidence": "low",
            "supporting_evidence": [],
            "missing_information": ["Emergency medical assistance is required."],
            "safety_note": "Emergency Redirect triggered."
        }
        
    # 2. Patient-Specific Diagnosis/Prescription Guardrail
    patient_specific_terms = [
        "do i have", "diagnose me", "my child has", "what treatment should i choose", 
        "should i take", "what dosage", "prescribe", "my dosage", "how much should i take",
        "diagnose my child", "treat my child"
    ]
    if any(term in q_lower for term in patient_specific_terms):
        return {
            "status": "refused",
            "recommendation": "I cannot provide a patient-specific diagnosis, prescription, dosage, or treatment selection. Please consult a qualified clinician.",
            "confidence": "low",
            "supporting_evidence": [],
            "missing_information": ["A qualified clinician must assess the individual case."],
            "safety_note": "Educational information only; not a diagnosis or medical advice."
        }
        
    # 3. Ambiguous General Cure Lookup
    if "cure" in q_lower and ("autism" in q_lower or "asd" in q_lower):
        return {
            "status": "clarified",
            "recommendation": "There is no known drug or medication that cures Autism Spectrum Disorder. Interventions focus on behavioral therapies, educational strategies, and speech-language support to develop skills.",
            "confidence": "high",
            "supporting_evidence": [
                {
                    "claim": "No drug cures autism; interventions focus on developmental and speech-language support.",
                    "citation": "[Autism-Spectrum-Disorder_-The-Complete-Guide-to-Understanding-Autism-PDFDrive-.pdf | Section: Chapter 5: Treatments, Therapies, and Interventions | Page: 170 | Chunk: N/A]"
                }
            ],
            "missing_information": [],
            "safety_note": "Autism has no medication-based cure; treatment is supportive."
        }
        
    return None


# ---------------------------------------------------------------------------
# Conversation Context Resolution
# ---------------------------------------------------------------------------

# Pronouns and short phrases that signal the question is referencing the
# prior conversation topic rather than being standalone.
_CONTEXT_DEPENDENT_PATTERNS = re.compile(
    r'\b(it|its|this|that|they|them|their|those|these|he|she|his|her|the same|'
    r'what about|how about|and what|tell me more|more about|can you explain|'
    r'elaborate|continue|go on|also|additionally|furthermore)\b',
    re.IGNORECASE
)

def resolve_question_with_context(
    question: str,
    conversation: Optional[List[dict]],
) -> str:
    """
    Deterministically produce a self-contained query that can be sent to
    the vector store and LLM without requiring conversation context.

    Strategy (no extra LLM call):
    - If there is no prior conversation, return the question unchanged.
    - If the question seems context-dependent (contains pronouns / short
      follow-up phrases), prepend the last user message as a topic hint.
    - Otherwise, return the question unchanged.

    IMPORTANT: This function NEVER treats conversation content as medical
    evidence.  The resolved question is only used for retrieval; all medical
    facts must still come exclusively from the Chroma vector store.
    """
    if not conversation:
        return question

    # Bound history to the most recent MAX_CONTEXT_TURNS turns.
    recent = conversation[-MAX_CONTEXT_TURNS * 2:]  # *2 because each turn has 2 messages

    # Find the last user message in the (already bounded) history.
    last_user_content = ""
    for turn in reversed(recent):
        if turn.get("role") == "user":
            last_user_content = turn.get("content", "").strip()
            break

    if not last_user_content:
        return question

    # Only rewrite if the question looks context-dependent.
    q_stripped = question.strip()
    is_short = len(q_stripped.split()) <= 8
    has_pronoun = bool(_CONTEXT_DEPENDENT_PATTERNS.search(q_stripped))

    if not (is_short or has_pronoun):
        # Long, self-contained question — no rewriting needed.
        return question

    # Construct a self-contained query by injecting the prior topic.
    # Format: "Given that we were discussing: <prior topic>. <current question>"
    resolved = f"Given that the previous question was: '{last_user_content}'. {q_stripped}"
    return resolved

def format_evidence_blocks(retrieved_results):
    evidence_blocks = []
    for rank, (doc, score) in enumerate(retrieved_results, start=1):
        m = doc.metadata
        doc_name = m.get('document_name', 'Autism-Spectrum-Disorder_-The-Complete-Guide-to-Understanding-Autism-PDFDrive-.pdf')
        section = m.get('section', 'N/A')
        page = m.get('page_number', 'N/A')
        chunk_id = m.get('chunk_id', 'N/A')
        
        citation_str = f"[{doc_name} | Section: {section} | Page: {page} | Chunk: {chunk_id}]"
        evidence_blocks.append(
            f"Evidence Block {rank}\n"
            f"Citation ID: {citation_str}\n"
            f"Similarity Score: {score:.4f}\n"
            f"Text: {doc.page_content.strip()}"
        )
    return "\n\n".join(evidence_blocks)

def extract_claims(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s.split()) > 3]

def is_claim_supported(claim: str, evidence_text: str, min_overlap=0.35) -> bool:
    claim_words = set(w.lower().strip(".,;:") for w in claim.split() if len(w) > 3)
    evidence_words = set(w.lower().strip(".,;:") for w in evidence_text.split() if len(w) > 3)
    if not claim_words:
        return True
    overlap = len(claim_words & evidence_words) / len(claim_words)
    return overlap >= min_overlap

def run_unsupported_claim_detection(answer_dict: dict, evidence_text: str) -> list:
    if answer_dict.get("status") in ["refused", "redirected", "insufficient_evidence"]:
        return []
    claims = extract_claims(answer_dict.get("recommendation", ""))
    flagged = [c for c in claims if not is_claim_supported(c, evidence_text)]
    return flagged

def validate_citations(answer_dict: dict, retrieved_results) -> dict:
    if answer_dict.get("status") in ["refused", "redirected", "insufficient_evidence"]:
        return {"citation_existence": True, "citation_binding": True, "details": []}
        
    valid_ids = []
    for doc, _ in retrieved_results:
        m = doc.metadata
        doc_name = m.get('document_name', 'Autism-Spectrum-Disorder_-The-Complete-Guide-to-Understanding-Autism-PDFDrive-.pdf')
        section = m.get('section', 'N/A')
        page = m.get('page_number', 'N/A')
        chunk_id = m.get('chunk_id', 'N/A')
        valid_ids.append(f"[{doc_name} | Section: {section} | Page: {page} | Chunk: {chunk_id}]")
        
    details = []
    existence = True
    binding = True
    
    # Verify claims have citations and that they map to retrieved IDs
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

def run_rag_pipeline(question: str, conversation: Optional[List[dict]] = None) -> dict:
    """
    Run the full RAG pipeline.

    Args:
        question:     The raw user question (used as-is for safety checks).
        conversation: Optional bounded list of prior turns, each a dict with
                      'role' ('user'|'assistant') and 'content'.  Used ONLY
                      to resolve ambiguous pronouns/references — never as
                      medical evidence.
    """
    # Step 1: Safety Pre-check — ALWAYS on the RAW question, never on the
    # resolved version.  Conversation context must NOT bypass safety.
    safety_res = check_safety_pre_check(question)
    if safety_res:
        return safety_res

    # Step 2: Resolve context — produce a self-contained retrieval query.
    # No extra LLM call is made here.
    retrieval_question = resolve_question_with_context(question, conversation)
    contextualized = retrieval_question != question  # flag for debug info

    # Step 3: Retrieve Top-K Chunks using the (possibly resolved) query.
    results = vectorstore.similarity_search_with_relevance_scores(retrieval_question, k=config.TOP_K)
    
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
    evidence_text = format_evidence_blocks(results)

    # Step 5: Grounded Generation — pass the resolved question so the LLM
    # can interpret "its treatment" correctly, but it still answers ONLY from
    # the retrieved evidence blocks, not from conversation history.
    messages = [
        SystemMessage(content=DAY3_SYSTEM_PROMPT),
        HumanMessage(content=f"Retrieved Evidence:\n{evidence_text}\n\nQuestion:\n{retrieval_question}")
    ]
    
    response_content = ""
    try:
        response = llm.invoke(messages)
        response_content = response.content
        json_output = parse_and_validate_json(response_content)
    except Exception as e:
        # Step 6: Retry ONCE with strict instructions if parsing fails
        try:
            retry_messages = [
                SystemMessage(content=DAY3_SYSTEM_PROMPT),
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

    # Step 7: Post-Generation Citation Validation
    val_report = validate_citations(json_output, results)
    json_output["citation_validation_report"] = val_report

    # Step 8: Post-Generation Unsupported Claim Overlap Detection
    flagged = run_unsupported_claim_detection(json_output, evidence_text)
    if flagged:
        json_output["status"] = "error"
        json_output["recommendation"] = "The system output failed grounded accuracy verification checks."
        json_output["supporting_evidence"] = []
        json_output["confidence"] = "low"
        json_output["missing_information"] = [f"Hallucination Warning: Claim '{c}' is unsupported by the context." for c in flagged]

    # Optional debug field — omitted when context was not used so the
    # response stays identical for single-turn requests.
    if contextualized:
        json_output["_context_resolved"] = True

    return json_output
