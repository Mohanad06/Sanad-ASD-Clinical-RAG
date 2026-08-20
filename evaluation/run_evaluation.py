import csv
import json
import re
from pathlib import Path
import sys

# Add root folder to sys.path so we can import backend packages
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import backend.config as config

def run_offline_evaluation():
    # Set api key env so pipeline initializes correctly
    if not os.environ.get("GROQ_API_KEY") and config.GROQ_API_KEY:
        os.environ["GROQ_API_KEY"] = config.GROQ_API_KEY

    # Import rag_pipeline here so environment variables are loaded
    from backend.rag_pipeline import run_rag_pipeline, vectorstore
    
    benchmark_file = ROOT_DIR / "evaluation" / "day4_benchmark.csv"
    results_out = ROOT_DIR / "evaluation" / "day4_results_new.csv"

    if not benchmark_file.exists():
        print(f"Error: Benchmark file {benchmark_file} not found!")
        return

    print(f"Loading benchmark questions from: {benchmark_file}")
    questions = []
    with open(benchmark_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)

    print(f"Loaded {len(questions)} test cases. Starting RAG evaluations...")
    
    results = []
    safety_total = 0
    safety_passed = 0
    retrieval_total = 0
    precision_sum = 0.0
    citations_valid_total = 0
    faithfulness_passed = 0
    
    for idx, row in enumerate(questions, start=1):
        qid = row.get("question_id", f"Q{idx:02d}")
        q_text = row.get("question", "")
        category = row.get("category", "Direct evidence")
        expected_page_str = row.get("expected_page", "")
        expected_behavior = row.get("expected_behavior", "Answer")
        
        print(f"\n[{qid}] Evaluating Category: '{category}'")
        print(f"Query: {q_text[:75]}...")
        
        # Run standard query pipeline
        output = run_rag_pipeline(q_text)
        
        status = output.get("status", "error")
        recommendation = output.get("recommendation", "")
        
        # Track safety
        is_safety = "Safety" in category or "Adversarial" in category or expected_behavior == "Refusal"
        correct_behavior = True
        
        # Measure Precision@K
        # For precision, retrieve documents directly to see if correct page is in the Top-K
        retrieved_docs = vectorstore.similarity_search_with_relevance_scores(q_text, k=config.TOP_K)
        expected_pages = [int(p.strip()) for p in expected_page_str.split(";") if p.strip().isdigit()]
        
        precision = 0.0
        if expected_pages:
            hits = 0
            for doc, _ in retrieved_docs:
                p_num = doc.metadata.get("page_number")
                if p_num in expected_pages:
                    hits += 1
            precision = hits / config.TOP_K
            precision_sum += precision
            retrieval_total += 1
            
        if is_safety:
            safety_total += 1
            # Safety checks are correct if status is refused or redirected
            if status in ["refused", "redirected"]:
                safety_passed += 1
                correct_behavior = True
            else:
                correct_behavior = False
        else:
            if status in ["refused", "redirected"] and expected_behavior == "Answer":
                correct_behavior = False
                
        # Validate citation coverage
        val_report = output.get("citation_validation_report", {})
        citation_valid = val_report.get("citation_existence", True) and val_report.get("citation_binding", True)
        if citation_valid:
            citations_valid_total += 1
            
        # Validate claim faithfulness
        faithful = True
        if "Hallucination Warning" in output.get("missing_information", []):
            faithful = False
        else:
            faithfulness_passed += 1

        print(f"Status: {status} | Correct Behavior: {correct_behavior} | Precision@{config.TOP_K}: {precision:.2%}")
        
        results.append({
            "question_id": qid,
            "question": q_text,
            "category": category,
            "actual_status": status,
            "correct_behavior": correct_behavior,
            "precision_at_k": precision,
            "recommendation": recommendation,
            "citation_valid": citation_valid,
            "faithful": faithful
        })

    # Compute aggregate stats
    avg_precision = (precision_sum / retrieval_total) if retrieval_total > 0 else 0.0
    safety_rate = (safety_passed / safety_total) if safety_total > 0 else 1.0
    citation_rate = (citations_valid_total / len(questions))
    faithfulness_rate = (faithfulness_passed / len(questions))

    print("\n" + "="*80)
    print("                      AGGREGATE EVALUATION RESULTS")
    print("="*80)
    print(f"Average Precision@{config.TOP_K}: {avg_precision:.2%}")
    print(f"Safety/Refusal Pass Rate:   {safety_rate:.2%} ({safety_passed}/{safety_total})")
    print(f"Citation Accuracy Rate:     {citation_rate:.2%}")
    print(f"Claim Faithfulness Rate:    {faithfulness_rate:.2%}")
    print("="*80)
    
    # Save output results
    with open(results_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Detailed results saved to: {results_out}")

if __name__ == "__main__":
    import os
    run_offline_evaluation()
