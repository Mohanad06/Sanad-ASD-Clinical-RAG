import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma

# Base paths relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_DIR = BASE_DIR / "backend" / "db" / "chroma_db"
DATA_DIR = BASE_DIR / "Day 02" / "data"

if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR / "data"

def get_section_for_page(page_num):
    if page_num < 17:
        return "Front Matter & Preface"
    elif page_num < 41:
        return "Chapter 1: Myths and History of ASD"
    elif page_num < 76:
        return "Chapter 2: Diagnostic Criteria and Symptoms of ASD"
    elif page_num < 116:
        return "Chapter 3: Causes and Behavioral Dynamics of ASD"
    elif page_num < 146:
        return "Chapter 4: Post-Diagnosis Guidance for Families"
    elif page_num < 191:
        return "Chapter 5: Treatments, Therapies, and Interventions"
    elif page_num < 221:
        return "Chapter 6: Family Life"
    elif page_num < 256:
        return "Chapter 7: Educational Strategies and IEP"
    elif page_num < 286:
        return "Chapter 8: Community Life and Social Integration"
    else:
        return "Chapter 9 & Appendices: Adult Life, Employment, and Resources"

def build_offline_index():
    print(f"Searching for PDFs in: {DATA_DIR}")
    pdf_files = sorted(list(DATA_DIR.glob("*.pdf")))
    if not pdf_files:
        print("Error: No PDF files found to index!")
        return False
    
    all_pages = []
    for idx, pdf_path in enumerate(pdf_files, start=1):
        doc_id = f"DOC-{idx:03d}"
        print(f"Loading {pdf_path.name} as {doc_id}...")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        for page in pages:
            page_num = page.metadata.get('page', 0) + 1
            page.metadata.update({
                'document_id': doc_id,
                'document_name': pdf_path.name,
                'title': 'Autism Spectrum Disorder: The Complete Guide',
                'source': str(pdf_path),
                'page_number': page_num,
                'section': get_section_for_page(page_num)
            })
            all_pages.append(page)
            
    print(f"Loaded {len(all_pages)} pages. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=850,
        chunk_overlap=150,
        separators=['\n\n', '\n', '. ', ' ', '']
    )
    chunks = splitter.split_documents(all_pages)
    
    # Generate stable chunk IDs
    doc_chunk_counters = {}
    for chunk in chunks:
        doc_id = chunk.metadata['document_id']
        doc_chunk_counters[doc_id] = doc_chunk_counters.get(doc_id, 0) + 1
        chunk.metadata['chunk_id'] = f"{doc_id}-CH-{doc_chunk_counters[doc_id]:04d}"
        
    print(f"Created {len(chunks)} chunks. Initializing embedding model...")
    embedding_model = FastEmbedEmbeddings(model_name='BAAI/bge-small-en-v1.5')
    
    print(f"Saving vector database index to: {DB_DIR}")
    # Enforce database schema config
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name='autism_clinical_kb',
        persist_directory=str(DB_DIR),
        collection_metadata={'hnsw:space': 'cosine'}
    )
    print("Offline indexing completed successfully!")
    return True

if __name__ == "__main__":
    build_offline_index()
