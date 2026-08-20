import os
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
import pypdf

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

def validate_file(file_path: str, filename: str) -> None:
    """
    Perform basic validations on the file:
    - Exists
    - Extension must be .pdf or .docx
    - Size must not exceed MAX_FILE_SIZE
    """
    if not os.path.exists(file_path):
        raise ValueError("File does not exist.")
        
    size = os.path.getsize(file_path)
    if size > MAX_FILE_SIZE:
        raise ValueError(f"File size exceeds limit of 10MB (actual: {size / (1024*1024):.2f}MB).")
        
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ('.pdf', '.docx'):
        raise ValueError(f"Unsupported file format '{ext}'. Only .pdf and .docx are supported.")

def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text and page numbers from a PDF file using pypdf.
    Returns: List of dicts, e.g., [{'page_number': 1, 'text': '...'}]
    """
    try:
        reader = pypdf.PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({
                "page_number": i + 1,
                "text": text.strip()
            })
        
        # Verify if text extraction was successful
        total_text = "".join(p["text"] for p in pages).strip()
        if not total_text:
            raise ValueError("No extractable text found in PDF. The file might be scanned or empty.")
            
        return pages
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        raise ValueError(f"Failed to parse PDF: Corrupted or invalid file. Details: {str(e)}")

def parse_docx(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a DOCX file using zipfile and xml.etree.ElementTree.
    Returns: List with a single dictionary representing page 1 since DOCX has no physical pages.
    """
    try:
        if not zipfile.is_zipfile(file_path):
            raise ValueError("Invalid DOCX file (not a valid zip archive).")
            
        with zipfile.ZipFile(file_path) as docx:
            if 'word/document.xml' not in docx.namelist():
                raise ValueError("Invalid DOCX format: word/document.xml missing.")
                
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            # Namespaces
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            paragraphs = []
            for p in root.findall('.//w:p', ns):
                texts = [t.text for t in p.findall('.//w:t', ns) if t.text]
                if texts:
                    paragraphs.append("".join(texts))
                    
            text = "\n\n".join(paragraphs).strip()
            if not text:
                raise ValueError("No extractable text found in DOCX. The document might be empty.")
                
            return [{
                "page_number": 1,
                "text": text
            }]
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        raise ValueError(f"Failed to parse DOCX: Corrupted or invalid file. Details: {str(e)}")

def extract_document_pages(file_path: str, filename: str) -> List[Dict[str, Any]]:
    """
    Validates and extracts pages from either a PDF or DOCX file.
    """
    validate_file(file_path, filename)
    ext = os.path.splitext(filename.lower())[1]
    if ext == '.pdf':
        return parse_pdf(file_path)
    elif ext == '.docx':
        return parse_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'")
