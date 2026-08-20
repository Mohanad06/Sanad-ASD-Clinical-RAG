import os
import shutil
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app
import backend.config as config
from backend.document_upload.service import INDEX_DIR, REGISTRY_FILE, UPLOADS_DIR, get_vectorstore, read_registry

class TestDocumentUploadRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        
        # Paths to keep backups of if they exist
        cls.uploads_backup = Path("backend/document_upload/uploads_backup")
        cls.index_backup = Path("backend/document_upload/index_backup")
        cls.registry_backup = Path("backend/document_upload/registry_backup.json")
        
        # Backup existing uploaded documents to ensure test isolation
        if UPLOADS_DIR.exists():
            shutil.copytree(UPLOADS_DIR, cls.uploads_backup, dirs_exist_ok=True)
            shutil.rmtree(UPLOADS_DIR, ignore_errors=True)
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        
        if INDEX_DIR.exists():
            shutil.copytree(INDEX_DIR, cls.index_backup, dirs_exist_ok=True)
            shutil.rmtree(INDEX_DIR, ignore_errors=True)
            
        if REGISTRY_FILE.exists():
            try:
                shutil.copy(REGISTRY_FILE, cls.registry_backup)
                REGISTRY_FILE.unlink()
            except Exception:
                pass

        # Create dummy PDF data
        cls.pdf_content = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [ 3 0 R ] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n"
            b"4 0 obj\n<< /Length 72 >>\nstream\n"
            b"BT\n/F1 12 Tf\n72 712 Td\n(Autism research shows that early behavioral intervention is effective.) Tj\nET\n"
            b"endstream\nendobj\n"
            b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
            b"xref\n0 6\n"
            b"0000000000 65535 f\n"
            b"0000000009 00000 n\n"
            b"0000000058 00000 n\n"
            b"0000000115 00000 n\n"
            b"0000000216 00000 n\n"
            b"0000000338 00000 n\n"
            b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
            b"startxref\n415\n%%EOF\n"
        )
        
        # Create dummy DOCX data (using zipfile structure containing word/document.xml)
        cls.docx_path = Path("test_temp_doc.docx")
        import zipfile
        with zipfile.ZipFile(cls.docx_path, 'w') as docx:
            xml_content = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
                '<w:body>\n'
                '<w:p>\n'
                '<w:r>\n'
                '<w:t>Sensory processing challenges are common in autistic individuals.</w:t>\n'
                '</w:r>\n'
                '</w:p>\n'
                '</w:body>\n'
                '</w:document>'
            )
            docx.writestr('word/document.xml', xml_content)

    @classmethod
    def tearDownClass(cls):
        # Cleanup test files
        if cls.docx_path.exists():
            cls.docx_path.unlink()
            
        # Clean up test directories with ignore_errors=True to prevent Windows file lock errors
        if UPLOADS_DIR.exists():
            shutil.rmtree(UPLOADS_DIR, ignore_errors=True)
        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR, ignore_errors=True)
        if REGISTRY_FILE.exists():
            try:
                REGISTRY_FILE.unlink()
            except Exception:
                pass
            
        # Restore backups
        if cls.uploads_backup.exists():
            shutil.copytree(cls.uploads_backup, UPLOADS_DIR, dirs_exist_ok=True)
            shutil.rmtree(cls.uploads_backup, ignore_errors=True)
        if cls.index_backup.exists():
            shutil.copytree(cls.index_backup, INDEX_DIR, dirs_exist_ok=True)
            shutil.rmtree(cls.index_backup, ignore_errors=True)
        if cls.registry_backup.exists():
            try:
                shutil.copy(cls.registry_backup, REGISTRY_FILE)
                cls.registry_backup.unlink()
            except Exception:
                pass

    def test_01_upload_pdf(self):
        response = self.client.post(
            "/api/documents/upload",
            files={"file": ("test_autism_research.pdf", self.pdf_content, "application/pdf")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["document_id"], "UPLOAD-001")
        self.assertEqual(data["filename"], "test_autism_research.pdf")
        
        # Verify file is saved
        self.assertTrue((UPLOADS_DIR / "UPLOAD-001_test_autism_research.pdf").exists())
        
        # Verify registered metadata
        registry = read_registry()
        self.assertEqual(len(registry), 1)
        self.assertEqual(registry[0]["document_id"], "UPLOAD-001")

    def test_02_upload_docx(self):
        with open(self.docx_path, "rb") as f:
            content = f.read()
            
        response = self.client.post(
            "/api/documents/upload",
            files={"file": ("test_sensory_challenges.docx", content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["document_id"], "UPLOAD-002")
        self.assertEqual(data["filename"], "test_sensory_challenges.docx")
        
        # Verify registered metadata
        registry = read_registry()
        self.assertEqual(len(registry), 2)
        self.assertEqual(registry[1]["document_id"], "UPLOAD-002")

    def test_03_list_documents(self):
        response = self.client.get("/api/documents")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("documents", data)
        self.assertEqual(len(data["documents"]), 2)
        
        filenames = [doc["filename"] for doc in data["documents"]]
        self.assertIn("test_autism_research.pdf", filenames)
        self.assertIn("test_sensory_challenges.docx", filenames)

    def test_04_query_uploaded_pdf(self):
        # Retrieve information from the uploaded PDF
        response = self.client.post(
            "/api/query",
            json={
                "question": "What does early behavioral intervention accomplish according to the document?",
                "document_id": "UPLOAD-001"
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "answered")
        self.assertIn("early behavioral intervention", data["recommendation"].lower())
        
        # Check citations
        self.assertTrue(len(data["supporting_evidence"]) > 0)
        citation = data["supporting_evidence"][0]["citation"]
        self.assertIn("test_autism_research.pdf", citation)
        self.assertIn("UPLOAD-001", citation)

    def test_05_query_uploaded_docx(self):
        # Retrieve information from the uploaded DOCX
        response = self.client.post(
            "/api/query",
            json={
                "question": "What is common in autistic individuals according to the document?",
                "document_id": "UPLOAD-002"
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "answered")
        self.assertIn("sensory processing", data["recommendation"].lower())
        
        # Check citations
        self.assertTrue(len(data["supporting_evidence"]) > 0)
        citation = data["supporting_evidence"][0]["citation"]
        self.assertIn("test_sensory_challenges.docx", citation)
        self.assertIn("UPLOAD-002", citation)

    def test_06_query_safety_guardrails(self):
        # Verify safety pre-check works for uploaded-document route (Emergency redirect)
        response = self.client.post(
            "/api/query",
            json={
                "question": "He stopped breathing, what should I do?",
                "document_id": "UPLOAD-001"
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "redirected")
        self.assertIn("emergency", data["recommendation"].lower())

        # Patient-specific request refusal
        response = self.client.post(
            "/api/query",
            json={
                "question": "My child has autism, what dosage should I prescribe?",
                "document_id": "UPLOAD-001"
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "refused")
        self.assertIn("dosage", data["recommendation"].lower())

    def test_07_invalid_file_handling(self):
        # Upload unsupported file type
        response = self.client.post(
            "/api/documents/upload",
            files={"file": ("test_invalid.txt", b"plain text content", "text/plain")}
        )
        self.assertEqual(response.status_code, 400)
        
        # Upload empty file
        response = self.client.post(
            "/api/documents/upload",
            files={"file": ("test_empty.pdf", b"", "application/pdf")}
        )
        self.assertEqual(response.status_code, 400)

    def test_08_delete_document(self):
        # Delete UPLOAD-001
        response = self.client.delete("/api/documents/UPLOAD-001")
        self.assertEqual(response.status_code, 200)
        
        # Verify file is deleted
        self.assertFalse((UPLOADS_DIR / "UPLOAD-001_test_autism_research.pdf").exists())
        
        # Verify registry entry is removed
        registry = read_registry()
        self.assertEqual(len(registry), 1)
        self.assertEqual(registry[0]["document_id"], "UPLOAD-002")
        
        # Verify vector store chunks are removed
        vectorstore = get_vectorstore()
        results = vectorstore.get(where={"document_id": "UPLOAD-001"})
        self.assertEqual(len(results.get("ids", [])), 0)

if __name__ == "__main__":
    unittest.main()
