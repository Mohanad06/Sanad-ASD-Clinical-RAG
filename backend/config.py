import os
from pathlib import Path
from dotenv import load_dotenv

# Load env variables from .env if present
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Server settings
PORT = int(os.environ.get("PORT", 8000))
HOST = os.environ.get("HOST", "0.0.0.0")

# API Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# RAG configuration settings (Non-negotiable calibrated rules)
SIMILARITY_THRESHOLD = 0.70
TOP_K = 5
MODEL_NAME = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# DB paths
DB_DIR = BASE_DIR / "backend" / "db" / "chroma_db"
BENCHMARK_CSV = BASE_DIR / "evaluation" / "day4_benchmark.csv"
RESULTS_CSV = BASE_DIR / "evaluation" / "day4_results.csv"
