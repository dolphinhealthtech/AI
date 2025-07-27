import os
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL", "gemma-3b-it")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "vectorstore")
DATA_PATH = os.getenv("DATA_PATH", "data")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# Validasi
if not MODEL:
    raise ValueError("MODEL belum diset di .env")
if not EMBEDDING_MODEL:
    raise ValueError("EMBEDDING_MODEL belum diset di .env")
