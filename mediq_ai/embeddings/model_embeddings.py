# embeddings/model_embedder.py

import requests
from typing import List
from langchain_core.embeddings import Embeddings
from utils.config import EMBEDDING_MODEL, BASE_URL
from utils.logger import get_logger
from typing import Optional

logger = get_logger(__name__, "embedding.log")

class OllamaEmbeddings(Embeddings):
    _instances = {}  # Caching berdasarkan model

    def __new__(cls, model: Optional[str] = None):
        model_name = model or EMBEDDING_MODEL
        if model_name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[model_name] = instance
        return cls._instances[model_name]

    def __init__(self, model: Optional[str] = None):
        model = model or EMBEDDING_MODEL
        if not hasattr(self, "_initialized"):  # Inisialisasi hanya sekali
            self.model = model
            self.base_url = BASE_URL
            logger.info(f"[EMBEDDING] Inisialisasi dengan model: {self.model}")
            self._initialized = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        logger.info(f"[EMBEDDING] Mengirim {len(texts)} dokumen ke API untuk embedding...")

        for i, text in enumerate(texts):
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    headers={"Content-Type": "application/json"},
                    json={"model": self.model, "prompt": text, "stream": False},
                    timeout=20
                )
                response.raise_for_status()
                result = response.json()

                vector = result.get("embedding")
                if not isinstance(vector, list) or not all(isinstance(v, float) for v in vector):
                    raise ValueError(f"Embedding tidak valid pada dokumen ke-{i+1}: {vector}")

                embeddings.append(vector)
                logger.debug(f"[EMBEDDING] Dokumen ke-{i+1} sukses. Dimensi: {len(vector)}")

            except Exception as e:
                logger.error(f"[EMBEDDING] Gagal memproses dokumen ke-{i+1}: {e}")
                raise

        dims = {len(vec) for vec in embeddings}
        if len(dims) != 1:
            logger.error(f"[EMBEDDING] Dimensi tidak konsisten: {dims}")
            raise ValueError("Dimensi embedding tidak konsisten!")

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        logger.debug("[EMBEDDING] Memproses embedding untuk satu query...")
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                headers={"Content-Type": "application/json"},
                json={"model": self.model, "prompt": text, "stream": False},
                timeout=20
            )
            response.raise_for_status()
            result = response.json()

            vector = result.get("embedding")
            if not isinstance(vector, list) or not all(isinstance(v, float) for v in vector):
                raise ValueError(f"Embedding query tidak valid: {vector}")

            logger.debug(f"[EMBEDDING] Query sukses. Dimensi: {len(vector)}")
            return vector

        except Exception as e:
            logger.error(f"[EMBEDDING] Gagal membuat embedding untuk query: {e}")
            raise
