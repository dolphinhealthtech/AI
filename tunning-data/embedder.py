# embedings/embedder.py

import os
import sys
from pathlib import Path

# Tambahkan root path agar semua import lintas-folder berjalan
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.vectorstores import FAISS
from mediq_ai.loaders.document_loader import load_documents
from mediq_ai.embeddings.model_embeddings import OllamaEmbeddings
from mediq_ai.utils.config import VECTORSTORE_PATH, EMBEDDING_MODEL
from mediq_ai.utils.logger import get_logger

logger = get_logger(__name__, "embedder.log")

def main():
    try:
        logger.info("[EMBEDDER] Memuat dokumen dari folder data/...")
        docs = load_documents("data/")

        if not docs:
            raise ValueError("Folder data/ tidak berisi dokumen valid.")
        logger.info(f"[EMBEDDER] Total dokumen sebelum filtering: {len(docs)}")

        # Filter dokumen kosong
        docs = [doc for doc in docs if doc.page_content.strip()]
        if not docs:
            raise ValueError("Tidak ada konten dokumen setelah filtering.")
        logger.info(f"[EMBEDDER] Total dokumen setelah filtering: {len(docs)}")

        # Inisialisasi embedding model
        logger.info(f"[EMBEDDER] Menggunakan model embedding: {EMBEDDING_MODEL}")
        embedding_model = OllamaEmbeddings(EMBEDDING_MODEL)

        # Buat FAISS vectorstore
        logger.info("[EMBEDDER] Membuat FAISS vectorstore dari dokumen...")
        vectordb = FAISS.from_documents(docs, embedding_model)

        # Simpan vectorstore
        output_path = Path(VECTORSTORE_PATH) / "faiss_index"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        vectordb.save_local(str(output_path))
        logger.info(f"[EMBEDDER] Vectorstore berhasil disimpan di: {output_path}")

    except Exception as e:
        logger.error(f"[EMBEDDER] Terjadi kesalahan: {e}")
        raise

if __name__ == "__main__":
    main()
