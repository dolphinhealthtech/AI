# loaders/document_loader.py
import os
from typing import List
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader

from utils.logger import get_logger

logger = get_logger(__name__, "document_loader.log")

# Ekstensi yang didukung oleh Unstructured
SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".docx", ".xlsx", ".csv", ".eml", ".pptx"]

def load_documents(directory: str) -> List[Document]:
    """
    Memuat semua dokumen dari direktori menggunakan UnstructuredFileLoader.
    Mendukung: .txt, .pdf, .docx, .xlsx, .csv, .pptx, .eml
    """
    if not os.path.exists(directory):
        logger.error(f"Folder '{directory}' tidak ditemukan.")
        return []

    documents = []
    total_files = 0

    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            path = os.path.join(root, file)

            if ext not in SUPPORTED_EXTENSIONS:
                logger.warning(f"[LOADER] Format tidak didukung: {file}")
                continue

            total_files += 1
            try:
                loader = UnstructuredLoader(path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = os.path.relpath(path, directory)

                documents.extend(docs)
                logger.debug(f"[LOADER] Dimuat: {file} ({len(docs)} dokumen)")
            except Exception as e:
                logger.error(f"[LOADER] Gagal memuat {file}: {e}")

    logger.info(f"[LOADER] Total: {len(documents)} dokumen dari {total_files} file yang diproses.")
    return documents
