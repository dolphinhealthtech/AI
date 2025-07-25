import os
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader
)
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".csv"]

def load_documents_from_folder(folder_path: str) -> List[Document]:
    all_docs = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()

            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                elif ext == ".docx":
                    loader = Docx2txtLoader(file_path)
                elif ext == ".txt":
                    loader = TextLoader(file_path, encoding="utf-8")
                elif ext == ".csv":
                    loader = CSVLoader(file_path)
                else:
                    print(f"[SKIP] Format tidak didukung: {file}")
                    continue

                documents = loader.load()
                all_docs.extend(documents)

            except Exception as e:
                print(f"[ERROR] Gagal memuat {file}: {e}")

    print(f"[INFO] Total dokumen berhasil dimuat: {len(all_docs)}")
    return all_docs
