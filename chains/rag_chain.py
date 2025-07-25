import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from loaders.document_loader import load_documents_from_folder

# Load env
load_dotenv()

VECTOR_DIR = os.getenv("VECTORSTORE_PATH", "vectorstore")
DATA_DIR = os.getenv("DATA_PATH", "data")
LLM_MODEL = os.getenv("OLLAMA_MODEL", "gemma-3b-it")
EMBED_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

embedding_model = OllamaEmbeddings(model=EMBED_MODEL)

def create_vectorstore():
    print("[INFO] Memuat dokumen dari folder:", DATA_DIR)
    docs = load_documents_from_folder(DATA_DIR)
    print(f"[INFO] Jumlah dokumen ditemukan: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Jumlah chunk hasil pemotongan: {len(chunks)}")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="medical-data",
        persist_directory=VECTOR_DIR
    )

    print("[INFO] Vectorstore berhasil dibuat dan disimpan.")
    return vectordb

# Load atau buat vectorstore
if not os.path.exists(os.path.join(VECTOR_DIR, "chroma.sqlite3")):
    vectordb = create_vectorstore()
else:
    print("[INFO] Memuat vectorstore dari direktori:", VECTOR_DIR)
    vectordb = Chroma(persist_directory=VECTOR_DIR, embedding_function=embedding_model)

# Retriever dan compressor
retriever = vectordb.as_retriever(search_kwargs={"k": 4})
llm = OllamaLLM(model=LLM_MODEL)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

def get_context_from_rag(question: str) -> str:
    try:
        print("\n[QUESTION] ➜", question)

        # Step 1: Similarity search
        results = vectordb.similarity_search_with_score(question, k=4)

        if not results:
            print("[WARN] Tidak ada dokumen mirip ditemukan.")
            return "Informasi tidak ditemukan di dokumen, namun saya akan mencoba menjawab berdasarkan pengetahuan saya."

        print(f"[INFO] Dokumen hasil similarity: {len(results)}")

        for i, (doc, score) in enumerate(results):
            print(f"\n[SIMILAR {i+1}] Score: {score:.4f}")
            print(doc.page_content[:300], "...\n")

        # Step 2: Kompresi
        compressed_docs = compression_retriever.invoke(question)

        if not compressed_docs:
            print("[WARN] Tidak ada hasil setelah compression.")
            return "Informasi tidak ditemukan di dokumen, namun saya akan mencoba menjawab berdasarkan pengetahuan saya."

        print(f"[INFO] Dokumen hasil compression: {len(compressed_docs)}")

        for i, doc in enumerate(compressed_docs):
            print(f"\n[COMPRESSED {i+1}]\n{doc.page_content[:300]}...\n")

        return "\n".join([doc.page_content for doc in compressed_docs])

    except Exception as e:
        print(f"[ERROR] {e}")
        return f"Terjadi kesalahan saat mengambil konteks: {e}\nSaya tetap akan mencoba menjawab berdasarkan pengetahuan saya."
