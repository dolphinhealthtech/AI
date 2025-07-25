import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.llms import Ollama
from loaders.document_loader import load_documents_from_folder

# Load env vars
load_dotenv()

# Konfigurasi direktori dan model
VECTOR_DIR = os.getenv("VECTORSTORE_PATH", "vectorstore")
DATA_DIR = os.getenv("DATA_PATH", "data")
LLM_MODEL = os.getenv("OLLAMA_MODEL", "gemma-3b-it")
EMBED_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# Embedding dari Ollama
embedding_model = OllamaEmbeddings(model=EMBED_MODEL)

# Buat vectorstore dari dokumen jika belum ada
def create_vectorstore():
    docs = load_documents_from_folder(DATA_DIR)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectordb = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory=VECTOR_DIR)
    vectordb.persist()
    return vectordb

# Load atau buat vectorstore
if not os.path.exists(os.path.join(VECTOR_DIR, "chroma.sqlite3")):
    vectordb = create_vectorstore()
else:
    vectordb = Chroma(persist_directory=VECTOR_DIR, embedding_function=embedding_model)

# Buat retriever dengan kompresi menggunakan LLM dari Ollama
retriever = vectordb.as_retriever(search_kwargs={"k": 4})
llm = Ollama(model=LLM_MODEL)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# Fungsi utama untuk mengambil konteks dari dokumen
def get_context_from_rag(question: str) -> str:
    try:
        docs = compression_retriever.get_relevant_documents(question)
        if not docs:
            return "Informasi tidak ditemukan di dokumen, namun saya akan mencoba menjawab berdasarkan pengetahuan saya."
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Terjadi kesalahan saat mengambil konteks: {e}\nSaya tetap akan mencoba menjawab berdasarkan pengetahuan saya."
