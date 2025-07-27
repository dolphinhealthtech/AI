# rag/rag_chain.py
import os
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from mediq_ai.embeddings.model_embeddings import OllamaEmbeddings
from utils.config import MODEL, EMBEDDING_MODEL, VECTORSTORE_PATH
from template.template import RAG_PROMPT_TEMPLATE
from utils.logger import get_logger

logger = get_logger(__name__, "rag.log")

class RAG:
    def __init__(self):
        self.embedding_model = EMBEDDING_MODEL
        self.llm_model = MODEL
        self.vectorstore_path = os.path.join(VECTORSTORE_PATH, "faiss_index")
        self.embedding = None
        self.qa_chain = None
        self._load_chain()

    def _load_chain(self):
        try:
            logger.info(f"[RAG] Memuat embedding: {self.embedding_model} dan LLM: {self.llm_model}")
            self.embedding = OllamaEmbeddings(model=self.embedding_model)

            if not os.path.exists(self.vectorstore_path):
                raise FileNotFoundError(f"Vectorstore tidak ditemukan di {self.vectorstore_path}")

            vectordb = FAISS.load_local(
                folder_path=self.vectorstore_path,
                embeddings=self.embedding,
                allow_dangerous_deserialization=True,
            )

            retriever = vectordb.as_retriever()
            prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
            llm = ChatOllama(model=self.llm_model)

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt},
            )

            logger.info("[RAG] QA Chain berhasil dimuat dan siap digunakan.")

        except Exception as e:
            logger.error(f"[RAG] Gagal memuat QA Chain: {e}")
            self.qa_chain = None

    def is_ready(self) -> bool:
        return self.qa_chain is not None

    def ask(self, query: str) -> Dict[str, Any]:
        if not self.qa_chain:
            logger.warning("[RAG] QA Chain belum siap.")
            return {
                "answer": "",
                "sources": [],
                "error": "[RAG] QA Chain belum siap."
            }

        try:
            logger.debug(f"[RAG] Pertanyaan diterima: {query}")
            result = self.qa_chain.invoke({"query": query})

            answer = result.get("result", "").strip()
            documents = result.get("source_documents", [])
            sources = [doc.page_content for doc in documents] if documents else []

            return {
                "answer": answer,
                "sources": sources
            }

        except Exception as e:
            logger.error(f"[RAG] Gagal memproses pertanyaan: {e}")
            return {
                "answer": "",
                "sources": [],
                "error": str(e)
            }
