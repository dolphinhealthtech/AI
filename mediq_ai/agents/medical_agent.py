from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from utils.config import MODEL
from utils.logger import get_logger

logger = get_logger(__name__, "medical_agent.log")

class MedicalAgent:
    def __init__(self, model: str = MODEL):
        try:
            self.llm = ChatOllama(model=model)
            logger.info(f"[MEDICAL_AGENT] Model {model} siap digunakan.")
        except Exception as e:
            logger.error(f"[MEDICAL_AGENT] Gagal inisialisasi model: {e}")
            raise

        self.prompt = PromptTemplate.from_template("""
Anda adalah asisten medis berbasis AI. Berdasarkan pertanyaan berikut dan dokumen pendukung jika ada, berikan jawaban akurat, padat, dan sesuai standar medis Indonesia:

Pertanyaan:
{question}

Dokumen Pendukung (jika ada):
{context}

Jawaban:
""")

    def generate_answer(self, question: str, context: str = "") -> str:
        try:
            chain = self.prompt | self.llm
            result = chain.invoke({"question": question, "context": context})
            return result.content.strip() if hasattr(result, "content") else str(result).strip()
        except Exception as e:
            logger.error(f"[MEDICAL_AGENT] Gagal menghasilkan jawaban: {e}")
            return "Maaf, terjadi kesalahan saat memproses jawaban."
