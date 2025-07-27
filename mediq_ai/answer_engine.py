from typing import Optional
from pydantic import BaseModel
from mediq_ai.embeddings.model_embeddings import OllamaEmbeddings
from guardrails.compliance import HallucinationChecker
from rag.rag_chain import RAG
from tools.external_search_tool import search_web
from utils.config import MODEL
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from utils.logger import get_logger

logger = get_logger(__name__, "engine.log")

# ============================
# SCHEMA RESPON
# ============================

class AskResponse(BaseModel):
    query: str
    answer_before_check: str
    hallucination: str
    answer_after_check: str
    sources: list
    external: Optional[str] = ""
    similarity_score_internal: Optional[float] = 0.0
    similarity_score_web: Optional[float] = 0.0

# ============================
# RINGKASAN EKSTERNAL
# ============================

def summarize_external(external_text: str) -> str:
    try:
        llm = ChatOllama(model=MODEL)
        prompt = PromptTemplate.from_template("""
Berikut adalah hasil pencarian dari web:

{external_text}

Buatlah ringkasan yang akurat dan jelas dari informasi di atas, dalam 3-4 kalimat.
""")
        chain = prompt | llm
        response = chain.invoke({"external_text": external_text})
        if isinstance(response, list) and len(response) > 0 and hasattr(response[0], "content"):
            return response[0].content.strip()
        elif hasattr(response, "content"):
            return response.content.strip()
        else:
            return str(response).strip()

    except Exception as e:
        logger.error(f"[WEB-SUMMARY] Gagal merangkum hasil pencarian: {e}")
        return "Jawaban tidak cukup didukung dokumen internal. Berikut hasil pencarian:"

# ============================
# ENGINE UTAMA
# ============================

class AnswerEngine:
    def __init__(self):
        self.rag = RAG()
        self.checker = HallucinationChecker(threshold=0.90)
        self.checker_web = HallucinationChecker(threshold=0.85)
        self.web_search_tool = search_web

    def run(self, query: str) -> AskResponse:
        if not self.rag.is_ready():
            logger.warning("[ANSWER_ENGINE] Sistem belum siap digunakan.")
            return AskResponse(
                query=query,
                answer_before_check="",
                hallucination="YA",
                answer_after_check="Sistem belum siap.",
                sources=[],
            )

        # Langkah 1: Jawaban dari internal
        rag_result = self.rag.ask(query)
        raw_answer = rag_result["answer"]
        sources = rag_result["sources"]

        # Sanitasi jawaban
        if isinstance(raw_answer, list):
            if all(isinstance(item, dict) for item in raw_answer):
                answer = " ".join(str(item.get("content", "")) for item in raw_answer)
            else:
                answer = " ".join(str(item) for item in raw_answer)
        else:
            answer = str(raw_answer)

        answer = answer.strip()

        # Langkah 2: Cek halusinasi dari internal
        hallucination, similarity_internal = self.checker.check(answer, sources)

        # Langkah 3: Cek hasil dari web
        web_results = self.web_search_tool(query)
        web_contents = [content for _, content in web_results]
        web_combined = "\n".join(web_contents)
        similarity_web = self.checker_web.check_similarity(answer, web_combined)
        logger.info(f"[ANSWER_ENGINE] Skor kemiripan web: {similarity_web:.4f}")

        web_text = "\n".join([f"{title}: {snippet}" for title, snippet in web_results]) if isinstance(web_results, list) else str(web_results)

        # Langkah 4: Jika halusinasi, ringkas web
        if hallucination:
            if similarity_web >= 0.80:
                summary = summarize_external(web_text)
                return AskResponse(
                    query=query,
                    answer_before_check=answer,
                    hallucination="YA",
                    answer_after_check=summary,
                    sources=[],
                    external=web_text,
                    similarity_score_internal=round(similarity_internal, 4),
                    similarity_score_web=round(similarity_web, 4),
                )
            else:
                return AskResponse(
                    query=query,
                    answer_before_check=answer,
                    hallucination="YA",
                    answer_after_check="Jawaban tidak cukup didukung dokumen internal maupun pencarian eksternal.",
                    sources=[],
                    external=web_text,
                    similarity_score_internal=round(similarity_internal, 4),
                    similarity_score_web=round(similarity_web, 4),
                )

        # Langkah 5: Ambil sumber internal yang relevan (>= 0.90)
        relevant_sources = []
        for s in sources:
            try:
                score = self.checker.check_similarity(answer, s)
                if score >= 0.90:
                    relevant_sources.append({"source": "internal", "content": s})
            except Exception as e:
                logger.debug(f"[ANSWER_ENGINE] Gagal menghitung similarity source: {e}")

        return AskResponse(
            query=query,
            answer_before_check=answer,
            hallucination="TIDAK",
            answer_after_check=answer,
            sources=relevant_sources,
            external="",
            similarity_score_internal=round(similarity_internal, 4),
            similarity_score_web=round(similarity_web, 4),
        )
