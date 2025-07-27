# guardrails/compliance.py
from mediq_ai.embeddings.model_embeddings import OllamaEmbeddings
from utils.config import EMBEDDING_MODEL
from utils.logger import get_logger
import numpy as np

logger = get_logger(__name__, "compliance.log")

class HallucinationChecker:
    def __init__(self, threshold: float = 0.90):  # Default baru: 0.90 (bisa override)
        self.embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.threshold = threshold
        logger.info(f"[HALLUCINATION] Checker aktif dengan threshold: {self.threshold}")

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def check(self, answer: str, sources: list[str]) -> tuple[bool, float]:
        if not sources:
            logger.warning("[HALLUCINATION] Tidak ada sumber. Diasumsikan halusinasi.")
            return True, 0.0

        answer_clean = answer.strip().lower()
        if (
            len(answer_clean) < 30 or
            answer_clean.startswith("maaf, saya tidak menemukan") or
            "tidak ditemukan" in answer_clean or
            "tidak tersedia" in answer_clean
        ):
            logger.warning("[HALLUCINATION] Jawaban terlalu pendek atau tidak informatif. Diasumsikan halusinasi.")
            return True, 0.0

        try:
            logger.debug("[HALLUCINATION] Membuat embedding untuk jawaban...")
            ans_emb = np.array(self.embedder.embed_query(answer))

            similarities = []
            for i, src in enumerate(sources, 1):
                logger.debug(f"[HALLUCINATION] Membuat embedding untuk sumber #{i}...")
                src_emb = np.array(self.embedder.embed_query(src))
                sim = self.cosine_similarity(ans_emb, src_emb)
                similarities.append(sim)

            max_score = max(similarities)
            hallucinated = max_score < self.threshold
            logger.info(f"[HALLUCINATION] Skor kemiripan tertinggi: {max_score:.4f} | Threshold: {self.threshold}")

            return hallucinated, round(max_score, 4)

        except Exception as e:
            logger.error(f"[HALLUCINATION] Gagal memproses halusinasi: {e}")
            return True, 0.0

    def check_similarity(self, text1: str, text2: str) -> float:
        try:
            logger.debug("[HALLUCINATION] Membuat embedding untuk dua teks...")
            emb1 = np.array(self.embedder.embed_query(text1))
            emb2 = np.array(self.embedder.embed_query(text2))
            similarity = self.cosine_similarity(emb1, emb2)
            logger.info(f"[HALLUCINATION] Skor similarity antar teks: {similarity:.4f}")
            return round(similarity, 4)
        except Exception as e:
            logger.error(f"[HALLUCINATION] Gagal hitung similarity antar teks: {e}")
            return 0.0
