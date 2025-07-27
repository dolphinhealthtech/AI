# learn/memory_store.py
import os
import json
from datetime import datetime
from utils.logger import get_logger

MEMORY_FILE = "memory/memory.jsonl"
logger = get_logger(__name__, "memory.log")

def save_to_memory(question: str, answer: str, source: str):
    """
    Menyimpan interaksi (pertanyaan, jawaban, sumber) ke memory.jsonl.

    Args:
        question (str): Pertanyaan user.
        answer (str): Jawaban LLM.
        source (str): Sumber dokumen yang digunakan.
    """
    try:
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "answer": answer,
            "source": source
        }

        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Singkat isi pertanyaan agar log tidak penuh
        short_q = (question[:50] + "...") if len(question) > 50 else question
        logger.debug(f"[MEMORY] Disimpan: '{short_q}' -> source: {source}")

    except Exception as e:
        logger.error(f"[MEMORY] Gagal menyimpan interaksi: {e}")
