# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models.schemas import AskRequest, AskResponse
from answer_engine import AnswerEngine
from learn.memory_store import save_to_memory
from utils.logger import get_logger

logger = get_logger(__name__, "main.log")

app = FastAPI(
    title="Medical RAG API",
    description="Menjawab pertanyaan medis berdasarkan dokumen menggunakan RAG + Ollama + FAISS",
    version="1.0.0",
)

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-inisialisasi engine di luar endpoint untuk efisiensi
engine = AnswerEngine()

# === Endpoint: /ask ===
@app.post("/ask", response_model=AskResponse)
def ask_question(data: AskRequest):
    query = data.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong.")

    logger.info(f"[ASK] Pertanyaan diterima")

    try:
        response = engine.run(query)

        # Simpan hanya jika jawaban valid (bukan halusinasi) dan sumber internal tersedia
        if response.hallucination == "TIDAK" and response.sources:
            save_to_memory(
                question=query,
                answer=response.answer_after_check,
                source=response.sources[0].get("source", "internal")
            )

        return response

    except Exception as e:
        logger.exception(f"[ASK] Gagal memproses pertanyaan: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat memproses pertanyaan.")

# === Untuk development lokal ===
if __name__ == "__main__":
    import uvicorn
    logger.info("Menjalankan server FastAPI di http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
