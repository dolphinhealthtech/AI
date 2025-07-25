# LangChain Medical RAG

Sistem RAG (Retrieval-Augmented Generation) untuk menjawab pertanyaan medis berbasis dokumen klinis seperti rekam medis, pedoman diagnosis, dan SOP. Cocok untuk digunakan oleh dokter, klinik, atau sistem pendukung keputusan medis.

## Fitur

- Upload file medis (.pdf, .docx, .txt)
- Otomatis bangun ulang vectorstore setelah upload
- Penjawaban otomatis berbasis konteks dokumen
- Logging similarity dan jumlah dokumen
- Gunakan embedding lokal (`nomic-embed-text`) dan LLM lokal (`ollama` - Gemma 3B)

---

## Installation (Python + pip)

Install project dan dependencies dalam virtual environment.

### Langkah 1: Clone Project

```bash
git clone https://github.com/username/langchain-medical-rag
cd langchain-medical-rag
```

### Langkah 2: Buat dan Aktifkan Virtual Environment

```bash
python -m venv venv
```

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

- **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```

### Langkah 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage/Examples (FastAPI Endpoint)

### 1. **Upload File Medis**

```bash
curl --location 'http://localhost:8000/upload' --form 'file=@"dokumen_anda.pdf"'
```

Setelah upload berhasil, vectorstore akan diperbarui secara otomatis.

### 2. **Ajukan Pertanyaan**

```bash
curl --location 'http://localhost:8000/ask' --header 'Content-Type: application/json' --data '{
  "question": "nyeri dada menjalar ke lengan kiri, sesak napas, dan keringat dingin"
}'
```

---

## Run Locally

```bash
uvicorn main:app --reload
```

---

## Environment Variables

Tambahkan variabel lingkungan berikut ke file `.env` jika diperlukan:

```
LLM_MODEL=gemma:3b
EMBEDDING_MODEL=nomic-embed-text
```

---

## Folder Struktur

```
langchain-medical-rag/
├── data/                  # Dokumen upload (pdf, docx, txt)
├── vectorstore/           # ChromaDB vectorstore
├── main.py                # FastAPI endpoint
├── chains/
│   └── rag_chain.py       # RAG logic
├── loaders/
│   └── document_loader.py # Load berbagai format dokumen
├── learn/
│   └── fine_tune.py       # (opsional) Fine-tune retriever
├── models/
│   └── schemas.py         # Pydantic schemas
└── requirements.txt       # Python dependencies
```

---

## API Endpoints

| Method | Endpoint     | Deskripsi                                     |  Done  |
|--------|--------------|-----------------------------------------------|--------|
| POST   | `/ask`       | Kirim pertanyaan dan dapatkan jawaban medis   |   ✅  |
| POST   | `/upload`    | Upload file dokumen (.pdf/.docx/.txt)         |   ✅  |
| GET    | `/review`    | Review hasil pelatihan sebelumnya             |   ❌  |
| POST   | `/train`     | Proses indexing manual dokumen                |   ❌  |
| POST   | `/retrain`   | Melatih ulang retriever (fine-tune)           |   ❌  |

---

## Tools & Libraries Used

- 🔗 [LangChain](https://www.langchain.com/)
- 🧠 [Gemma 3B](https://ollama.com/library/gemma)
- 📦 [ChromaDB](https://www.trychroma.com/)
- 🔍 [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1)
- 🚀 [FastAPI](https://fastapi.tiangolo.com/)

---
