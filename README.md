# 🧠 Mediq AI

**Mediq AI** adalah platform NLP medis berbasis **LangChain** yang memungkinkan analisis pertanyaan, ringkasan diagnosis, deteksi halusinasi, dan integrasi dengan pencarian eksternal. Cocok untuk klinik, rumah sakit, atau pengembang sistem decision support medis berbasis dokumen.

---

## 📁 Struktur Proyek

```
mediq_ai/
├── mediq_ai/                # Modul utama Python
│   ├── main.py              # FastAPI app (entry point)
│   ├── answer_engine.py     # Mesin jawaban utama
│   ├── .env                 # Variabel lingkungan (API key, dll)
│   ├── rag/                 # Retrieval-Augmented Generation
│   │   └── rag_chain.py
│   ├── agents/              # Agent medis untuk reasoning
│   │   └── medical_agent.py
│   ├── loaders/             # Pemuat dokumen
│   │   └── document_loader.py
│   ├── embeddings/          # Model embedding lokal
│   │   └── model_embeddings.py
│   ├── guardrails/          # Pemeriksa halusinasi & validasi jawaban
│   │   └── compliance.py
│   ├── learn/               # Memory store atau adaptive learning
│   │   └── memory_store.py
│   ├── tools/               # Tools tambahan (search, dsb)
│   │   └── external_search_tool.py
│   ├── models/              # Skema Pydantic
│   │   └── schemas.py
│   ├── utils/               # Konfigurasi & logging
│   │   ├── config.py
│   │   └── logger.py
├── tunning-data/            # Preprocessing & vectorizing dokumen
│   └── embedder.py
├── data/                    # Folder input dokumen mentah
├── memory/                  # Temporary memory agent
├── vectorstore/             # Simpanan hasil embedding
├── requirements.txt         # Daftar dependensi
├── setup.py                 # Konfigurasi packaging
├── README.md                # Dokumentasi ini
```

---

## 🚀 Fitur Utama

- 🔍 **RAG (Retrieval-Augmented Generation)** berbasis dokumen lokal.
- 🧠 **Agent reasoning** untuk diagnosis dan konsultasi medis.
- 🧬 **Embeddings lokal** via `Ollama` + `nomic-embed-text`.
- 🔒 **Guardrails**: cek halusinasi via cosine similarity antar embeddings.
- 🌐 **Integrasi pencarian eksternal** bila jawaban tidak ditemukan secara lokal.
- 🔄 **Retraining retriever** berbasis JSONL via endpoint `/retrain`.

---

## ⚙️ Cara Menjalankan

```bash
# 1. Clone repository
git clone https://github.com/namamu/mediq_ai.git
cd mediq_ai

# 2. Membuat virtual environment Python 
python -m venv venv

# 3. Activkan virtual environment Python
venv/Scripts/Activate

# 4. Install dependensi
pip install -r requirements.txt

# 5. Jalankan server FastAPI
uvicorn mediq_ai.main:app --reload
```

---

## 📬 API Endpoint

| Endpoint       | Fungsi                                          |
|----------------|-------------------------------------------------|
| `/ask`         | Menjawab pertanyaan berbasis dokumen & agent    |
| `/train`       | Preprocessing dokumen dan membuat embeddings    |

---

## 🛠 Teknologi

- **Python 3.10+**
- **FastAPI**
- **LangChain**
- **Ollama (LLM lokal)**
- **FAISS**
- **Nomic Embedding / Sentence Transformers**

---

## 📄 Lisensi

Proyek ini dikembangkan untuk riset dan pengembangan sistem NLP medis. Untuk penggunaan komersial atau distribusi, silakan hubungi pengembang.

---

## 👨‍💻 Developer

Dikembangkan oleh: **PT. Dolphin Teknologi Kesehatan**  
🌍 Bandung, Indonesia

---