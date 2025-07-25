from fastapi import FastAPI, Request
from pydantic import BaseModel
from agents.medical_agent import get_medical_answer
from chains.rag_chain import get_context_from_rag
from langchain_ollama  import OllamaLLM
from dotenv import load_dotenv
import os
from fastapi import UploadFile, File
import shutil
from chains.rag_chain import create_vectorstore

# Load environment variables
load_dotenv()
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma-3b-it")

# Initialize FastAPI app and model
app = FastAPI()
llm = OllamaLLM(model=MODEL_NAME)

# Request schema
class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "LangChain + Gemma Medical API is running."}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    upload_folder = "data"
    destination = f"{upload_folder}/{file.filename}"

    # Simpan file ke folder data/
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Bangun ulang vectorstore
    create_vectorstore()

    return {"message": f"File '{file.filename}' berhasil diupload dan vectorstore diperbarui."}

@app.post("/ask")
async def ask(query: Query):
    # 1. Ambil konteks dari dokumen (RAG)
    context = get_context_from_rag(query.question)

    # 2. Dapatkan jawaban medis dalam format DiagnosisResult
    result = get_medical_answer(llm, query.question, context)

    # 3. Kembalikan hasil
    return {
        "question": query.question,
        "context_used": context,
        "result": result.dict()
    }
