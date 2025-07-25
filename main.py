from fastapi import FastAPI, Request
from pydantic import BaseModel
from agents.medical_agent import get_medical_answer
from chains.rag_chain import get_context_from_rag
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma-3b-it")

# Initialize FastAPI app and model
app = FastAPI()
llm = Ollama(model=MODEL_NAME)

# Request schema
class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "LangChain + Gemma Medical API is running."}

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
