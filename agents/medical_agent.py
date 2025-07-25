import os
import json
import re
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.language_models import BaseLanguageModel
from models.schemas import DiagnosisResult

load_dotenv()
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma-3b-it")
llm = OllamaLLM(model=MODEL_NAME)

def build_prompt(user_question: str, context: str = "") -> str:
    context_note = "Berikut konteks dari dokumen medis:\n" + context if context else \
        "Catatan: Tidak ada dokumen referensi yang ditemukan. Jawaban diberikan berdasarkan pengetahuan medis umum model."

    return f"""
Anda adalah asisten medis yang menjawab pertanyaan klinis dari dokter.
Berikan jawaban dalam format JSON berikut:

{{
  "soap": {{
    "subjective": "...",
    "objective": "...",
    "assessment": "...",
    "plan": "...",
    "medications": [
      {{
        "name": "...",
        "dosage": "...",
        "route": "...",
        "frequency": "...",
        "duration": "..."
      }}
    ]
  }},
  "icd_9": "...",
  "icd_10": "...",
  "treatment_plan": "...",
  "referral_decision": "..."
}}

{context_note}

Pertanyaan:
{user_question}
"""

def extract_json(text: str) -> str:
    # Cari blok JSON dari model, termasuk jika dia pakai tag ```json
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)

    # Jika tidak pakai tag, cari JSON pertama
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1)

    # Fallback: return teks mentah
    return text

def get_medical_answer(llm: BaseLanguageModel, question: str, context: str = "") -> DiagnosisResult:
    prompt = build_prompt(question, context)
    output = llm.invoke(prompt)

    # Ekstrak JSON dari output model
    json_text = extract_json(output)
    try:
        parsed = json.loads(json_text)
        return DiagnosisResult(**parsed)
    except Exception as e:
        raise ValueError(f"Gagal parsing output model: {e}\nOutput:\n{output}")
