# prompts/prompting.py

from langchain.prompts import PromptTemplate

RAG_PROMPT_TEMPLATE = """
Anda adalah asisten medis berbasis AI. Berdasarkan informasi berikut yang diambil dari dokumen medis:

----------------
{context}
----------------

Tolong analisis dan berikan jawaban medis profesional berdasarkan data di atas untuk pertanyaan berikut.
Jawaban harus mencakup Diagnosis dan Terapi/Rencana Tatalaksana jika relevan.

Jika jawaban tidak tersedia dalam konteks, cukup jawab:
"Maaf, saya tidak menemukan informasi tersebut dalam dokumen yang tersedia."

Pertanyaan:
{question}

Jawaban:
"""

def get_prompt():
    return PromptTemplate(
        input_variables=["context", "question"],
        template=RAG_PROMPT_TEMPLATE,
    )
