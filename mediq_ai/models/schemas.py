# models/schemas.py

from pydantic import BaseModel
from typing import List, Optional

class SourceDoc(BaseModel):
    source: str
    content: str

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    query: str
    answer_before_check: str
    hallucination: str  # "YA" atau "TIDAK"
    answer_after_check: str
    sources: List[SourceDoc]
    external: Optional[str] = ""
    similarity_score_internal: Optional[float] = 0.0
    similarity_score_web: Optional[float] = 0.0