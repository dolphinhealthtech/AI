from pydantic import BaseModel
from typing import List

class Medication(BaseModel):
    name: str
    dosage: str
    route: str
    frequency: str
    duration: str

class SOAP(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str
    medications: List[Medication]

class DiagnosisResult(BaseModel):
    soap: SOAP
    icd_9: str
    icd_10: str
    treatment_plan: str
    referral_decision: str
