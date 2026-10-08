from typing import List
from pydantic import BaseModel


class RecognitionItem(BaseModel):
    label: str
    confidence: float


class KnowledgeItem(BaseModel):
    title: str
    summary: str
    date: str = ""
    source_url: str = ""


class AnalysisResponse(BaseModel):
    recognition: List[RecognitionItem] = []
    caption: str
    observation: str
    scientific_observation: str = ""
    facts: List[str] = []
    documents: List[str] = []
    knowledge: List[KnowledgeItem] = []