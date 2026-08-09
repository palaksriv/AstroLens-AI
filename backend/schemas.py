from pydantic import BaseModel
class AnalysisResponse(BaseModel):
    caption: str
    documents: list
    observation: str