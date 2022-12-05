from pydantic import BaseModel

class SearchResponse(BaseModel):
    doc_id: str
    score: float
    contents: str