from pydantic import BaseModel, Field
from typing import Optional, Dict

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Complaint raw narrative text")

class ConfidenceScores(BaseModel):
    product: float
    sub_product: float
    issue: float
    sub_issue: float
    overall: float

class ClassifyResponse(BaseModel):
    product: str
    sub_product: str
    issue: str
    sub_issue: str
    confidence: ConfidenceScores
    routed_to: str
    hierarchy_valid: bool
    model_version: str

class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    checkpoint_path: Optional[str] = None
