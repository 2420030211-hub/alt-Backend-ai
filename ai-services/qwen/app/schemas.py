from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class MessageItem(BaseModel):
    role: str # 'user', 'assistant', 'system'
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User prompt or complaint text")
    conversation_id: Optional[str] = None
    history: Optional[List[MessageItem]] = []
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    reply: str
    conversation_id: str
    model: str
    source: str

class QwenHealthResponse(BaseModel):
    status: str
    service: str
    model_name: str
    model_loaded: bool
    device: str
