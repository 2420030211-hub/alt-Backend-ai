from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
from .schemas import ChatRequest, ChatResponse, QwenHealthResponse
from .config import config
from .model_loader import qwen_loader
from .inference import generate_response
from .conversation import conversation_mgr

app = FastAPI(
    title="Qwen3-4B Banking Conversational Service",
    version="1.0.0",
    description="Dedicated microservice for customer complaints assistance & conversational banking"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=QwenHealthResponse)
def health():
    return QwenHealthResponse(
        status="healthy",
        service=config.service_name,
        model_name=config.model_name,
        model_loaded=qwen_loader.model_loaded,
        device=config.device
    )

@app.get("/ready")
def ready():
    return {
        "ready": True,
        "model_name": config.model_name,
        "model_loaded": qwen_loader.model_loaded,
        "mode": "neural_weights" if qwen_loader.model_loaded else "baseline_conversational_engine"
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    conv_id = req.conversation_id or f"conv-{uuid.uuid4().hex[:8]}"
    
    # Store turn
    conversation_mgr.append_turn(conv_id, "user", req.message)
    
    res = generate_response(req.message, conv_id, req.history)
    conversation_mgr.append_turn(conv_id, "assistant", res["reply"])

    return ChatResponse(
        reply=res["reply"],
        conversation_id=conv_id,
        model=res["model"],
        source=res["source"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.host, port=config.port, reload=True)
