from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import ClassifyRequest, ClassifyResponse, HealthResponse
from .config import config
from .model_loader import model_loader
from .inference import run_inference

app = FastAPI(
    title="CORA - Banking Complaint Hierarchical Classification Service",
    version="1.0.0",
    description="Microservice for predicting Product, Sub-Product, Issue, and Sub-Issue"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        service=config.service_name,
        model_loaded=model_loader.model_loaded,
        checkpoint_path=model_loader.loaded_checkpoint_path
    )

@app.get("/ready")
def ready():
    # Service responds whether model checkpoint is present or serving baseline
    return {
        "ready": True,
        "model_loaded": model_loader.model_loaded,
        "checkpoint": model_loader.loaded_checkpoint_path,
        "mode": "trained_checkpoint" if model_loader.model_loaded else "baseline_service"
    }

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    if not req.text or len(req.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text field must not be empty")
    result = run_inference(req.text)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.host, port=config.port, reload=True)
