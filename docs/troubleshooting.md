# Troubleshooting Guide

## Common Issues & Resolutions

### 1. "CORA service unavailable or timed out"
- **Cause**: Port 8001 is not running, or network is blocked.
- **Handling**: The Node.js gateway automatically falls back to the compliant internal business rule classifier (`cora-rule-fallback`) if `AI_MOCK_FALLBACK=true`.
- **Fix**: Launch the CORA microservice in a dedicated terminal window: `uvicorn app.main:app --port 8001`.

### 2. "Qwen service offline or timed out"
- **Cause**: Port 8002 is not running.
- **Handling**: The Node.js gateway falls back to the safety-guarded banking assistant.
- **Fix**: Launch `uvicorn app.main:app --port 8002`.

### 3. Missing CUDA / Out of Memory on Local PC
- **Solution**: Keep local development on CPU/mock baseline mode. Run heavy transformer fine-tuning on RunPod with an RTX 3090/4090.
