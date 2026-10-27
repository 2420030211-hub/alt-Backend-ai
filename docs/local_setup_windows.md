# Windows Local Development Setup

## Prerequisites
- Node.js v18+ (tested with v20+)
- Python 3.10+
- PowerShell 7 or Windows PowerShell

## Service Execution

### Terminal 1: Node.js Express Gateway
```powershell
cd server
npm install
npm run dev
```

### Terminal 2: CORA FastAPI Microservice
```powershell
cd ai-services/cora
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8001 --reload
```

### Terminal 3: Qwen FastAPI Microservice
```powershell
cd ai-services/qwen
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8002 --reload
```
