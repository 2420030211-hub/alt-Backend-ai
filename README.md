# Banking Complaint Management AI Backend

Enterprise multi-service backend architecture integrating Node.js/Express, CORA Hierarchical Complaint Classification, and Qwen Conversational Assistant.

## Architecture Overview

```text
banking-complaint-backend/
├── server/          # Node.js/Express API (Auth, Complaints, RBAC, Business Rules)
├── ai-services/     # Python FastAPI AI Microservices
│   ├── cora/        # CORA Hierarchical Classifier (Product, Sub-product, Issue, Sub-issue)
│   └── qwen/        # Qwen3-4B Conversational Banking Assistant
├── datasets/        # Audited datasets and splits
├── artifacts/       # Saved models, vectorizers, and checkpoints
└── docs/            # Comprehensive Architecture & Deployment Guides
```

## Quick Start (Windows PowerShell)

### 1. Start Node.js API Gateway (Port 4000)
```powershell
cd server
npm install
npm start
```

### 2. Start CORA Classification Microservice (Port 8001)
```powershell
cd ai-services/cora
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8001
```

### 3. Start Qwen Assistant Microservice (Port 8002)
```powershell
cd ai-services/qwen
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8002
```

## Running Unit & Regression Tests
```powershell
cd server
npm test
```
