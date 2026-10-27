# System Architecture

## Component Separation

1. **Frontend**: Pure client application (`ALT-FrontendClaude`), completely isolated, unaltered.
2. **Backend Server**: Node.js/Express (`server/`) handling sessions, RBAC, business rule state machines, and AI service orchestration.
3. **CORA Microservice**: FastAPI service (`ai-services/cora`) providing hierarchical complaint prediction (Product -> Sub-product -> Issue -> Sub-issue).
4. **Qwen Conversational Service**: FastAPI service (`ai-services/qwen`) hosting Qwen3-4B conversational reasoning with prompt guards and banking safety policies.

```mermaid
graph TD
    Client[ALT-Frontend] -->|JSON / SSE| Gateway[Node.js Gateway Port 4000]
    Gateway -->|Auth / Storage| Store[(In-Memory Storage)]
    Gateway -->|POST /classify| CORA[CORA Service Port 8001]
    Gateway -->|POST /chat| Qwen[Qwen Service Port 8002]
    CORA --> Model1[BERT / ModernBERT / TF-IDF]
    Qwen --> Model2[Qwen3-4B]
```
