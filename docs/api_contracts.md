# API Contracts & Integration Points

## 1. Authentication
- `POST /api/auth/register`: `{ email, password, fullName, phone }`
- `POST /api/auth/login`: `{ email, password }`
- `POST /api/auth/employee-login`: `{ employeeId, password }`
- `GET /api/auth/me`: Cookie `sid` or `Bearer <token>`

## 2. Complaints Management
- `POST /api/complaints`: `{ text, subject, product?, subProduct?, issue?, subIssue? }`
- `GET /api/complaints`: Returns customer-owned complaints (or all complaints for employees).
- `GET /api/complaints/:id`: Detailed complaint information.
- `GET /api/complaints/:id/status`: Public or customer status check.
- `PATCH /api/employee/complaints/:id`: `{ status, assignedTo, resolutionNotes }`

## 3. AI Services
- `POST /api/ai/classify`: `{ text }` -> `{ product, subProduct, issue, subIssue, confidence, routedTo }`
- `POST /api/chat`: `{ message, conversationId?, history?, stream?: boolean }`
  - Supports standard JSON response
  - Supports SSE streaming format (`data: {"token":"..."}\n\n`) matching frontend adapter.
