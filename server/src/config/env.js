'use strict';
require('dotenv').config();

module.exports = {
  port: parseInt(process.env.PORT, 10) || 4000,
  nodeEnv: process.env.NODE_ENV || 'development',
  corsOrigins: (process.env.CORS_ORIGIN || 'http://localhost:3000,http://127.0.0.1:3000').split(',').map(s => s.trim()),
  demoEmployee: {
    id: process.env.DEMO_EMPLOYEE_ID || 'EMP1001',
    email: (process.env.DEMO_EMPLOYEE_EMAIL || 'employee@demo.local').toLowerCase(),
    password: process.env.DEMO_EMPLOYEE_PASSWORD || 'Employee@123',
    fullName: process.env.DEMO_EMPLOYEE_NAME || 'Demo Compliance Officer'
  },
  aiServices: {
    coraUrl: process.env.CORA_SERVICE_URL || 'http://localhost:8001',
    qwenUrl: process.env.QWEN_SERVICE_URL || 'http://localhost:8002',
    mockFallback: process.env.AI_MOCK_FALLBACK !== 'false',
    confidenceThreshold: parseFloat(process.env.CORA_CONFIDENCE_THRESHOLD) || 0.65
  }
};
