'use strict';
const express = require('express');
const cors = require('cors');
const config = require('./config/env');
const { logger } = require('./utils/helpers');
const { authMiddleware } = require('./middleware/auth.middleware');

const authRoutes = require('./routes/auth.routes');
const complaintRoutes = require('./routes/complaint.routes');
const employeeRoutes = require('./routes/employee.routes');
const aiRoutes = require('./routes/ai.routes');
const chatRoutes = require('./routes/chat.routes');

const app = express();

app.use(cors({
  origin: (origin, callback) => {
    if (!origin || config.corsOrigins.includes(origin) || origin.startsWith('http://localhost:')) {
      callback(null, true);
    } else {
      callback(null, true); // Allow during local test/dev
    }
  },
  credentials: true
}));

app.use(express.json({ limit: '2mb' }));
app.use(express.urlencoded({ extended: true }));
app.use(authMiddleware);

// Health & Readiness Endpoints
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

app.get('/api/ready', async (req, res) => {
  // Check microservices status
  let coraOnline = false;
  let qwenOnline = false;

  try {
    const coraRes = await fetch(`${config.aiServices.coraUrl}/ready`, { signal: AbortSignal.timeout(1000) });
    coraOnline = coraRes.ok;
  } catch (_) {}

  try {
    const qwenRes = await fetch(`${config.aiServices.qwenUrl}/ready`, { signal: AbortSignal.timeout(1000) });
    qwenOnline = qwenRes.ok;
  } catch (_) {}

  res.json({
    status: 'ready',
    services: {
      backend: 'online',
      coraService: coraOnline ? 'online' : (config.aiServices.mockFallback ? 'fallback-mode' : 'offline'),
      qwenService: qwenOnline ? 'online' : (config.aiServices.mockFallback ? 'fallback-mode' : 'offline')
    },
    mockFallbackEnabled: config.aiServices.mockFallback
  });
});

// App API Routes
app.use('/api/auth', authRoutes);
app.use('/api/complaints', complaintRoutes);
app.use('/api/employee', employeeRoutes);
app.use('/api/ai', aiRoutes);
app.use('/api/chat', chatRoutes);

// Centralized error handler
app.use((err, req, res, next) => {
  logger.error('Unhandled request error', { error: err.message, stack: err.stack, path: req.path });
  const status = err.statusCode || 500;
  res.status(status).json({
    error: err.message || 'Internal Server Error'
  });
});

module.exports = app;
