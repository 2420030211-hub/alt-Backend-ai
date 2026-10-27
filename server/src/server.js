'use strict';
const app = require('./app');
const config = require('./config/env');
const { logger } = require('./utils/helpers');

const server = app.listen(config.port, () => {
  logger.info(`Banking Complaint Management API running on port ${config.port}`, {
    port: config.port,
    environment: config.nodeEnv,
    coraUrl: config.aiServices.coraUrl,
    qwenUrl: config.aiServices.qwenUrl
  });
});

process.on('SIGTERM', () => {
  logger.info('SIGTERM signal received. Closing HTTP server...');
  server.close(() => {
    logger.info('HTTP server closed.');
    process.exit(0);
  });
});

module.exports = server;
