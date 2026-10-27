'use strict';
const crypto = require('crypto');

function info(message, meta = {}) {
  const ts = new Date().toISOString();
  console.log(JSON.stringify({ timestamp: ts, level: 'INFO', message, ...meta }));
}

function warn(message, meta = {}) {
  const ts = new Date().toISOString();
  console.warn(JSON.stringify({ timestamp: ts, level: 'WARN', message, ...meta }));
}

function error(message, meta = {}) {
  const ts = new Date().toISOString();
  console.error(JSON.stringify({ timestamp: ts, level: 'ERROR', message, ...meta }));
}

function hashPassword(password, salt = crypto.randomBytes(16).toString('hex')) {
  const hash = crypto.scryptSync(password, salt, 64).toString('hex');
  return { salt, hash };
}

function verifyPassword(password, user) {
  if (!user || !user.salt || !user.hash) return false;
  const a = Buffer.from(crypto.scryptSync(password, user.salt, 64).toString('hex'), 'hex');
  const b = Buffer.from(user.hash, 'hex');
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

function generateTicketId() {
  const year = new Date().getFullYear();
  const rand = Math.floor(100000 + Math.random() * 900000);
  return `CMP-${year}-${rand}`;
}

module.exports = {
  logger: { info, warn, error },
  hashPassword,
  verifyPassword,
  generateTicketId
};
