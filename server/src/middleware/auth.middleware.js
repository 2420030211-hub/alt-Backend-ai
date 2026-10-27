'use strict';
const storage = require('../services/storage.service');

function parseCookies(req) {
  const list = {};
  const rc = req.headers.cookie;
  if (rc) {
    rc.split(';').forEach(cookie => {
      const parts = cookie.split('=');
      list[parts.shift().trim()] = decodeURI(parts.join('='));
    });
  }
  return list;
}

function authMiddleware(req, res, next) {
  let token = null;

  // 1. Check HttpOnly cookie
  const cookies = parseCookies(req);
  if (cookies.sid) {
    token = cookies.sid;
  }

  // 2. Fallback to Authorization Header
  const authHeader = req.headers.authorization;
  if (!token && authHeader && authHeader.startsWith('Bearer ')) {
    token = authHeader.substring(7).trim();
  }

  if (!token) {
    req.user = null;
    return next();
  }

  const user = storage.getSession(token);
  req.user = user;
  req.token = token;
  next();
}

function requireAuth(req, res, next) {
  if (!req.user) {
    return res.status(401).json({ error: 'Authentication required. Please log in.' });
  }
  next();
}

function requireEmployee(req, res, next) {
  if (!req.user || req.user.role !== 'employee') {
    return res.status(403).json({ error: 'Access denied: Employee credentials required.' });
  }
  next();
}

module.exports = {
  authMiddleware,
  requireAuth,
  requireEmployee
};
