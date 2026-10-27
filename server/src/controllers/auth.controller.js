'use strict';
const storage = require('../services/storage.service');
const { verifyPassword } = require('../utils/helpers');

class AuthController {
  async register(req, res, next) {
    try {
      const { email, password, fullName, phone } = req.body;
      if (!email || !password || !fullName) {
        return res.status(400).json({ error: 'Full name, email and password are required' });
      }
      if (password.length < 8) {
        return res.status(400).json({ error: 'Password must be at least 8 characters long' });
      }

      const user = storage.createUser({ email, password, fullName, phone, role: 'customer' });
      const token = storage.createSession(user.id);

      res.cookie('sid', token, {
        httpOnly: true,
        sameSite: 'lax',
        maxAge: 8 * 3600 * 1000
      });

      return res.status(201).json({
        message: 'Registration successful',
        user,
        token
      });
    } catch (err) {
      if (err.statusCode) {
        return res.status(err.statusCode).json({ error: err.message });
      }
      next(err);
    }
  }

  async login(req, res, next) {
    try {
      const { email, password } = req.body;
      if (!email || !password) {
        return res.status(400).json({ error: 'Email and password are required' });
      }

      const user = storage.findUserByEmail(email);
      if (!user || user.role !== 'customer' || !verifyPassword(password, user)) {
        return res.status(401).json({ error: 'Invalid email or password' });
      }

      const token = storage.createSession(user.id);
      res.cookie('sid', token, {
        httpOnly: true,
        sameSite: 'lax',
        maxAge: 8 * 3600 * 1000
      });

      return res.json({
        message: 'Login successful',
        user: storage.sanitizeUser(user),
        token
      });
    } catch (err) {
      next(err);
    }
  }

  async employeeLogin(req, res, next) {
    try {
      const { employeeId, password } = req.body;
      if (!employeeId || !password) {
        return res.status(400).json({ error: 'Employee ID and password are required' });
      }

      const user = storage.findEmployeeById(employeeId);
      if (!user || !verifyPassword(password, user)) {
        return res.status(401).json({ error: 'Invalid Employee credentials' });
      }

      const token = storage.createSession(user.id);
      res.cookie('sid', token, {
        httpOnly: true,
        sameSite: 'lax',
        maxAge: 8 * 3600 * 1000
      });

      return res.json({
        message: 'Employee authentication successful',
        user: storage.sanitizeUser(user),
        token
      });
    } catch (err) {
      next(err);
    }
  }

  async me(req, res) {
    if (!req.user) {
      return res.status(401).json({ authenticated: false, user: null });
    }
    return res.json({ authenticated: true, user: req.user });
  }

  async logout(req, res) {
    if (req.token) {
      storage.destroySession(req.token);
    }
    res.clearCookie('sid');
    return res.json({ message: 'Logged out successfully' });
  }
}

module.exports = new AuthController();
