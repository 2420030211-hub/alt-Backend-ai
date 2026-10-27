'use strict';
const crypto = require('crypto');
const config = require('../config/env');
const { hashPassword } = require('../utils/helpers');

class MemoryStore {
  constructor() {
    this.users = new Map();         // userId -> user
    this.sessions = new Map();      // token -> { userId, expiresAt }
    this.complaints = new Map();    // ticketId -> complaint
    this.seedDemoAccounts();
  }

  seedDemoAccounts() {
    const { id, email, password, fullName } = config.demoEmployee;
    const { salt, hash } = hashPassword(password);
    const userId = crypto.randomUUID();
    const user = {
      id: userId,
      role: 'employee',
      employeeId: id,
      email,
      fullName,
      salt,
      hash,
      createdAt: new Date().toISOString()
    };
    this.users.set(userId, user);
  }

  // Users
  createUser({ email, password, fullName, phone, role = 'customer', employeeId = null }) {
    const existing = this.findUserByEmail(email);
    if (existing) {
      const err = new Error('Email already registered');
      err.statusCode = 409;
      throw err;
    }
    const { salt, hash } = hashPassword(password);
    const id = crypto.randomUUID();
    const user = {
      id,
      role,
      fullName,
      email: email.toLowerCase(),
      phone: phone || '',
      employeeId: employeeId || null,
      salt,
      hash,
      createdAt: new Date().toISOString()
    };
    this.users.set(id, user);
    return this.sanitizeUser(user);
  }

  findUserById(id) {
    return this.users.get(id) || null;
  }

  findUserByEmail(email) {
    if (!email) return null;
    const lower = email.toLowerCase().trim();
    for (const user of this.users.values()) {
      if (user.email === lower) return user;
    }
    return null;
  }

  findEmployeeById(employeeId) {
    if (!employeeId) return null;
    const cleanId = employeeId.trim().toUpperCase();
    for (const user of this.users.values()) {
      if (user.role === 'employee' && user.employeeId && user.employeeId.toUpperCase() === cleanId) {
        return user;
      }
    }
    return null;
  }

  sanitizeUser(user) {
    if (!user) return null;
    const { salt, hash, ...safe } = user;
    return safe;
  }

  // Sessions
  createSession(userId, durationMs = 8 * 60 * 60 * 1000) {
    const token = crypto.randomBytes(32).toString('hex');
    const expiresAt = Date.now() + durationMs;
    this.sessions.set(token, { userId, expiresAt });
    return token;
  }

  getSession(token) {
    if (!token) return null;
    const session = this.sessions.get(token);
    if (!session) return null;
    if (Date.now() > session.expiresAt) {
      this.sessions.delete(token);
      return null;
    }
    const user = this.findUserById(session.userId);
    return user ? this.sanitizeUser(user) : null;
  }

  destroySession(token) {
    if (token) this.sessions.delete(token);
  }

  // Complaints
  createComplaint(complaintData) {
    this.complaints.set(complaintData.ticketId, complaintData);
    return complaintData;
  }

  getComplaint(ticketId) {
    return this.complaints.get(ticketId) || null;
  }

  listComplaints(filterFn) {
    const all = Array.from(this.complaints.values());
    return filterFn ? all.filter(filterFn) : all;
  }

  updateComplaint(ticketId, updates) {
    const existing = this.complaints.get(ticketId);
    if (!existing) return null;
    const updated = {
      ...existing,
      ...updates,
      updatedAt: new Date().toISOString()
    };
    this.complaints.set(ticketId, updated);
    return updated;
  }
}

module.exports = new MemoryStore();
