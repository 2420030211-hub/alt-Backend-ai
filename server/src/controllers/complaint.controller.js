'use strict';
const storage = require('../services/storage.service');
const coraService = require('../services/cora.service');
const businessRules = require('../services/business-rules.service');
const { generateTicketId, logger } = require('../utils/helpers');

class ComplaintController {
  async create(req, res, next) {
    try {
      const { text, subject, product, subProduct, issue, subIssue } = req.body;
      const cleanText = businessRules.sanitizeText(text || subject || '');

      if (!cleanText || cleanText.length < 5) {
        return res.status(400).json({ error: 'Complaint description is too short (minimum 5 characters).' });
      }

      // Perform CORA hierarchical classification if labels not fully specified
      let classification = null;
      let finalProduct = product;
      let finalSubProduct = subProduct;
      let finalIssue = issue;
      let finalSubIssue = subIssue;

      try {
        classification = await coraService.classify(cleanText);
        finalProduct = finalProduct || classification.product;
        finalSubProduct = finalSubProduct || classification.subProduct;
        finalIssue = finalIssue || classification.issue;
        finalSubIssue = finalSubIssue || classification.subIssue;
      } catch (classifyErr) {
        logger.warn('Automated classification failed or bypassed', { error: classifyErr.message });
      }

      // Validate hierarchy business rules
      const hierarchyCheck = businessRules.validateHierarchy(finalProduct, finalSubProduct, finalIssue, finalSubIssue);

      const ticketId = generateTicketId();
      const customerId = req.user ? req.user.id : 'guest';
      const customerEmail = req.user ? req.user.email : (req.body.email || 'guest@portal.local');

      const complaint = {
        ticketId,
        customerId,
        customerEmail,
        customerName: req.user ? req.user.fullName : (req.body.fullName || 'Valued Customer'),
        text: cleanText,
        subject: subject || cleanText.substring(0, 60) + '...',
        product: finalProduct || 'Checking or savings account',
        subProduct: finalSubProduct || 'Checking account',
        issue: finalIssue || 'Managing an account',
        subIssue: finalSubIssue || 'Deposits and withdrawals',
        hierarchyValid: hierarchyCheck.isValid,
        hierarchyWarning: hierarchyCheck.reason,
        aiClassification: classification,
        status: 'Submitted',
        assignedTo: null,
        resolutionNotes: null,
        history: [
          {
            status: 'Submitted',
            timestamp: new Date().toISOString(),
            by: customerId,
            note: 'Complaint lodged through portal'
          }
        ],
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };

      storage.createComplaint(complaint);

      return res.status(201).json({
        message: 'Complaint successfully created',
        ticketId: complaint.ticketId,
        complaint
      });
    } catch (err) {
      next(err);
    }
  }

  async list(req, res, next) {
    try {
      let list = [];
      if (!req.user) {
        return res.status(401).json({ error: 'Authentication required' });
      }

      if (req.user.role === 'employee') {
        list = storage.listComplaints();
      } else {
        list = storage.listComplaints(c => c.customerId === req.user.id);
      }

      return res.json({ complaints: list });
    } catch (err) {
      next(err);
    }
  }

  async getById(req, res, next) {
    try {
      const { id } = req.params;
      const complaint = storage.getComplaint(id);

      if (!complaint) {
        return res.status(404).json({ error: `Complaint #${id} not found` });
      }

      // Check ownership unless employee
      if (req.user && req.user.role !== 'employee' && complaint.customerId !== req.user.id) {
        return res.status(403).json({ error: 'Access forbidden: You do not own this complaint.' });
      }

      return res.json({ complaint });
    } catch (err) {
      next(err);
    }
  }

  async getStatus(req, res, next) {
    try {
      const { id } = req.params;
      const complaint = storage.getComplaint(id);

      if (!complaint) {
        return res.status(404).json({ error: `Complaint #${id} not found` });
      }

      return res.json({
        ticketId: complaint.ticketId,
        status: complaint.status,
        product: complaint.product,
        updatedAt: complaint.updatedAt,
        history: complaint.history
      });
    } catch (err) {
      next(err);
    }
  }

  async updateEmployee(req, res, next) {
    try {
      const { id } = req.params;
      const { status, assignedTo, resolutionNotes } = req.body;

      const complaint = storage.getComplaint(id);
      if (!complaint) {
        return res.status(404).json({ error: `Complaint #${id} not found` });
      }

      const updates = {};

      if (status && status !== complaint.status) {
        if (!businessRules.canTransitionStatus(complaint.status, status)) {
          return res.status(400).json({
            error: `Invalid status transition from '${complaint.status}' to '${status}'.`
          });
        }
        updates.status = status;
        const historyItem = {
          status,
          timestamp: new Date().toISOString(),
          by: req.user.employeeId || req.user.id,
          note: resolutionNotes || `Status updated to ${status}`
        };
        updates.history = [...complaint.history, historyItem];
      }

      if (assignedTo !== undefined) {
        updates.assignedTo = assignedTo;
      }

      if (resolutionNotes !== undefined) {
        updates.resolutionNotes = resolutionNotes;
      }

      const updatedComplaint = storage.updateComplaint(id, updates);

      return res.json({
        message: 'Complaint updated successfully',
        complaint: updatedComplaint
      });
    } catch (err) {
      next(err);
    }
  }
}

module.exports = new ComplaintController();
