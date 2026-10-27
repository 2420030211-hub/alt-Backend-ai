'use strict';
const express = require('express');
const router = express.Router();
const complaintController = require('../controllers/complaint.controller');
const { requireAuth } = require('../middleware/auth.middleware');

router.post('/', complaintController.create);
router.get('/', requireAuth, complaintController.list);
router.get('/mine', requireAuth, complaintController.list);
router.get('/:id', complaintController.getById);
router.get('/:id/status', complaintController.getStatus);

module.exports = router;
