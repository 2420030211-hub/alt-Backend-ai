'use strict';
const express = require('express');
const router = express.Router();
const complaintController = require('../controllers/complaint.controller');
const { requireAuth, requireEmployee } = require('../middleware/auth.middleware');

router.get('/complaints', requireAuth, requireEmployee, complaintController.list);
router.patch('/complaints/:id', requireAuth, requireEmployee, complaintController.updateEmployee);

module.exports = router;
