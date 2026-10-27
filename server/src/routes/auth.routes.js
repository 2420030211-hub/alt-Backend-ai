'use strict';
const express = require('express');
const router = express.Router();
const authController = require('../controllers/auth.controller');
const { requireAuth } = require('../middleware/auth.middleware');

router.post('/register', authController.register);
router.post('/login', authController.login);
router.post('/employee-login', authController.employeeLogin);
router.get('/me', authController.me);
router.post('/logout', requireAuth, authController.logout);

module.exports = router;
