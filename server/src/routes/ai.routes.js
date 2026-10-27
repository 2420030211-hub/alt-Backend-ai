'use strict';
const express = require('express');
const router = express.Router();
const aiController = require('../controllers/ai.controller');

router.post('/classify', aiController.classify);

module.exports = router;
