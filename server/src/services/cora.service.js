'use strict';
const config = require('../config/env');
const { logger } = require('../utils/helpers');

class CoraService {
  constructor() {
    this.baseUrl = config.aiServices.coraUrl;
    this.fallbackEnabled = config.aiServices.mockFallback;
    this.threshold = config.aiServices.confidenceThreshold;
  }

  async classify(text) {
    if (!text || typeof text !== 'string' || text.trim().length === 0) {
      throw new Error('Classification requires non-empty text input');
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 4000);

      const response = await fetch(`${this.baseUrl}/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        return {
          source: 'cora-fastapi',
          product: data.product,
          subProduct: data.sub_product,
          issue: data.issue,
          subIssue: data.sub_issue,
          confidence: data.confidence,
          routedTo: data.routed_to || 'General Support',
          isReliable: (data.confidence?.overall || 0) >= this.threshold
        };
      }
    } catch (err) {
      logger.warn('CORA service unavailable or timed out, evaluating fallback', { error: err.message });
    }

    if (this.fallbackEnabled) {
      return this.fallbackClassifier(text);
    }

    throw new Error('CORA AI service is offline and mock fallback is disabled.');
  }

  fallbackClassifier(text) {
    const t = text.toLowerCase();
    let product = 'Checking or savings account';
    let subProduct = 'Checking account';
    let issue = 'Managing an account';
    let subIssue = 'Deposits and withdrawals';
    let confidence = 0.72;

    if (t.includes('card') || t.includes('charge') || t.includes('credit') || t.includes('cvv') || t.includes('transaction')) {
      product = 'Credit card or prepaid card';
      subProduct = 'General-purpose credit card';
      issue = 'Problem with a purchase shown on your statement';
      subIssue = 'Unauthorized charge';
      confidence = 0.88;
    } else if (t.includes('mortgage') || t.includes('foreclosure') || t.includes('escrow') || t.includes('home loan')) {
      product = 'Mortgage';
      subProduct = 'Conventional home mortgage';
      issue = 'Trouble during payment process';
      subIssue = 'Payment processing or escrow issue';
      confidence = 0.85;
    } else if (t.includes('transfer') || t.includes('wire') || t.includes('zelle') || t.includes('venmo') || t.includes('scam')) {
      product = 'Money transfer, virtual currency, or money service';
      subProduct = 'Domestic (US) money transfer';
      issue = 'Fraud or scam';
      subIssue = 'Unauthorized or disputed money transfer';
      confidence = 0.82;
    } else if (t.includes('loan') || t.includes('interest rate') || t.includes('emi') || t.includes('collateral')) {
      product = 'Personal loan';
      subProduct = 'Installment loan';
      issue = 'Charged fees or interest you did not agree to';
      subIssue = 'Calculation or agreement dispute';
      confidence = 0.79;
    }

    return {
      source: 'cora-rule-fallback',
      product,
      subProduct,
      issue,
      subIssue,
      confidence: {
        product: confidence,
        sub_product: Math.max(0.5, confidence - 0.05),
        issue: Math.max(0.5, confidence - 0.08),
        sub_issue: Math.max(0.5, confidence - 0.12),
        overall: confidence
      },
      routedTo: product === 'Credit card or prepaid card' ? 'Card Dispute Division' : 'Account Services Team',
      isReliable: confidence >= this.threshold
    };
  }
}

module.exports = new CoraService();
