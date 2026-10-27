'use strict';
const config = require('../config/env');
const { logger } = require('../utils/helpers');

class QwenService {
  constructor() {
    this.baseUrl = config.aiServices.qwenUrl;
    this.fallbackEnabled = config.aiServices.mockFallback;
  }

  async generateReply({ message, conversationId, history = [], stream = false }) {
    if (!message || typeof message !== 'string') {
      throw new Error('Message is required');
    }

    // Try calling Qwen FastAPI microservice
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 8000);

      const response = await fetch(`${this.baseUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          conversation_id: conversationId,
          history,
          stream
        }),
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        return {
          source: 'qwen-fastapi',
          reply: data.reply,
          conversationId: data.conversation_id || conversationId || `conv-${Date.now()}`,
          model: data.model || 'Qwen/Qwen3-4B'
        };
      }
    } catch (err) {
      logger.warn('Qwen service offline or timed out, using fallback rule assistant', { error: err.message });
    }

    if (this.fallbackEnabled) {
      return this.fallbackAssistant(message, conversationId);
    }

    throw new Error('Qwen conversational service is currently offline and fallback is disabled.');
  }

  fallbackAssistant(message, conversationId) {
    const text = message.toLowerCase();
    let reply = "Hello! I am your Banking Virtual Assistant. How can I assist you with your accounts, transactions, or complaint filing today?";

    if (text.includes('password') || text.includes('pin') || text.includes('otp')) {
      reply = "Security Alert: For your protection, our bank employees and virtual assistants will NEVER ask for your password, PIN, or OTP. Please keep these confidential.";
    } else if (text.includes('status') || text.includes('track') || text.includes('cmp-')) {
      reply = "To track an existing complaint, please provide your Ticket ID (e.g., CMP-2026-XXXXXX) or visit the 'Complaints' tab in your Customer Dashboard.";
    } else if (text.includes('fraud') || text.includes('unauthorized') || text.includes('stolen')) {
      reply = "I understand you have reported an urgent security or fraud matter. If your card is lost or stolen, please freeze it immediately from the portal or file a priority complaint below.";
    } else if (text.includes('complaint') || text.includes('issue') || text.includes('charge') || text.includes('fee')) {
      reply = "I can help you lodge an official complaint. Please summarize the problem, the affected account or card, transaction date, and the disputed amount. Would you like to submit this now?";
    }

    return {
      source: 'qwen-rule-fallback',
      reply,
      conversationId: conversationId || `conv-${Date.now()}`,
      model: 'Qwen/Qwen3-4B (Pretrained baseline fallback)'
    };
  }
}

module.exports = new QwenService();
