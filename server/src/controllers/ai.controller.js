'use strict';
const coraService = require('../services/cora.service');
const qwenService = require('../services/qwen.service');

class AiController {
  async classify(req, res, next) {
    try {
      const { text } = req.body;
      if (!text || typeof text !== 'string') {
        return res.status(400).json({ error: 'Field "text" is required.' });
      }

      const result = await coraService.classify(text);
      return res.json(result);
    } catch (err) {
      next(err);
    }
  }

  async chat(req, res, next) {
    try {
      const { message, conversationId, history, stream } = req.body;
      if (!message || typeof message !== 'string') {
        return res.status(400).json({ error: 'Field "message" is required.' });
      }

      // If SSE streaming is requested
      if (stream) {
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');

        const replyObj = await qwenService.generateReply({ message, conversationId, history });
        const words = replyObj.reply.split(' ');

        for (let i = 0; i < words.length; i++) {
          const chunk = words[i] + (i < words.length - 1 ? ' ' : '');
          res.write(`data: ${JSON.stringify({ token: chunk, conversationId: replyObj.conversationId })}\n\n`);
          // simulate slight streaming delay
          await new Promise(r => setTimeout(r, 20));
        }

        res.write('data: [DONE]\n\n');
        return res.end();
      }

      // Standard JSON response
      const result = await qwenService.generateReply({ message, conversationId, history, stream: false });
      return res.json({
        reply: result.reply,
        conversationId: result.conversationId,
        model: result.model,
        source: result.source
      });
    } catch (err) {
      next(err);
    }
  }
}

module.exports = new AiController();
