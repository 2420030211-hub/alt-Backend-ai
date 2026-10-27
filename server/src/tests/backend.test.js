'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const storage = require('../services/storage.service');
const businessRules = require('../services/business-rules.service');
const coraService = require('../services/cora.service');
const qwenService = require('../services/qwen.service');

test('Business Rules: Status transitions follow allowed state machine', () => {
  assert.equal(businessRules.canTransitionStatus('Submitted', 'In Progress'), true);
  assert.equal(businessRules.canTransitionStatus('Submitted', 'Resolved'), false);
  assert.equal(businessRules.canTransitionStatus('In Progress', 'Resolved'), true);
  assert.equal(businessRules.canTransitionStatus('Resolved', 'In Progress'), true);
  assert.equal(businessRules.canTransitionStatus('Closed', 'In Progress'), false);
});

test('Business Rules: Hierarchy validation flags incompatible product/sub-product', () => {
  const valid = businessRules.validateHierarchy('Checking or savings account', 'Checking account');
  assert.equal(valid.isValid, true);

  const invalid = businessRules.validateHierarchy('Checking or savings account', 'Conventional home mortgage');
  assert.equal(invalid.isValid, false);
});

test('Storage Service: User and Session management', () => {
  const user = storage.createUser({
    email: 'tester@test.local',
    password: 'Password123!',
    fullName: 'Jane Tester'
  });
  assert.equal(user.email, 'tester@test.local');
  assert.equal(user.role, 'customer');

  const token = storage.createSession(user.id);
  assert.ok(token);

  const sessionUser = storage.getSession(token);
  assert.equal(sessionUser.id, user.id);
});

test('Storage Service: Demo Employee is pre-seeded', () => {
  const emp = storage.findEmployeeById('EMP1001');
  assert.ok(emp);
  assert.equal(emp.role, 'employee');
});

test('CORA Service: Rule-based fallback classifier returns compliant schema', async () => {
  const res = await coraService.classify('My credit card was charged twice by the vendor.');
  assert.ok(res.product);
  assert.ok(res.subProduct);
  assert.ok(res.issue);
  assert.ok(res.subIssue);
  assert.ok(typeof res.confidence.overall === 'number');
  assert.equal(res.product, 'Credit card or prepaid card');
});

test('Qwen Service: Generates safety-compliant baseline response', async () => {
  const res = await qwenService.generateReply({ message: 'What is the status of my ticket?' });
  assert.ok(res.reply);
  assert.ok(res.conversationId);
});

test('Qwen Service: Flags sensitive credential leak attempts', async () => {
  const res = await qwenService.generateReply({ message: 'Can I give you my secret pin?' });
  assert.match(res.reply, /NEVER ask for your password, PIN/);
});
