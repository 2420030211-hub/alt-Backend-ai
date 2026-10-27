'use strict';

const VALID_STATUSES = ['Submitted', 'In Progress', 'Awaiting Customer', 'Resolved', 'Closed'];

const ALLOWED_TRANSITIONS = {
  'Submitted': ['In Progress', 'Closed'],
  'In Progress': ['Awaiting Customer', 'Resolved', 'Closed'],
  'Awaiting Customer': ['In Progress', 'Resolved', 'Closed'],
  'Resolved': ['Closed', 'In Progress'],
  'Closed': [] // Terminal state, requires supervisor intervention to reopen
};

const VALID_HIERARCHY = {
  'Checking or savings account': {
    subProducts: ['Checking account', 'Savings account', 'Other banking account'],
    issues: ['Managing an account', 'Problem with a lender or other company', 'Deposits and withdrawals']
  },
  'Credit card or prepaid card': {
    subProducts: ['General-purpose credit card', 'Store credit card', 'Prepaid card'],
    issues: ['Problem with a purchase shown on your statement', 'Fees or interest', 'Billing problem']
  },
  'Mortgage': {
    subProducts: ['Conventional home mortgage', 'FHA mortgage', 'VA mortgage'],
    issues: ['Applying for a mortgage', 'Struggling to pay mortgage', 'Trouble during payment process']
  },
  'Personal loan': {
    subProducts: ['Installment loan', 'Pawn loan', 'Personal line of credit'],
    issues: ['Problem with credit report or credit score', 'Charged fees or interest you did not agree to']
  },
  'Money transfer, virtual currency, or money service': {
    subProducts: ['Domestic (US) money transfer', 'International money transfer', 'Mobile or digital wallet'],
    issues: ['Fraud or scam', 'Money was not available when promised', 'Incorrect details on statement']
  }
};

function canTransitionStatus(currentStatus, targetStatus) {
  if (!VALID_STATUSES.includes(targetStatus)) return false;
  if (currentStatus === targetStatus) return true;
  const allowed = ALLOWED_TRANSITIONS[currentStatus] || [];
  return allowed.includes(targetStatus);
}

function validateHierarchy(product, subProduct, issue, subIssue) {
  // If product is known, check against rules
  if (product && VALID_HIERARCHY[product]) {
    const pData = VALID_HIERARCHY[product];
    const isSubProductValid = !subProduct || pData.subProducts.includes(subProduct);
    return {
      isValid: isSubProductValid,
      reason: isSubProductValid ? null : `Sub-product "${subProduct}" is not standard under product "${product}"`
    };
  }
  return { isValid: true, reason: null };
}

function sanitizeText(text) {
  if (!text) return '';
  return text.trim().replace(/[\u0000-\u0008\u000B-\u000C\u000E-\u001F]/g, '');
}

module.exports = {
  VALID_STATUSES,
  ALLOWED_TRANSITIONS,
  VALID_HIERARCHY,
  canTransitionStatus,
  validateHierarchy,
  sanitizeText
};
