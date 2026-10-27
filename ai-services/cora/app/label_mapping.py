import json
import os
from typing import Dict, List, Any

DEFAULT_HIERARCHY = {
    "Checking or savings account": {
        "sub_products": ["Checking account", "Savings account", "Other banking account"],
        "issues": ["Managing an account", "Problem with a lender or other company", "Deposits and withdrawals"],
        "department": "Retail Deposit Operations"
    },
    "Credit card or prepaid card": {
        "sub_products": ["General-purpose credit card", "Store credit card", "Prepaid card"],
        "issues": ["Problem with a purchase shown on your statement", "Fees or interest", "Billing problem"],
        "department": "Card Disputes & Servicing"
    },
    "Mortgage": {
        "sub_products": ["Conventional home mortgage", "FHA mortgage", "VA mortgage"],
        "issues": ["Applying for a mortgage", "Struggling to pay mortgage", "Trouble during payment process"],
        "department": "Home Lending Division"
    },
    "Money transfer, virtual currency, or money service": {
        "sub_products": ["Domestic (US) money transfer", "International money transfer", "Mobile or digital wallet"],
        "issues": ["Fraud or scam", "Money was not available when promised", "Incorrect details on statement"],
        "department": "Payments & Wire Disputes"
    },
    "Personal loan": {
        "sub_products": ["Installment loan", "Pawn loan", "Personal line of credit"],
        "issues": ["Problem with credit report or credit score", "Charged fees or interest you did not agree to"],
        "department": "Consumer Credit Operations"
    }
}

class LabelRegistry:
    def __init__(self, mapping_file: str = None):
        self.hierarchy = DEFAULT_HIERARCHY
        if mapping_file and os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self.hierarchy = json.load(f)

    def validate_hierarchy(self, product: str, sub_product: str, issue: str, sub_issue: str) -> bool:
        if product in self.hierarchy:
            valid_subs = self.hierarchy[product].get("sub_products", [])
            if sub_product and sub_product not in valid_subs:
                return False
        return True

    def get_department(self, product: str) -> str:
        return self.hierarchy.get(product, {}).get("department", "General Banking Resolution")

label_registry = LabelRegistry()
