import re
from typing import Tuple

SENSITIVE_PATTERNS = [
    r'\b(?:\d[ -]*?){13,16}\b', # Credit card number pattern
    r'\b\d{4}\b.*\b(?:pin|otp|password)\b',
    r'\b(?:cvv|cvc)\s*[:=]?\s*\d{3,4}\b'
]

def check_input_safety(text: str) -> Tuple[bool, str]:
    lower = text.lower()
    
    # 1. Credentials solicitation or disclosure
    if any(k in lower for k in ["my password is", "my pin is", "my otp is"]):
        return False, "Security Notice: Please never disclose your passwords, PINs, or OTP codes in this chat. For your protection, clear this sensitive data."

    # 2. Credit card CVV leak
    if re.search(r'\b(?:cvv|cvc)\s*[:=]?\s*\d{3,4}\b', lower):
        return False, "Security Notice: Please do not transmit credit card CVV/security codes over chat."

    return True, ""
