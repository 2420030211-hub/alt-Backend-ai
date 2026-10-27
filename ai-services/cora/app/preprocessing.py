import re

def clean_complaint_text(text: str) -> str:
    """Standardize and clean complaint narrative without removing financial context."""
    if not text or not isinstance(text, str):
        return ""
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    # Anonymize typical CFPB redacted tokens like XXXX while preserving word boundary
    text = re.sub(r'X{2,}', '[REDACTED]', text)
    return text.strip()
