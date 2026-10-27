from typing import Dict, Any
from .config import CoraConfig

def calibrate_confidence(scores: Dict[str, float], threshold: float = 0.65) -> Dict[str, Any]:
    overall = sum(scores.values()) / max(len(scores), 1)
    return {
        "scores": scores,
        "overall": round(overall, 4),
        "is_confident": overall >= threshold
    }
