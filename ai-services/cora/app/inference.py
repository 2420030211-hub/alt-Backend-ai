from typing import Dict, Any

from .preprocessing import clean_complaint_text
from .label_mapping import label_registry
from .model_loader import model_loader
from .production_hierarchy import ProductionHierarchy

production_hierarchy = ProductionHierarchy()
from .confidence import calibrate_confidence


def run_inference(text: str) -> Dict[str, Any]:
    cleaned = clean_complaint_text(text)

    if model_loader.model_loaded and model_loader.model is not None:
        try:
            preds, scores = model_loader.predict(cleaned)

            # Keep confidence fields compatible with the existing API schema.
            calibrated = calibrate_confidence(
                scores,
                threshold=0.65,
            )

            product = preds["product"]
            sub_product = preds["sub_product"]
            issue = preds["issue"]
            sub_issue = preds["sub_issue"]

            return {
                "product": product,
                "sub_product": sub_product,
                "issue": issue,
                "sub_issue": sub_issue,
                "confidence": {
                    "product": scores["product"],
                    "sub_product": scores["sub_product"],
                    "issue": scores["issue"],
                    "sub_issue": scores["sub_issue"],
                    "overall": calibrated["overall"],
                },
                "routed_to": label_registry.get_department(product),
                # The old hard-coded registry is incomplete for the trained
                # dataset, so do not claim the full hierarchy was validated.
                "hierarchy_valid": production_hierarchy.validate(
                    product, sub_product, issue, sub_issue
                ),
                "model_version": "bert-base-uncased-hierarchical-v1",
            }
        except Exception as exc:
            print(f"BERT inference failed; using fallback: {exc}")

    # Existing rule-based fallback
    t = cleaned.lower()
    product = "Checking or savings account"
    sub_product = "Checking account"
    issue = "Managing an account"
    sub_issue = "Deposits and withdrawals"
    conf = 0.72

    if any(k in t for k in ["card", "charge", "credit", "cvv", "atm"]):
        product = "Credit card or prepaid card"
        sub_product = "General-purpose credit card"
        issue = "Problem with a purchase shown on your statement"
        sub_issue = "Unauthorized charge"
        conf = 0.88
    elif any(k in t for k in ["mortgage", "foreclosure", "escrow", "home loan"]):
        product = "Mortgage"
        sub_product = "Conventional home mortgage"
        issue = "Trouble during payment process"
        sub_issue = "Escrow or payment calculation dispute"
        conf = 0.85
    elif any(k in t for k in ["transfer", "wire", "zelle", "scam"]):
        product = "Money transfer, virtual currency, or money service"
        sub_product = "Domestic (US) money transfer"
        issue = "Fraud or scam"
        sub_issue = "Disputed transfer"
        conf = 0.81
    elif any(k in t for k in ["loan", "interest", "emi"]):
        product = "Personal loan"
        sub_product = "Installment loan"
        issue = "Charged fees or interest you did not agree to"
        sub_issue = "Rate dispute"
        conf = 0.78

    scores = {
        "product": conf,
        "sub_product": max(0.5, conf - 0.04),
        "issue": max(0.5, conf - 0.08),
        "sub_issue": max(0.5, conf - 0.12),
    }

    return {
        "product": product,
        "sub_product": sub_product,
        "issue": issue,
        "sub_issue": sub_issue,
        "confidence": {
            **scores,
            "overall": round(sum(scores.values()) / len(scores), 4),
        },
        "routed_to": label_registry.get_department(product),
        "hierarchy_valid": production_hierarchy.validate(
            product, sub_product, issue, sub_issue
        ),
        "model_version": (
            "bert-base-uncased-hierarchical-v1-fallback"
            if not model_loader.model_loaded
            else "bert-base-uncased-hierarchical-v1-fallback-after-error"
        ),
    }
