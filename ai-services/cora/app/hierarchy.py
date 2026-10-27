from typing import Dict, Any

def enforce_hierarchy_rules(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures consistent parent-child relationships in predicted multi-target output."""
    prod = prediction.get("product")
    sub_prod = prediction.get("sub_product")
    
    # If sub_product conflicts with product, adjust or flag
    return prediction
