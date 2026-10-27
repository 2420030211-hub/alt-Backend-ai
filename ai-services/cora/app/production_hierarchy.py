"""Dataset-derived hierarchy validation for Cora predictions."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DEFAULT_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "production_hierarchy.json"
)


class ProductionHierarchy:
    def __init__(self, artifact_path: Optional[str] = None):
        path = Path(artifact_path) if artifact_path else DEFAULT_ARTIFACT
        if not path.is_file():
            raise FileNotFoundError(
                f"Production hierarchy artifact not found: {path}"
            )

        self.artifact_path = path
        self.data = json.loads(path.read_text(encoding="utf-8"))
        self.valid_paths = {
            tuple(path) for path in self.data["valid_full_paths"]
        }

    def validate(
        self,
        product: str,
        sub_product: str,
        issue: str,
        sub_issue: str,
    ) -> bool:
        """True only when the complete path appeared in training."""
        return (
            product,
            sub_product,
            issue,
            sub_issue,
        ) in self.valid_paths

    def inspect(
        self,
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return validation status without silently changing predictions."""
        path: Tuple[str, ...] = (
            str(prediction.get("product", "")),
            str(prediction.get("sub_product", "")),
            str(prediction.get("issue", "")),
            str(prediction.get("sub_issue", "")),
        )
        return {
            "hierarchy_valid": path in self.valid_paths,
            "hierarchy_status": (
                "observed_in_training"
                if path in self.valid_paths
                else "unseen_path"
            ),
        }
