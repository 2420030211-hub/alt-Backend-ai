import os
import joblib
from typing import Optional, Any
from .config import config

class ModelLoader:
    def __init__(self):
        self.model: Optional[Any] = None
        self.model_loaded: bool = False
        self.loaded_checkpoint_path: Optional[str] = None
        self.load_model()

    def load_model(self):
        # Look for trained model checkpoints in artifacts
        checkpoint_file = os.path.join(config.checkpoint_dir, "cora_best_model.joblib")
        if os.path.exists(checkpoint_file):
            try:
                self.model = joblib.load(checkpoint_file)
                self.model_loaded = True
                self.loaded_checkpoint_path = checkpoint_file
            except Exception as e:
                self.model = None
                self.model_loaded = False
        else:
            # Checkpoint not present yet: accurately reflect not loaded
            self.model = None
            self.model_loaded = False
            self.loaded_checkpoint_path = None

model_loader = ModelLoader()
