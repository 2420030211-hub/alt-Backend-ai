from typing import Optional, Any
from .config import config

class QwenModelLoader:
    def __init__(self):
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.model_loaded: bool = False
        # In baseline development mode without 8GB+ VRAM or prior download,
        # we accurately register model_loaded as False until explicitly loaded.

    def load_model(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            print(f"Loading {config.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self.model_loaded = True
        except Exception as e:
            print(f"Qwen model loading skipped or failed: {e}")
            self.model_loaded = False

qwen_loader = QwenModelLoader()
