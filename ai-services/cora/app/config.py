import os
from pydantic_settings import BaseSettings

class CoraConfig(BaseSettings):
    service_name: str = "cora-classification-service"
    port: int = 8001
    host: str = "0.0.0.0"
    model_type: str = "tfidf_svm" # 'tfidf_svm', 'bert-base-uncased', 'modernbert'
    model_name_or_path: str = "google-bert/bert-base-uncased"
    checkpoint_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../artifacts/cora"))
    confidence_threshold: float = 0.65
    device: str = "cpu"

    class Config:
        env_file = ".env"
        extra = "allow"

config = CoraConfig()
