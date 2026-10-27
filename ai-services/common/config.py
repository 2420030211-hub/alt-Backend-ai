from pydantic_settings import BaseSettings
from typing import Optional

class BaseServiceSettings(BaseSettings):
    service_name: str = "ai-service"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    mock_mode: bool = True
    device: str = "cpu"

    class Config:
        env_file = ".env"
        extra = "allow"
