from pydantic_settings import BaseSettings

class QwenConfig(BaseSettings):
    service_name: str = "qwen-chatbot-service"
    port: int = 8002
    host: str = "0.0.0.0"
    model_name: str = "Qwen/Qwen3-4B"
    device: str = "cpu"
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9

    class Config:
        env_file = ".env"
        extra = "allow"

config = QwenConfig()
