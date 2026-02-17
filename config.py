"""
Configuración Security Camera LITE
Upgrade: YOLOv11n + ONNX Runtime
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración de la aplicación LITE"""

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    # YOLO - Upgrade a v11 + ONNX
    YOLO_MODEL: str = "yolo11n.pt"         # YOLOv11 nano
    YOLO_MODEL_ONNX: str = "yolo11n.onnx"  # Versión ONNX optimizada
    USE_ONNX: bool = True                   # True = más rápido
    CONFIDENCE_THRESHOLD: float = 0.5
    MAX_IMAGE_SIZE: int = 1280

    # Rate limiting
    MAX_REQUESTS_PER_MINUTE: int = 60

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
