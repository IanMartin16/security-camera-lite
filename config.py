"""
Configuración simple para Security Camera LITE
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración de la aplicación LITE"""
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    
    # YOLO
    YOLO_MODEL: str = "yolov8n.pt"  # Modelo nano (más rápido)
    CONFIDENCE_THRESHOLD: float = 0.5
    MAX_IMAGE_SIZE: int = 1280  # Máximo ancho/alto
    
    # Rate limiting (ajustar según tier de RapidAPI)
    MAX_REQUESTS_PER_MINUTE: int = 60
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
