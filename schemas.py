"""
Schemas de Pydantic para validación
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class BoundingBox(BaseModel):
    """Bounding box coordenadas"""
    x1: float = Field(..., description="Coordenada X superior izquierda")
    y1: float = Field(..., description="Coordenada Y superior izquierda")
    x2: float = Field(..., description="Coordenada X inferior derecha")
    y2: float = Field(..., description="Coordenada Y inferior derecha")


class Detection(BaseModel):
    """Detección individual"""
    class_: str = Field(..., alias='class', description="Clase del objeto detectado")
    confidence: float = Field(..., ge=0, le=1, description="Confianza de la detección (0-1)")
    bbox: BoundingBox = Field(..., description="Bounding box del objeto")
    
    class Config:
        populate_by_name = True


class ImageSize(BaseModel):
    """Tamaño de imagen"""
    width: int = Field(..., description="Ancho de la imagen en pixels")
    height: int = Field(..., description="Alto de la imagen en pixels")


class DetectionResponse(BaseModel):
    """Respuesta de detección"""
    success: bool = Field(..., description="Si la detección fue exitosa")
    image_size: ImageSize = Field(..., description="Dimensiones de la imagen procesada")
    detections_count: int = Field(..., description="Número de objetos detectados")
    detections: List[Detection] = Field(..., description="Lista de detecciones")
    error: Optional[str] = Field(None, description="Mensaje de error si hubo fallo")


class ErrorResponse(BaseModel):
    """Respuesta de error"""
    success: bool = False
    error: str = Field(..., description="Descripción del error")
    detections_count: int = 0
    detections: List[Detection] = []


class HealthResponse(BaseModel):
    """Respuesta de health check"""
    status: str = Field(..., description="Estado del servicio")
    model_loaded: bool = Field(..., description="Si el modelo YOLO está cargado")
    model_name: str = Field(..., description="Nombre del modelo en uso")
    version: str = Field(..., description="Versión de la API")


class ClassesResponse(BaseModel):
    """Respuesta de clases disponibles"""
    total_classes: int = Field(..., description="Número total de clases")
    classes: List[str] = Field(..., description="Lista de nombres de clases")
