"""
Security Camera LITE API - Versión para RapidAPI
Simple, rápida, eficiente
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import sys
from contextlib import asynccontextmanager

from config import settings
from detection import detection_service
from schemas import (
    DetectionResponse,
    ErrorResponse,
    HealthResponse,
    ClassesResponse
)

# Configurar logging
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level=settings.LOG_LEVEL
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events"""
    # Startup
    logger.info("🚀 Iniciando Security Camera LITE API...")
    
    # Cargar modelo YOLO
    try:
        detection_service.load_model()
        logger.info("✅ API lista para recibir requests")
    except Exception as e:
        logger.error(f"❌ Error al cargar modelo: {e}")
        logger.warning("⚠️  API iniciada pero sin modelo cargado")
    
    yield
    
    # Shutdown
    logger.info("👋 Deteniendo API...")


# Crear aplicación
app = FastAPI(
    title="Security Camera LITE API",
    description="API de detección de objetos con IA usando YOLOv8. Versión optimizada para RapidAPI.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS (RapidAPI lo maneja, pero por si acaso)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "name": "Security Camera LITE API",
        "version": "1.0.0",
        "description": "Detección de objetos con YOLOv8",
        "endpoints": {
            "detect": "/detect - POST: Detectar objetos en imagen",
            "classes": "/classes - GET: Ver clases disponibles",
            "health": "/health - GET: Estado del servicio"
        },
        "documentation": "/docs",
        "model": settings.YOLO_MODEL
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check - Verificar estado del servicio
    
    Retorna el estado del servicio y si el modelo está cargado.
    """
    return {
        "status": "healthy" if detection_service.model is not None else "degraded",
        "model_loaded": detection_service.model is not None,
        "model_name": settings.YOLO_MODEL,
        "version": "1.0.0"
    }


@app.get("/classes", response_model=ClassesResponse, tags=["Detection"])
async def get_classes():
    """
    Obtener clases disponibles
    
    Retorna la lista de todas las clases de objetos que el modelo puede detectar.
    Son las 80 clases del dataset COCO.
    """
    try:
        classes = detection_service.get_available_classes()
        return {
            "total_classes": len(classes),
            "classes": sorted(classes)
        }
    except Exception as e:
        logger.error(f"Error obteniendo clases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_objects(
    file: UploadFile = File(..., description="Imagen a procesar (JPG, PNG, etc.)"),
    confidence: float = Query(
        default=None,
        ge=0.1,
        le=1.0,
        description="Umbral de confianza personalizado (0.1-1.0). Si no se especifica, usa 0.5"
    )
):
    """
    Detectar objetos en una imagen
    
    ## Parámetros:
    - **file**: Imagen a procesar (formatos: JPG, PNG, BMP, etc.)
    - **confidence** (opcional): Umbral de confianza (0.1-1.0). Default: 0.5
    
    ## Respuesta:
    - **success**: Si la detección fue exitosa
    - **image_size**: Dimensiones de la imagen procesada
    - **detections_count**: Número de objetos detectados
    - **detections**: Lista de objetos con clase, confianza y bounding box
    
    ## Ejemplo de uso:
    ```bash
    curl -X POST "http://localhost:8000/detect?confidence=0.6" \\
      -F "file=@mi_imagen.jpg"
    ```
    
    ## Clases detectables:
    80 clases del dataset COCO: personas, vehículos, animales, objetos comunes, etc.
    Ver endpoint `/classes` para lista completa.
    """
    
    # Validar tipo de archivo
    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Archivo debe ser una imagen. Recibido: {file.content_type}"
        )
    
    # Si no tiene content_type, validar por extensión
    if not file.content_type:
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif']
        if not any(file.filename.lower().endswith(ext) for ext in valid_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"Archivo debe ser una imagen (JPG, PNG, etc.). Recibido: {file.filename}"
            )
    
    try:
        # Leer imagen
        image_bytes = await file.read()
        
        # Validar tamaño (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(image_bytes) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Imagen muy grande. Máximo: 10MB. Recibido: {len(image_bytes) / 1024 / 1024:.2f}MB"
            )
        
        logger.info(f"📸 Procesando imagen: {file.filename} ({len(image_bytes) / 1024:.2f}KB)")
        
        # Detectar objetos
        result = detection_service.detect_from_bytes(image_bytes, confidence)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Error desconocido'))
        
        logger.info(f"✅ Detección exitosa: {result['detections_count']} objetos encontrados")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error procesando imagen: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejador global de excepciones"""
    logger.error(f"Error no manejado: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Error interno del servidor" if not settings.DEBUG else str(exc),
            "detections_count": 0,
            "detections": []
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🚀 Iniciando servidor en {settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
