"""
Security Camera LITE API
Upgrade: YOLOv11n + ONNX Runtime (~40% más rápido que YOLOv8)
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import sys
import os
from contextlib import asynccontextmanager

from config import settings
from detection import detection_service
from schemas import DetectionResponse, HealthResponse, ClassesResponse


# Logging
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
    logger.info("🚀 Iniciando Security Camera LITE API...")
    logger.info("⚡ Engine: YOLOv11n + ONNX Runtime")

    try:
        detection_service.load_model()
        logger.info(f"✅ Modelo listo: YOLOv11n-{detection_service.model_type.upper()}")
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")

    yield

    logger.info("👋 Deteniendo API...")


# App
app = FastAPI(
    title="Security Camera LITE API",
    description="""
## 🎯 AI-Powered Object Detection API

Detect 80+ objects in images using **YOLOv11n + ONNX Runtime**.

### ⚡ Performance
- **~40% faster** than previous version
- **~100-200ms** response time
- **90%+ accuracy**

### 🎨 Use Cases
- Security & Surveillance
- Retail Analytics  
- Traffic Monitoring
- Content Moderation

### 📦 80 Detectable Classes
People, vehicles, animals, household items, and more!
    """,
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    return {
        "name": "Security Camera LITE API",
        "version": "2.0.0",
        "engine": "YOLOv11n + ONNX Runtime",
        "improvement": "~40% faster than v1.0",
        "endpoints": {
            "detect": "POST /detect",
            "classes": "GET /classes",
            "health": "GET /health",
            "stats": "GET /stats"
        },
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check - Estado del servicio"""
    return {
        "status": "healthy" if detection_service.model is not None else "degraded",
        "model_loaded": detection_service.model is not None,
        "model_name": f"YOLOv11n-{detection_service.model_type.upper() if detection_service.model_type else 'N/A'}",
        "version": "2.0.0"
    }


@app.get("/stats", tags=["General"])
async def get_stats():
    """
    Performance stats del modelo

    Muestra tiempos de inferencia promedio, mínimo y máximo.
    Útil para monitorear la performance en producción.
    """
    return detection_service.get_stats()


@app.get("/classes", response_model=ClassesResponse, tags=["Detection"])
async def get_classes():
    """
    Clases detectables - 80 objetos del dataset COCO

    Incluye personas, vehículos, animales, objetos del hogar y más.
    """
    try:
        classes = detection_service.get_available_classes()
        return {
            "total_classes": len(classes),
            "classes": sorted(classes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_objects(
    file: UploadFile = File(..., description="Imagen a procesar (JPG, PNG, etc.) - Max 10MB"),
    confidence: float = Query(
        default=None,
        ge=0.1,
        le=1.0,
        description="Umbral de confianza (0.1-1.0). Default: 0.5. Mayor = menos detecciones pero más precisas"
    )
):
    """
    ## Detectar objetos en imagen

    Sube una imagen y recibe todos los objetos detectados con:
    - **Clase** del objeto (person, car, dog, etc.)
    - **Confianza** de la detección (0-1)
    - **Bounding box** con coordenadas exactas

    ### Motor
    **YOLOv11n + ONNX Runtime** - ~40% más rápido que versiones anteriores

    ### Formatos soportados
    JPG, PNG, BMP, WEBP, GIF - Máximo **10MB**

    ### Ejemplo de uso
    ```python
    import requests
    files = {"file": open("imagen.jpg", "rb")}
    response = requests.post("/detect?confidence=0.6", files=files)
    print(response.json())
    ```
    """

    # Validar tipo de archivo
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Debe ser una imagen. Recibido: {file.content_type}"
        )

    if not file.content_type:
        valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"]
        if not any(file.filename.lower().endswith(ext) for ext in valid_ext):
            raise HTTPException(
                status_code=400,
                detail=f"Extensión no válida: {file.filename}"
            )

    try:
        # Leer imagen
        image_bytes = await file.read()

        # Validar tamaño (max 10MB)
        max_size = 10 * 1024 * 1024
        if len(image_bytes) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Imagen muy grande: {len(image_bytes)/1024/1024:.1f}MB. Máximo: 10MB"
            )

        logger.info(f"📸 {file.filename} ({len(image_bytes)/1024:.1f}KB)")

        # Detectar
        result = detection_service.detect_from_bytes(image_bytes, confidence)

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error"))

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Error no manejado: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Error interno" if not settings.DEBUG else str(exc),
            "detections_count": 0,
            "detections": []
        }
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", settings.API_PORT))
    logger.info(f"🚀 Iniciando en {settings.API_HOST}:{port}")
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=port,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
