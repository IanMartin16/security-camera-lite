"""
Servicio de detección LITE
Upgrade: YOLOv11n + ONNX Runtime (40% más rápido que YOLOv8)
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Optional
from loguru import logger
from PIL import Image
import io
import os
import time

from config import settings


class DetectionService:
    """Servicio de detección con YOLOv11 + ONNX"""

    def __init__(self):
        self.model = None
        self.model_type = None  # 'pt' o 'onnx'
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        self._inference_times = []  # Para calcular promedio de velocidad

    def load_model(self):
        """
        Cargar modelo YOLOv11.
        Intenta ONNX primero (más rápido), fallback a .pt
        """
        if self.model is not None:
            return

        # Intentar cargar ONNX primero
        if settings.USE_ONNX:
            onnx_loaded = self._load_onnx()
            if onnx_loaded:
                return

        # Fallback: cargar .pt y exportar a ONNX
        self._load_pt_and_export()

    def _load_onnx(self) -> bool:
        """Cargar modelo ONNX si existe"""
        onnx_path = settings.YOLO_MODEL_ONNX

        if os.path.exists(onnx_path):
            try:
                logger.info(f"⚡ Cargando modelo ONNX: {onnx_path}")
                self.model = YOLO(onnx_path, task="detect")
                self.model_type = "onnx"
                logger.info("✅ Modelo ONNX cargado (modo ultra-rápido)")
                return True
            except Exception as e:
                logger.warning(f"⚠️  No se pudo cargar ONNX: {e}")
                return False

        logger.info("📦 Modelo ONNX no encontrado, generando...")
        return self._export_to_onnx()

    def _load_pt_and_export(self):
        """Cargar modelo .pt y exportar a ONNX"""
        try:
            logger.info(f"📦 Cargando modelo PyTorch: {settings.YOLO_MODEL}")
            pt_model = YOLO(settings.YOLO_MODEL)
            logger.info("✅ Modelo PyTorch cargado")

            if settings.USE_ONNX:
                # Exportar a ONNX para futuras ejecuciones
                logger.info("🔄 Exportando a ONNX para mayor velocidad...")
                try:
                    pt_model.export(
                        format="onnx",
                        imgsz=640,
                        simplify=True,
                        opset=12
                    )
                    logger.info(f"✅ Exportado a: {settings.YOLO_MODEL_ONNX}")

                    # Cargar versión ONNX
                    self.model = YOLO(settings.YOLO_MODEL_ONNX, task="detect")
                    self.model_type = "onnx"
                    logger.info("⚡ Usando modelo ONNX (modo ultra-rápido)")
                    return
                except Exception as e:
                    logger.warning(f"⚠️  Export ONNX falló, usando .pt: {e}")

            # Usar .pt directamente
            self.model = pt_model
            self.model_type = "pt"

        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise

    def _export_to_onnx(self) -> bool:
        """Exportar modelo .pt a ONNX"""
        try:
            logger.info(f"📦 Descargando y exportando {settings.YOLO_MODEL}...")
            pt_model = YOLO(settings.YOLO_MODEL)

            pt_model.export(
                format="onnx",
                imgsz=640,
                simplify=True,
                opset=12
            )

            if os.path.exists(settings.YOLO_MODEL_ONNX):
                self.model = YOLO(settings.YOLO_MODEL_ONNX, task="detect")
                self.model_type = "onnx"
                logger.info("⚡ Modelo ONNX listo")
                return True

        except Exception as e:
            logger.warning(f"Export falló: {e}")

        return False

    def detect_from_bytes(self, image_bytes: bytes, confidence: float = None) -> Dict[str, Any]:
        """
        Detectar objetos en imagen

        Args:
            image_bytes: Imagen en bytes
            confidence: Umbral de confianza (0.1-1.0)

        Returns:
            Detecciones con clase, confianza y bounding box
        """
        if self.model is None:
            self.load_model()

        conf_threshold = confidence if confidence is not None else self.confidence_threshold

        try:
            # Convertir bytes a imagen
            image = Image.open(io.BytesIO(image_bytes))

            # Convertir RGBA a RGB si es necesario
            if image.mode == "RGBA":
                image = image.convert("RGB")

            # Redimensionar si es muy grande
            if max(image.size) > settings.MAX_IMAGE_SIZE:
                ratio = settings.MAX_IMAGE_SIZE / max(image.size)
                new_size = tuple(int(d * ratio) for d in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            img_array = np.array(image)

            # Medir tiempo de inferencia
            start = time.time()
            results = self.model(img_array, conf=conf_threshold, verbose=False)
            inference_ms = (time.time() - start) * 1000

            # Guardar para estadísticas
            self._inference_times.append(inference_ms)
            if len(self._inference_times) > 100:
                self._inference_times.pop(0)

            # Procesar detecciones
            detections = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())

                    detections.append({
                        "class": class_name,
                        "confidence": round(conf, 4),
                        "bbox": {
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2)
                        }
                    })

            logger.info(f"✅ {len(detections)} objetos | {inference_ms:.1f}ms | {self.model_type.upper()}")

            return {
                "success": True,
                "image_size": {
                    "width": image.size[0],
                    "height": image.size[1]
                },
                "detections_count": len(detections),
                "detections": detections,
                "inference_ms": round(inference_ms, 2),
                "model": f"YOLOv11n-{self.model_type.upper()}"
            }

        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return {
                "success": False,
                "error": str(e),
                "detections_count": 0,
                "detections": []
            }

    def get_available_classes(self) -> List[str]:
        """Obtener clases detectables"""
        if self.model is None:
            self.load_model()
        return list(self.model.names.values())

    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas de rendimiento"""
        if not self._inference_times:
            return {"avg_inference_ms": 0, "model_type": self.model_type}

        return {
            "avg_inference_ms": round(sum(self._inference_times) / len(self._inference_times), 2),
            "min_inference_ms": round(min(self._inference_times), 2),
            "max_inference_ms": round(max(self._inference_times), 2),
            "total_inferences": len(self._inference_times),
            "model_type": self.model_type,
            "model_name": "YOLOv11n"
        }


# Instancia global
detection_service = DetectionService()
