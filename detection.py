"""
Servicio de detección LITE - Solo procesamiento de imágenes
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any
from loguru import logger
from PIL import Image
import io

from config import settings


class DetectionService:
    """Servicio simplificado de detección"""
    
    def __init__(self):
        self.model = None
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
        
    def load_model(self):
        """Cargar modelo YOLO"""
        if self.model is None:
            try:
                logger.info(f"Cargando modelo YOLO: {settings.YOLO_MODEL}")
                self.model = YOLO(settings.YOLO_MODEL)
                logger.info("✅ Modelo YOLO cargado exitosamente")
            except Exception as e:
                logger.error(f"❌ Error al cargar modelo YOLO: {e}")
                raise
    
    def detect_from_bytes(self, image_bytes: bytes, confidence: float = None) -> Dict[str, Any]:
        """
        Detectar objetos en imagen desde bytes
        
        Args:
            image_bytes: Imagen en bytes
            confidence: Umbral de confianza (opcional, usa default si no se especifica)
            
        Returns:
            Diccionario con detecciones
        """
        if self.model is None:
            self.load_model()
        
        # Usar confianza especificada o default
        conf_threshold = confidence if confidence is not None else self.confidence_threshold
        
        try:
            # Convertir bytes a imagen PIL
            image = Image.open(io.BytesIO(image_bytes))
            
            # Redimensionar si es muy grande (optimización)
            max_size = settings.MAX_IMAGE_SIZE
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convertir a numpy array
            img_array = np.array(image)
            
            # Si es RGBA, convertir a RGB
            if img_array.shape[-1] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            # Realizar detección
            results = self.model(img_array, conf=conf_threshold, verbose=False)
            
            # Procesar resultados
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Obtener datos
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(float, bbox)
                    
                    detection = {
                        'class': class_name,
                        'confidence': round(confidence, 4),
                        'bbox': {
                            'x1': round(x1, 2),
                            'y1': round(y1, 2),
                            'x2': round(x2, 2),
                            'y2': round(y2, 2)
                        }
                    }
                    
                    detections.append(detection)
            
            # Preparar respuesta
            response = {
                'success': True,
                'image_size': {
                    'width': image.size[0],
                    'height': image.size[1]
                },
                'detections_count': len(detections),
                'detections': detections
            }
            
            logger.info(f"✅ Detectados {len(detections)} objetos")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error en detección: {e}")
            return {
                'success': False,
                'error': str(e),
                'detections_count': 0,
                'detections': []
            }
    
    def get_available_classes(self) -> List[str]:
        """Obtener lista de clases disponibles"""
        if self.model is None:
            self.load_model()
        
        return list(self.model.names.values())


# Instancia global
detection_service = DetectionService()
