# 🎯 Security Camera LITE API

API simplificada de detección de objetos con YOLOv8. Optimizada para RapidAPI.

## 🚀 Características

✅ **Detección rápida**: YOLOv8 nano (optimizado para velocidad)  
✅ **80 clases**: Personas, vehículos, animales, objetos comunes  
✅ **API REST simple**: Un solo endpoint principal  
✅ **Sin dependencias pesadas**: No DB, no streaming, solo detección  
✅ **Documentación Swagger**: `/docs` interactiva  
✅ **Listo para RapidAPI**: Estructura optimizada para marketplace  

## 📦 Instalación Local

```bash
# Clonar/descargar el proyecto
cd security-camera-lite

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Copiar configuración
cp .env.example .env

# Iniciar servidor
python main.py
```

El servidor iniciará en `http://localhost:8000`

## 🎯 Endpoints

### `POST /detect`
Detectar objetos en una imagen.

**Parámetros:**
- `file`: Imagen (JPG, PNG, etc.) - **Requerido**
- `confidence`: Umbral de confianza (0.1-1.0) - Opcional, default 0.5

**Ejemplo:**
```bash
curl -X POST "http://localhost:8000/detect?confidence=0.6" \
  -F "file=@imagen.jpg"
```

**Respuesta:**
```json
{
  "success": true,
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "detections_count": 3,
  "detections": [
    {
      "class": "person",
      "confidence": 0.92,
      "bbox": {
        "x1": 100.5,
        "y1": 200.3,
        "x2": 300.8,
        "y2": 600.2
      }
    },
    {
      "class": "car",
      "confidence": 0.87,
      "bbox": {
        "x1": 500.1,
        "y1": 300.5,
        "x2": 800.9,
        "y2": 500.7
      }
    }
  ]
}
```

### `GET /classes`
Obtener lista de clases detectables.

**Respuesta:**
```json
{
  "total_classes": 80,
  "classes": [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    ...
  ]
}
```

### `GET /health`
Health check del servicio.

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "yolov8n.pt",
  "version": "1.0.0"
}
```

## 📊 Clases Detectables (80 total)

### Personas y Animales
person, dog, cat, horse, sheep, cow, elephant, bear, zebra, giraffe

### Vehículos
car, motorcycle, airplane, bus, train, truck, boat, bicycle

### Objetos del Hogar
chair, couch, bed, dining table, toilet, tv, laptop, mouse, keyboard, cell phone

### Comida
banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

### Deportes
frisbee, skis, snowboard, sports ball, kite, baseball bat, skateboard, tennis racket

...y 50+ clases más! Ver endpoint `/classes` para lista completa.

## ⚡ Performance

- **Velocidad**: ~100-300ms por imagen (depende del tamaño)
- **Precisión**: 90%+ en condiciones normales
- **Límite de imagen**: 10MB máximo
- **Formatos**: JPG, PNG, BMP, WEBP, etc.

## 🔧 Configuración

Editar `.env`:

```env
# Modelo YOLO (n=nano, s=small, m=medium, l=large, x=xlarge)
YOLO_MODEL=yolov8n.pt

# Umbral de confianza por defecto (0-1)
CONFIDENCE_THRESHOLD=0.5

# Tamaño máximo de imagen (ancho o alto)
MAX_IMAGE_SIZE=1280

# Rate limiting (requests por minuto)
MAX_REQUESTS_PER_MINUTE=60
```

## 📖 Documentación Interactiva

Una vez iniciado el servidor, visita:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🚀 Deploy

### Docker (Recomendado)

```bash
# Crear imagen
docker build -t security-camera-lite .

# Correr
docker run -p 8000:8000 security-camera-lite
```

### Cloud (Railway, Render, etc.)

1. Hacer push a GitHub
2. Conectar con plataforma
3. Configurar:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python main.py`
4. Deploy!

## 🛠️ Troubleshooting

**Error: "Model not loaded"**
- Primera vez descarga el modelo (~6MB)
- Espera 30-60 segundos

**Error: "Image too large"**
- Máximo 10MB
- Reducir tamaño de imagen

**Detecciones incorrectas**
- Aumentar `confidence` (ej: 0.7)
- Usar mejor modelo: `yolov8s.pt`

## 📝 Ejemplos de Código

### Python
```python
import requests

url = "http://localhost:8000/detect"
files = {"file": open("imagen.jpg", "rb")}
params = {"confidence": 0.6}

response = requests.post(url, files=files, params=params)
print(response.json())
```

### JavaScript
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/detect?confidence=0.6', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => console.log(data));
```

### cURL
```bash
curl -X POST "http://localhost:8000/detect?confidence=0.6" \
  -F "file=@imagen.jpg" \
  -H "accept: application/json"
```

## 📄 Licencia

MIT License - Uso libre comercial y personal

## 🔗 Links

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [RapidAPI Marketplace](https://rapidapi.com/)

## 🆘 Soporte

- GitHub Issues
- Email: support@tu-dominio.com
- Discord: Tu servidor

---

**Versión LITE** - Optimizada para velocidad y simplicidad  
**Versión PRO** - Sistema completo con streaming, facial recognition, alertas, etc.
