import requests

# URL de tu API local
url = "http://localhost:8000/detect"

# Usar cualquier imagen JPG que tengas
imagen = "C:\\Users\\imart\\Desktop\\test.jpg"  # Cambia esta ruta

with open(imagen, "rb") as f:
    response = requests.post(
        url,
        files={"file": f},
        params={"confidence": 0.5}
    )

data = response.json()
print(f"\n✅ Detecciones: {data['detections_count']}")
print(f"⚡ Tiempo: {data.get('inference_ms', 'N/A')}ms")
print(f"🤖 Modelo: {data.get('model', 'N/A')}")
print(f"\n📊 Objetos encontrados:")
for det in data["detections"]:
    print(f"   {det['class']}: {det['confidence']:.1%}")
