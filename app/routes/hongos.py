from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
from pathlib import Path
import uuid
from datetime import datetime

router = APIRouter(prefix="/hongos", tags=["hongos"])

# Configurar rutas
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELOS_DIR = BASE_DIR / "app" / "ml" / "models"
UPLOAD_DIR = BASE_DIR / "uploads" / "hongos"

# Crear directorios
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Cargar modelo al iniciar
MODELO_PATH = MODELOS_DIR / "modelo_finetuned.keras"
CLASS_PATH = MODELOS_DIR / "class_names.json"

# VARIABLE GLOBAL PARA EL MODELO
model = None
class_names = None

def load_model_on_startup():
    global model, class_names
    print("🔄 Cargando modelo de hongos...")
    
    # Verificar si existen los archivos
    if not MODELO_PATH.exists():
        print(f"❌ ERROR: El archivo del modelo no existe en {MODELO_PATH}")
        # Listar contenido para depuración
        if MODELOS_DIR.exists():
            print(f"Contenido de {MODELOS_DIR}: {os.listdir(MODELOS_DIR)}")
        return

    try:
        model = tf.keras.models.load_model(MODELO_PATH)
        print(f"✅ Modelo cargado exitosamente: {MODELO_PATH}")
    except Exception as e:
        print(f"❌ Error fatal cargando modelo: {str(e)}")
        import traceback
        traceback.print_exc()

    try:
        if CLASS_PATH.exists():
            with open(CLASS_PATH, 'r', encoding='utf-8') as f:
                class_names = json.load(f)
            print(f"✅ Clases cargadas: {len(class_names)} especies")
        else:
            print(f"❌ ERROR: El archivo de clases no existe en {CLASS_PATH}")
    except Exception as e:
        print(f"❌ Error cargando clases: {e}")

# Ejecutar carga al importar (esto se puede mejorar con lifespan en FastAPI)
load_model_on_startup()

def preprocesar_imagen(contents):
    """Preprocesa imagen para el modelo"""
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

@router.post("/identificar")
async def identificar_hongo(file: UploadFile = File(...)):
    """
    Identifica un hongo a partir de una imagen
    """
    if not model or not class_names:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    try:
        # Leer imagen
        contents = await file.read()
        
        # Preprocesar
        img_array, img_original = preprocesar_imagen(contents)
        
        # Predecir
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Mejor predicción
        idx = np.argmax(predictions)
        confianza = float(predictions[idx])
        especie = class_names[str(idx)]
        
        # Guardar imagen
        filename = f"{uuid.uuid4()}.jpg"
        filepath = UPLOAD_DIR / filename
        img_original.save(filepath)
        
        # Top 3 predicciones
        top_3_idx = np.argsort(predictions)[-3:][::-1]
        sugerencias = [
            {
                "especie": class_names[str(tidx)],
                "confianza": float(predictions[tidx])
            }
            for tidx in top_3_idx
        ]
        
        return {
            "success": True,
            "especie": especie,
            "confianza": confianza,
            "confianza_porcentaje": f"{confianza*100:.2f}%",
            "sugerencias": sugerencias,
            "imagen_url": f"/uploads/hongos/{filename}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/especies")
async def listar_especies():
    """
    Lista todas las especies que puede identificar el modelo
    """
    if not class_names:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    especies = []
    for idx, nombre in class_names.items():
        especies.append({
            "id": int(idx),
            "nombre_cientifico": nombre,
            "nombre_comun": nombre.replace("_", " ").title()
        })
    
    return {"especies": especies}

@router.get("/estado")
async def estado_modelo():
    """
    Verifica el estado del modelo
    """
    return {
        "modelo_cargado": model is not None,
        "especies_capacidad": len(class_names) if class_names else 0,
        "modelo_path": str(MODELO_PATH),
        "clases_path": str(CLASS_PATH)
    }