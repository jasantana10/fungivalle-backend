from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
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
MODELO_PATH_KERAS = MODELOS_DIR / "modelo_finetuned.keras"
MODELO_PATH_TFLITE = MODELOS_DIR / "modelo_finetuned.tflite"
CLASS_PATH = MODELOS_DIR / "class_names.json"

# VARIABLE GLOBAL PARA EL MODELO
model = None
interpreter = None  # Para TFLite
class_names = None
model_type = None   # 'keras' o 'tflite'

import threading

def load_model_on_startup():
    def load():
        global model, interpreter, class_names, model_type
        print("🔄 Cargando modelo de hongos (en segundo plano)...")
        
        # 1. Intentar cargar TFLite (Prioridad por memoria)
        if MODELO_PATH_TFLITE.exists():
            try:
                import tflite_runtime.interpreter as tflite
                interpreter = tflite.Interpreter(model_path=str(MODELO_PATH_TFLITE))
                interpreter.allocate_tensors()
                model_type = 'tflite'
                print(f"✅ Modelo TFLITE cargado exitosamente: {MODELO_PATH_TFLITE}")
            except Exception as e:
                print(f"⚠️ Error cargando TFLite, intentando Keras: {e}")
        
        # 2. Si no hay TFLite o falló, intentar Keras (Solo si TFLite falló y el archivo existe)
        if model_type is None and MODELO_PATH_KERAS.exists():
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(MODELO_PATH_KERAS)
                model_type = 'keras'
                print(f"✅ Modelo KERAS cargado exitosamente: {MODELO_PATH_KERAS}")
            except Exception as e:
                print(f"❌ Error fatal cargando Keras: {str(e)}")
                # import traceback
                # traceback.print_exc()

        if model_type is None:
            print(f"❌ ERROR: No se encontró ningún modelo en {MODELOS_DIR}")
            if MODELOS_DIR.exists():
                print(f"Contenido de {MODELOS_DIR}: {os.listdir(MODELOS_DIR)}")
        else:
            # Cargar clases solo si el modelo se cargó
            try:
                if CLASS_PATH.exists():
                    with open(CLASS_PATH, 'r', encoding='utf-8') as f:
                        class_names = json.load(f)
                    print(f"✅ Clases cargadas: {len(class_names)} especies")
                else:
                    print(f"❌ ERROR: El archivo de clases no existe en {CLASS_PATH}")
            except Exception as e:
                print(f"❌ Error cargando clases: {e}")
    
    # Ejecutar en un hilo separado para no bloquear el inicio del servidor
    thread = threading.Thread(target=load)
    thread.daemon = True
    thread.start()

# Ejecutar carga al importar (ahora es asíncrona)
load_model_on_startup()

def preprocesar_imagen(contents):
    """Preprocesa imagen para el modelo"""
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

@router.post("/identificar")
async def identificar_hongo(file: UploadFile = File(...)):
    """
    Identifica un hongo a partir de una imagen
    """
    if model_type is None or not class_names:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    try:
        # Leer imagen
        contents = await file.read()
        
        # Preprocesar
        img_array, img_original = preprocesar_imagen(contents)
        
        # Predecir
        if model_type == 'tflite':
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        else:
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