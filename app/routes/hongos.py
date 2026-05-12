from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
import numpy as np
from PIL import Image
import io
import json
import os
from pathlib import Path
import uuid
from datetime import datetime
import threading
import joblib

router = APIRouter(prefix="/hongos", tags=["hongos"])

# Configurar rutas
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELOS_DIR = BASE_DIR / "app" / "ml" / "models"
UPLOAD_DIR = BASE_DIR / "uploads" / "hongos"

# Crear directorios
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Cargar modelo al iniciar
MODELO_PATH_SKLEARN = MODELOS_DIR / "modelo_sklearn.pkl"
SCALER_PATH = MODELOS_DIR / "scaler.pkl"
CLASS_PATH = MODELOS_DIR / "class_names.json"

# VARIABLE GLOBAL PARA EL MODELO
sklearn_model = None
scaler = None
class_names = None
model_ready = False

# MobileNetV2 para extracción de características (solo se carga una vez, sin pesos entrenables)
feature_extractor = None

def load_feature_extractor():
    global feature_extractor
    if feature_extractor is None:
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            from tensorflow.keras.preprocessing.image import img_to_array
            
            feature_extractor = {
                'model': MobileNetV2(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224, 3),
                    pooling='avg'
                ),
                'preprocess_input': preprocess_input,
                'img_to_array': img_to_array
            }
            print("✅ Extractor de características (MobileNetV2) cargado")
        except ImportError:
            print("⚠️ TensorFlow no disponible. Usando extractor ligero (solo para inferencia)...")
            feature_extractor = None
    return feature_extractor

def load_model_on_startup():
    def load():
        global sklearn_model, scaler, class_names, model_ready
        print("🔄 Cargando modelo de hongos (Scikit-learn)...")
        
        # Cargar extractor de características primero
        load_feature_extractor()
        
        # 1. Intentar cargar Scikit-learn (Prioridad: más ligero)
        if MODELO_PATH_SKLEARN.exists() and SCALER_PATH.exists():
            try:
                sklearn_model = joblib.load(MODELO_PATH_SKLEARN)
                scaler = joblib.load(SCALER_PATH)
                print(f"✅ Modelo SKLEARN cargado exitosamente: {MODELO_PATH_SKLEARN}")
            except Exception as e:
                print(f"⚠️ Error cargando Scikit-learn: {e}")
        
        # Cargar clases
        try:
            if CLASS_PATH.exists():
                with open(CLASS_PATH, 'r', encoding='utf-8') as f:
                    class_names = json.load(f)
                print(f"✅ Clases cargadas: {len(class_names)} especies")
            else:
                print(f"❌ ERROR: El archivo de clases no existe en {CLASS_PATH}")
        except Exception as e:
            print(f"❌ Error cargando clases: {e}")
        
        model_ready = True
        print("✅ Carga del modelo completada")
    
    # Ejecutar en un hilo separado para no bloquear el inicio del servidor
    thread = threading.Thread(target=load)
    thread.daemon = True
    thread.start()

# Ejecutar carga al importar (ahora es asíncrona)
load_model_on_startup()

def extract_features(contents):
    """Extrae características de una imagen usando MobileNetV2 (si está disponible)"""
    fe = load_feature_extractor()
    
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = img.resize((224, 224))
    
    if fe:
        # Usar MobileNetV2 para características (mejor precisión)
        img_array = fe['img_to_array'](img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = fe['preprocess_input'](img_array)
        return fe['model'].predict(img_array, verbose=0)[0]
    else:
        # Fallback: características simples (píxeles aplanados)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array.flatten()[:1280]  # Mismo tamaño que MobileNetV2

@router.post("/identificar")
async def identificar_hongo(file: UploadFile = File(...)):
    """
    Identifica un hongo a partir de una imagen usando Scikit-learn
    """
    if not model_ready:
        raise HTTPException(status_code=503, detail="Modelo aún cargándose, por favor intenta en unos segundos")
    
    if sklearn_model is None or scaler is None or not class_names:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    try:
        # Leer imagen
        contents = await file.read()
        
        # Extraer características
        features = extract_features(contents)
        features_scaled = scaler.transform([features])
        
        # Predecir
        predictions_proba = sklearn_model.predict_proba(features_scaled)[0]
        idx = np.argmax(predictions_proba)
        confianza = float(predictions_proba[idx])
        especie = class_names[str(idx)]
        
        # Guardar imagen
        filename = f"{uuid.uuid4()}.jpg"
        filepath = UPLOAD_DIR / filename
        img_original = Image.open(io.BytesIO(contents))
        img_original.save(filepath)
        
        # Top 3 predicciones
        top_3_idx = np.argsort(predictions_proba)[-3:][::-1]
        sugerencias = [
            {
                "especie": class_names[str(tidx)],
                "confianza": float(predictions_proba[tidx])
            }
            for tidx in top_3_idx
        ]
        
        return {
            "success": True,
            "especie": especie,
            "confianza": confianza,
            "sugerencias": sugerencias,
            "modelo_usado": "scikit-learn",
            "imagen_url": f"/uploads/hongos/{filename}"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
