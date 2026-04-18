# app/ml/test_model.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import os
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def probar_modelo():
    """
    Prueba el modelo entrenado con una imagen
    """
    print("🔬 PROBANDO MODELO DE HONGOS")
    print("="*50)
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MODELOS_DIR = BASE_DIR / "app" / "ml" / "models"
    
    # Buscar modelo
    modelo_path = MODELOS_DIR / "modelo_finetuned.keras"
    if not modelo_path.exists():
        print(f"❌ No existe el modelo en: {modelo_path}")
        # Buscar cualquier .keras
        modelos = list(MODELOS_DIR.glob("*.keras"))
        if modelos:
            modelo_path = max(modelos, key=lambda p: p.stat().st_mtime)
            print(f"✅ Usando: {modelo_path.name}")
        else:
            return
    
    # Cargar modelo
    print(f"\n🔄 Cargando modelo: {modelo_path.name}")
    model = tf.keras.models.load_model(modelo_path)
    
    # Cargar clases
    class_path = MODELOS_DIR / "class_names.json"
    if not class_path.exists():
        print("❌ No existe class_names.json")
        return
    
    with open(class_path, 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    
    print(f"\n📊 Especies que reconoce: {len(class_names)}")
    
    # Ruta de imagen a probar
    while True:
        ruta = input("\n📸 Ruta de imagen (o 'q' para salir): ").strip()
        if ruta.lower() == 'q':
            break
        
        if not os.path.exists(ruta):
            print("❌ Archivo no encontrado")
            continue
        
        # Procesar
        img = image.load_img(ruta, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Predecir
        predictions = model.predict(img_array, verbose=0)[0]
        idx = np.argmax(predictions)
        confianza = predictions[idx] * 100
        especie = class_names[str(idx)]
        
        print(f"\n🎯 RESULTADO:")
        print(f"   Especie: {especie}")
        print(f"   Confianza: {confianza:.2f}%")
        
        # Top 3
        top_3 = np.argsort(predictions)[-3:][::-1]
        print(f"\n📊 Top 3:")
        for i, tidx in enumerate(top_3, 1):
            print(f"   {i}. {class_names[str(tidx)]}: {predictions[tidx]*100:.1f}%")

if __name__ == "__main__":
    probar_modelo()