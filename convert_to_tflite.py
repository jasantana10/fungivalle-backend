# convert_to_tflite.py
import tensorflow as tf
import os
from pathlib import Path

def convert():
    base_dir = Path(__file__).resolve().parent
    modelos_dir = base_dir / "app" / "ml" / "models"
    model_path = modelos_dir / "modelo_finetuned.keras"
    tflite_path = modelos_dir / "modelo_finetuned.tflite"

    if not model_path.exists():
        print(f"❌ Error: No se encuentra {model_path}")
        return

    print(f"🔄 Cargando modelo Keras: {model_path}...")
    try:
        # Cargamos el modelo
        model = tf.keras.models.load_model(model_path)
        
        # Convertimos a TFLite
        print("⚡ Convirtiendo a formato TFLite (optimizado para memoria)...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Opcional: optimización para tamaño
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()

        # Guardamos el archivo
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ ¡Éxito! Modelo TFLite guardado en: {tflite_path}")
        print(f"   Tamaño original: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        print(f"   Tamaño TFLite: {os.path.getsize(tflite_path) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"❌ Error durante la conversión: {e}")

if __name__ == "__main__":
    convert()
