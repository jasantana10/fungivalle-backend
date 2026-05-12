#!/usr/bin/env python3
import os
import sys
import json
import joblib
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
except ImportError:
    print("ADVERTENCIA: TensorFlow/Keras no instalado. Por favor instala requirements-train.txt")
    sys.exit(1)

# Configuracion
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = BASE_DIR / "app" / "ml" / "dataset"
MODELS_DIR = BASE_DIR / "app" / "ml" / "models"
MIN_IMAGES_PER_SPECIES = 50  # Minimo de imagenes por especie
TARGET_IMAGES_PER_SPECIES = 150  # Objetivo de imagenes por especie

def load_data():
    """Carga y preprocesa el dataset"""
    print("Cargando dataset...")
    
    if not DATASET_DIR.exists():
        print(f"ERROR: Dataset no encontrado en: {DATASET_DIR}")
        print(f"   Por favor coloca tu dataset de imagenes organizadas por carpetas en esa ruta")
        sys.exit(1)

    features = []
    labels = []
    class_names = []

    # Cargar MobileNetV2 para extraccion de caracteristicas
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg'  # Pooling promedio para reducir dimensionalidad
    )

    # Recorrer carpetas de especies
    species_folders = sorted([f for f in DATASET_DIR.iterdir() if f.is_dir()])
    
    for species_idx, species_folder in enumerate(species_folders):
        species_name = species_folder.name
        image_files = sorted(list(species_folder.glob("*.jpg")) + list(species_folder.glob("*.jpeg")) + list(species_folder.glob("*.png")))
        
        if len(image_files) < MIN_IMAGES_PER_SPECIES:
            print(f"ADVERTENCIA: Saltando {species_name}: solo {len(image_files)} imagenes (minimo {MIN_IMAGES_PER_SPECIES})")
            continue
            
        class_names.append(species_name)
        print(f"Procesando {species_name} ({len(image_files)} imagenes)...")
        
        # Procesar imagenes
        for img_file in image_files[:TARGET_IMAGES_PER_SPECIES]:  # Maximo TARGET_IMAGES_PER_SPECIES
            try:
                img = Image.open(img_file).convert('RGB')
                img = img.resize(IMG_SIZE)
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                
                # Extraer caracteristicas
                feat = base_model.predict(img_array, verbose=0)[0]
                features.append(feat)
                labels.append(species_idx)
                
            except Exception as e:
                print(f"   ERROR: Al procesar {img_file.name}: {e}")

    if not features:
        print("ERROR: No se encontraron suficientes imagenes para entrenar")
        sys.exit(1)

    print(f"\nDataset cargado: {len(features)} imagenes, {len(class_names)} especies")
    return np.array(features), np.array(labels), class_names

def train_model(X, y, class_names):
    """Entrena el modelo de clasificacion"""
    print("\nEntrenando modelo...")
    
    # Dividir datos: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Escalar caracteristicas
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar SVM (buen balance entre rendimiento y memoria)
    print("   Entrenando SVM...")
    model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluar
    print("\nEvaluando modelo...")
    y_pred_val = model.predict(X_val_scaled)
    print(f"\n   Accuracy (Validacion): {accuracy_score(y_val, y_pred_val):.4f}")
    
    y_pred_test = model.predict(X_test_scaled)
    print(f"   Accuracy (Test): {accuracy_score(y_test, y_pred_test):.4f}")
    
    print("\n" + classification_report(y_test, y_pred_test, target_names=class_names))
    
    return model, scaler

def save_model(model, scaler, class_names):
    """Guarda el modelo y archivos relacionados"""
    print("\nGuardando modelo...")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Guardar modelo y scaler
    joblib.dump(model, MODELS_DIR / "modelo_sklearn.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    
    # Guardar nombres de clases
    with open(MODELS_DIR / "class_names.json", 'w', encoding='utf-8') as f:
        json.dump({str(i): name for i, name in enumerate(class_names)}, f, ensure_ascii=False, indent=2)
    
    print(f"Modelo guardado en: {MODELS_DIR}")
    print(f"   - modelo_sklearn.pkl (clasificador)")
    print(f"   - scaler.pkl (escalador de caracteristicas)")
    print(f"   - class_names.json (nombres de especies)")

def main():
    print("="*60)
    print("ENTRENAMIENTO DE MODELO CON SCIKIT-LEARN")
    print("="*60)
    
    # Cargar datos
    X, y, class_names = load_data()
    
    # Entrenar
    model, scaler = train_model(X, y, class_names)
    
    # Guardar
    save_model(model, scaler, class_names)
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
    print("="*60)

if __name__ == "__main__":
    main()
