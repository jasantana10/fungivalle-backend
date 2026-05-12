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
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
except ImportError:
    print("ADVERTENCIA: TensorFlow/Keras no instalado. Por favor instala requirements-train.txt")
    sys.exit(1)

# Configuracion
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = BASE_DIR / "app" / "ml" / "dataset"
MODELS_DIR = BASE_DIR / "app" / "ml" / "models"
MIN_IMAGES_PER_SPECIES = 20  # Reducido para que sea más rápido
TARGET_IMAGES_PER_SPECIES = 30  # Muy reducido para evitar errores de memoria

def load_data():
    """Carga y preprocesa el dataset usando lotes para mayor velocidad"""
    print("Cargando dataset...")
    
    if not DATASET_DIR.exists():
        print(f"ERROR: Dataset no encontrado en: {DATASET_DIR}")
        print(f"   Por favor coloca tu dataset de imagenes organizadas por carpetas en esa ruta")
        sys.exit(1)

    features = []
    labels = []
    class_names = []
    image_paths = []

    # Recorrer carpetas de especies y recopilar rutas
    species_folders = sorted([f for f in DATASET_DIR.iterdir() if f.is_dir()])
    
    for species_idx, species_folder in enumerate(species_folders):
        species_name = species_folder.name
        img_files = sorted(list(species_folder.glob("*.jpg")) + list(species_folder.glob("*.jpeg")) + list(species_folder.glob("*.png")))
        
        if len(img_files) < MIN_IMAGES_PER_SPECIES:
            print(f"ADVERTENCIA: Saltando {species_name}: solo {len(img_files)} imagenes (minimo {MIN_IMAGES_PER_SPECIES})")
            continue
            
        class_names.append(species_name)
        selected_imgs = img_files[:TARGET_IMAGES_PER_SPECIES]
        print(f"Procesando {species_name} ({len(selected_imgs)} imagenes)...")
        
        for img_path in selected_imgs:
            image_paths.append((img_path, species_idx))

    if not image_paths:
        print("ERROR: No se encontraron suficientes imagenes para entrenar")
        sys.exit(1)

    print(f"\nTotal de imagenes a procesar: {len(image_paths)}")
    
    # Cargar MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    
    # Procesar en lotes
    all_imgs = []
    all_labels = []
    
    for img_path, label in image_paths:
        try:
            img = load_img(img_path, target_size=IMG_SIZE)
            img_array = img_to_array(img)
            all_imgs.append(img_array)
            all_labels.append(label)
        except Exception as e:
            print(f"   ERROR: Al procesar {img_path.name}: {e}")
    
    # Preprocesar y extraer caracteristicas en lotes
    print("\nExtrayendo caracteristicas en lotes...")
    X = np.array(all_imgs)
    X = preprocess_input(X)
    features = base_model.predict(X, batch_size=BATCH_SIZE, verbose=1)
    labels = np.array(all_labels)
    
    print(f"\nDataset cargado: {len(features)} imagenes, {len(class_names)} especies")
    return features, labels, class_names

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
    print("ENTRENAMIENTO DE MODELO CON SCIKIT-LEARN (Rapido)")
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
