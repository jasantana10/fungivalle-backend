# app/ml/train_model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import json
import sys
from pathlib import Path
import shutil
from datetime import datetime

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def entrenar_modelo_hongos():
    """
    Entrena el modelo de identificación de hongos y lo guarda en formato .keras
    """
    print("="*60)
    print("🍄 ENTRENAMIENTO DE MODELO DE HONGOS - FUNGIVALLE")
    print("="*60)
    
    # Configuración de rutas
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATASET_DIR = BASE_DIR / "app" / "ml" / "dataset"
    MODELOS_DIR = BASE_DIR / "app" / "ml" / "models"
    
    # Crear directorios si no existen
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Verificar que existe el dataset
    if not DATASET_DIR.exists():
        print(f"❌ No existe el dataset en: {DATASET_DIR}")
        print("   Por favor, coloca tus imágenes organizadas en carpetas por especie")
        print("   Ejemplo:")
        print(f"   {DATASET_DIR}/")
        print("       Amanita_muscaria/")
        print("           img1.jpg")
        print("           img2.jpg")
        print("       Boletus_edulis/")
        print("           img1.jpg")
        return False
    
    # Parámetros de entrenamiento
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    EPOCHS = 20
    
    print(f"\n📊 Configuración:")
    print(f"   - Dataset: {DATASET_DIR}")
    print(f"   - Tamaño imágenes: {IMG_HEIGHT}x{IMG_WIDTH}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Épocas: {EPOCHS}")
    
    # Contar clases (especies)
    clases = [d for d in os.listdir(DATASET_DIR) 
              if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    if not clases:
        print("❌ No se encontraron carpetas de especies en el dataset")
        return False
    
    print(f"\n🍄 Especies encontradas: {len(clases)}")
    for i, clase in enumerate(clases, 1):
        ruta_clase = DATASET_DIR / clase
        num_imagenes = len([f for f in os.listdir(ruta_clase) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"   {i}. {clase}: {num_imagenes} imágenes")
    
    # 1. PREPARAR DATOS CON DATA AUGMENTATION
    print("\n🔄 Preparando datos...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2
    )
    
    # Datos de entrenamiento
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Datos de validación
    validation_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    
    # Guardar mapeo de clases
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    
    with open(MODELOS_DIR / 'class_names.json', 'w', encoding='utf-8') as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)
    
    print(f"\n📝 Mapeo de clases guardado en: {MODELOS_DIR / 'class_names.json'}")
    
    # 2. CONSTRUIR MODELO CON TRANSFER LEARNING
    print("\n🏗️ Construyendo modelo...")
    
    # Cargar MobileNetV2 pre-entrenado
    base_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    base_model.trainable = False
    
    # Construir modelo completo
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(clases), activation='softmax')
    ])
    
    # Compilar
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 3. CALLBACKS
    callbacks = [
        # Guardar el mejor modelo en formato .keras
        keras.callbacks.ModelCheckpoint(
            str(MODELOS_DIR / 'mejor_modelo.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Detener si no mejora
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Reducir learning rate
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # 4. ENTRENAR
    print("\n🎓 Comenzando entrenamiento...")
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. GUARDAR MODELO FINAL EN FORMATO .KERAS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modelo_final_path = MODELOS_DIR / f'modelo_hongos_{timestamp}.keras'
    model.save(modelo_final_path)
    
    # También guardar como modelo_finetuned.keras (el que usará la app)
    shutil.copy(modelo_final_path, MODELOS_DIR / 'modelo_finetuned.keras')
    
    print(f"\n💾 Modelo final guardado como: modelo_finetuned.keras")
    print(f"   Ruta completa: {MODELOS_DIR / 'modelo_finetuned.keras'}")
    
    # 6. EVALUAR
    print("\n📊 Evaluando modelo...")
    loss, accuracy = model.evaluate(validation_generator)
    print(f"   Precisión en validación: {accuracy:.2%}")
    
    # Guardar métricas
    with open(MODELOS_DIR / 'metricas.json', 'w') as f:
        json.dump({
            'accuracy': float(accuracy),
            'loss': float(loss),
            'epochs': len(history.history['accuracy']),
            'species': len(clases)
        }, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ **ENTRENAMIENTO COMPLETADO**")
    print("="*60)
    print(f"📁 Modelo guardado en: {MODELOS_DIR}")
    print(f"   - modelo_finetuned.keras (principal)")
    print(f"   - mejor_modelo.keras (mejor versión)")
    print(f"   - class_names.json (mapeo de especies)")
    print(f"   - metricas.json (rendimiento)")
    print(f"\n🎯 Precisión final: {accuracy:.2%}")
    print("="*60)
    
    return True

if __name__ == "__main__":
    entrenar_modelo_hongos()