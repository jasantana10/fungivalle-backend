# app/ml/train_model_completo.py
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

def entrenar_modelo_completo():
    """
    Entrena el modelo desde cero en DOS fases para alcanzar 80% de precisión
    """
    print("="*60)
    print("🍄 ENTRENAMIENTO COMPLETO DE HONGOS - 2 FASES")
    print("="*60)
    
    # Configuración de rutas para FUNGIVALLE
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATASET_DIR = BASE_DIR / "app" / "ml" / "dataset"
    MODELOS_DIR = BASE_DIR / "app" / "ml" / "models"
    
    # Crear directorios
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Verificar dataset
    if not DATASET_DIR.exists():
        print(f"❌ No existe el dataset en: {DATASET_DIR}")
        print("   Ejecuta primero preparar_dataset.py")
        return False
    
    # Parámetros
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    EPOCHS_FASE1 = 30  # Más épocas para fase 1
    EPOCHS_FASE2 = 20  # Fine-tuning
    
    print(f"\n📊 Configuración:")
    print(f"   - Dataset: {DATASET_DIR}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Épocas fase 1: {EPOCHS_FASE1}")
    print(f"   - Épocas fase 2: {EPOCHS_FASE2}")
    
    # ===== FASE 1: ENTRENAR DESDE CERO CON TRANSFER LEARNING =====
    print("\n" + "="*60)
    print("🎓 FASE 1: ENTRENAMIENTO INICIAL (desde ImageNet)")
    print("="*60)
    
    # Data augmentation AGGRESSIVE para fase 1
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        zoom_range=0.3,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Generadores
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    
    # Guardar clases
    class_names = {v: k for k, v in train_generator.class_indices.items()}
    with open(MODELOS_DIR / 'class_names.json', 'w', encoding='utf-8') as f:
        json.dump(class_names, f, indent=2)
    
    print(f"\n📝 Clases a reconocer: {len(class_names)}")
    
    # Modelo base CONGELADO
    base_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    base_model.trainable = False  # Congelado en fase 1
    
    # Clasificador POTENTE
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(len(class_names), activation='softmax')
    ])
    
    # Compilar
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks fase 1
    callbacks_fase1 = [
        keras.callbacks.ModelCheckpoint(
            str(MODELOS_DIR / 'fase1_modelo.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    # Entrenar fase 1
    history_fase1 = model.fit(
        train_generator,
        epochs=EPOCHS_FASE1,
        validation_data=validation_generator,
        callbacks=callbacks_fase1,
        verbose=1
    )
    
    # Evaluar fase 1
    loss_fase1, acc_fase1 = model.evaluate(validation_generator, verbose=0)
    print(f"\n📊 Resultados Fase 1: {acc_fase1:.2%}")
    
    # ===== FASE 2: FINE-TUNING =====
    print("\n" + "="*60)
    print("🎓 FASE 2: FINE-TUNING (ajuste fino)")
    print("="*60)
    
    # Descongelar parte del modelo base
    model.trainable = True
    
    # Congelar primeras capas, descongelar últimas 100
    for i, layer in enumerate(model.layers[0].layers):
        if i < 100:
            layer.trainable = False
        else:
            layer.trainable = True
    
    # Recompilar con learning rate MUY bajo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"   Capas entrenables: {sum(1 for l in model.layers[0].layers if l.trainable)}/{len(model.layers[0].layers)}")
    
    # Callbacks fase 2
    callbacks_fase2 = [
        keras.callbacks.ModelCheckpoint(
            str(MODELOS_DIR / 'mejor_modelo.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Entrenar fase 2
    history_fase2 = model.fit(
        train_generator,
        epochs=EPOCHS_FASE2,
        validation_data=validation_generator,
        callbacks=callbacks_fase2,
        verbose=1
    )
    
    # Evaluación final
    loss_final, acc_final = model.evaluate(validation_generator, verbose=0)
    
    # ===== GUARDAR MODELO FINAL =====
    # Guardar como .keras (formato moderno)
    model.save(MODELOS_DIR / 'modelo_finetuned.keras')
    
    # También guardar copia con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(MODELOS_DIR / f'modelo_hongos_{timestamp}.keras')
    
    print("\n" + "="*60)
    print("✅ **ENTRENAMIENTO COMPLETADO**")
    print("="*60)
    print(f"📊 EVOLUCIÓN:")
    print(f"   - Precisión fase 1: {acc_fase1:.2%}")
    print(f"   - Precisión final:  {acc_final:.2%}")
    print(f"   - Mejora:           {acc_final - acc_fase1:+.2%}")
    print(f"\n📁 Modelos guardados en: {MODELOS_DIR}")
    print(f"   - modelo_finetuned.keras (para la app)")
    print(f"   - mejor_modelo.keras (mejor versión)")
    print("="*60)
    
    return True

if __name__ == "__main__":
    entrenar_modelo_completo()