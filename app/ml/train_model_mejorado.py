# app/ml/train_model_mejorado.py
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

def entrenar_modelo_hongos_mejorado():
    """
    Entrena el modelo con fine-tuning para alcanzar mayor precisión
    """
    print("="*60)
    print("🍄 ENTRENAMIENTO MEJORADO DE HONGOS - CON FINE-TUNING")
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
        return False
    
    # Parámetros de entrenamiento (optimizados)
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    EPOCHS_FASE1 = 20  # Primera fase (solo clasificador)
    EPOCHS_FASE2 = 15  # Segunda fase (fine-tuning)
    
    print(f"\n📊 Configuración:")
    print(f"   - Dataset: {DATASET_DIR}")
    print(f"   - Tamaño imágenes: {IMG_HEIGHT}x{IMG_WIDTH}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Épocas fase 1: {EPOCHS_FASE1}")
    print(f"   - Épocas fase 2: {EPOCHS_FASE2}")
    
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
    
    # Verificar que hay suficientes imágenes
    clases_validas = []
    for clase in clases:
        ruta_clase = DATASET_DIR / clase
        num_imagenes = len([f for f in os.listdir(ruta_clase) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if num_imagenes >= 20:
            clases_validas.append(clase)
    
    if len(clases_validas) < len(clases):
        print(f"\n⚠️ Algunas especies tienen menos de 20 imágenes")
        print(f"   Se usarán {len(clases_validas)} especies con suficientes imágenes")
    
    # 1. PREPARAR DATOS CON DATA AUGMENTATION MEJORADO
    print("\n🔄 Preparando datos con aumento mejorado...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,           # Más rotación
        width_shift_range=0.3,       # Más desplazamiento
        height_shift_range=0.3,
        brightness_range=[0.7, 1.3], # Más variación de brillo
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.3,               # Más zoom
        shear_range=0.2,              # Distorsión
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Datos de entrenamiento
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        classes=clases_validas if len(clases_validas) < len(clases) else None
    )
    
    # Datos de validación (sin aumentos)
    valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    validation_generator = valid_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True,
        classes=clases_validas if len(clases_validas) < len(clases) else None
    )
    
    # Guardar mapeo de clases
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    
    with open(MODELOS_DIR / 'class_names.json', 'w', encoding='utf-8') as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)
    
    print(f"\n📝 Mapeo de clases guardado en: {MODELOS_DIR / 'class_names.json'}")
    print(f"   Clases finales: {len(class_names)}")
    
    # ===== FASE 1: ENTRENAR SOLO EL CLASIFICADOR =====
    print("\n" + "="*60)
    print("🎓 FASE 1: ENTRENANDO CLASIFICADOR (capas base congeladas)")
    print("="*60)
    
    # Cargar MobileNetV2 pre-entrenado
    base_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    base_model.trainable = False  # Congelar durante fase 1
    
    # Construir modelo completo
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),  # Más neuronas
        layers.Dropout(0.4),                    # Más dropout
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation='softmax')
    ])
    
    # Compilar
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks para fase 1
    callbacks_fase1 = [
        keras.callbacks.ModelCheckpoint(
            str(MODELOS_DIR / 'fase1_modelo.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
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
    print(f"\n📊 Resultados Fase 1:")
    print(f"   Precisión: {acc_fase1:.2%}")
    
    # ===== FASE 2: FINE-TUNING =====
    print("\n" + "="*60)
    print("🎓 FASE 2: FINE-TUNING (descongelando capas superiores)")
    print("="*60)
    
    # Descongelar las últimas capas de MobileNetV2
    model.trainable = True
    
    # Congelar las primeras capas, descongelar las últimas 100
    for layer in model.layers[0].layers[:100]:
        layer.trainable = False
    for layer in model.layers[0].layers[100:]:
        layer.trainable = True
    
    # Recompilar con learning rate mucho más bajo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00005),  # 20x más lento
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"   Capas entrenables: {sum(1 for l in model.layers[0].layers if l.trainable)} de {len(model.layers[0].layers)}")
    
    # Callbacks para fase 2
    callbacks_fase2 = [
        keras.callbacks.ModelCheckpoint(
            str(MODELOS_DIR / 'mejor_modelo.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
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
            min_lr=0.000001,
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modelo_final_path = MODELOS_DIR / f'modelo_hongos_{timestamp}.keras'
    model.save(modelo_final_path)
    
    # Guardar como modelo_finetuned.keras (el que usará la app)
    shutil.copy(modelo_final_path, MODELOS_DIR / 'modelo_finetuned.keras')
    
    print(f"\n💾 Modelo final guardado como: modelo_finetuned.keras")
    
    # Guardar métricas
    with open(MODELOS_DIR / 'metricas.json', 'w') as f:
        json.dump({
            'accuracy_fase1': float(acc_fase1),
            'accuracy_final': float(acc_final),
            'loss_final': float(loss_final),
            'epochs_fase1': len(history_fase1.history['accuracy']),
            'epochs_fase2': len(history_fase2.history['accuracy']),
            'species': len(class_names),
            'mejora': float(acc_final - acc_fase1)
        }, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ **ENTRENAMIENTO MEJORADO COMPLETADO**")
    print("="*60)
    print(f"📊 COMPARATIVA:")
    print(f"   - Precisión fase 1: {acc_fase1:.2%}")
    print(f"   - Precisión final:  {acc_final:.2%}")
    print(f"   - Mejora:           {acc_final - acc_fase1:+.2%}")
    print(f"\n📁 Modelo guardado en: {MODELOS_DIR}")
    print("="*60)
    
    return True

if __name__ == "__main__":
    entrenar_modelo_hongos_mejorado()