# app/ml/data_augmentation.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import os
import shutil
from pathlib import Path
import random

def augment_images_for_species(original_images_dir, output_dir, target_count=150):
    """
    Genera imágenes aumentadas para una especie específica
    
    Args:
        original_images_dir: Carpeta con las imágenes originales
        output_dir: Carpeta donde guardar las imágenes aumentadas
        target_count: Número total de imágenes deseadas
    """
    # Crear directorio de salida
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Obtener imágenes originales
    original_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        original_images.extend(list(original_images_dir.glob(ext)))
    
    if not original_images:
        print(f"   ❌ No se encontraron imágenes en {original_images_dir}")
        return 0
    
    print(f"   📸 Generando aumentos desde {len(original_images)} imágenes originales...")
    
    # Generador de aumentos
    datagen = ImageDataGenerator(
        rotation_range=45,           # Rotación aleatoria
        width_shift_range=0.3,       # Desplazamiento horizontal
        height_shift_range=0.3,      # Desplazamiento vertical
        brightness_range=[0.6, 1.4], # Variación de brillo
        horizontal_flip=True,        # Volteo horizontal
        vertical_flip=False,         # Sin volteo vertical
        zoom_range=0.3,              # Zoom
        shear_range=0.2,             # Cortado
        fill_mode='nearest'          # Relleno
    )
    
    # Contar imágenes existentes
    existing_images = list(output_dir.glob("*.jpg"))
    existing_count = len(existing_images)
    
    if existing_count >= target_count:
        print(f"   ✅ Ya hay {existing_count} imágenes (suficientes)")
        return existing_count
    
    needed = target_count - existing_count
    print(f"   🎯 Necesitamos {needed} imágenes más para llegar a {target_count}")
    
    # Calcular cuántas generar por imagen original
    target_per_original = max(1, needed // len(original_images) + 1)
    
    generated_count = 0
    
    for img_path in original_images:
        # Cargar imagen
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        
        # Generar variaciones
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=str(output_dir),
                                   save_prefix=f"aug_{img_path.stem}", save_format='jpg'):
            i += 1
            generated_count += 1
            if i >= target_per_original or generated_count >= needed:
                break
        if generated_count >= needed:
            break
    
    # Contar imágenes finales
    final_count = len(list(output_dir.glob("*.jpg")))
    print(f"   ✅ Total imágenes: {final_count} (+{final_count - existing_count} nuevas)")
    
    return final_count