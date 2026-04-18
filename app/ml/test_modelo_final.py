import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt

def probar_modelo_final():
    """
    Prueba el modelo entrenado con imágenes
    """
    print("="*60)
    print("🍄 PROBANDO MODELO FINAL DE HONGOS")
    print("="*60)
    
    # Configuración de rutas
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MODELOS_DIR = BASE_DIR / "app" / "ml" / "models"
    
    # Buscar el modelo fine-tuned
    modelo_path = MODELOS_DIR / "modelo_finetuned.keras"
    
    # Si no existe, buscar cualquier modelo .keras
    if not modelo_path.exists():
        print("⚠️ No se encuentra modelo_finetuned.keras")
        modelos = list(MODELOS_DIR.glob("*.keras"))
        if modelos:
            # Tomar el más reciente
            modelo_path = max(modelos, key=lambda p: p.stat().st_mtime)
            print(f"✅ Usando: {modelo_path.name}")
        else:
            print("❌ No hay modelos disponibles en:", MODELOS_DIR)
            return
    
    # Cargar modelo
    print(f"\n🔄 Cargando modelo: {modelo_path.name}")
    try:
        model = tf.keras.models.load_model(modelo_path)
        print("✅ Modelo cargado exitosamente")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    # Cargar clases
    class_path = MODELOS_DIR / "class_names.json"
    if not class_path.exists():
        print(f"❌ No existe {class_path}")
        return
    
    with open(class_path, 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    
    print(f"\n📊 ESPECIES QUE RECONOCE EL MODELO:")
    for idx, nombre in class_names.items():
        print(f"   {idx}: {nombre}")
    
    # Menú de opciones
    while True:
        print("\n" + "-"*60)
        print("📸 OPCIONES:")
        print("   1. Probar una imagen específica")
        print("   2. Probar con imagen del dataset (aleatoria)")
        print("   3. Ver estadísticas del modelo")
        print("   4. Salir")
        
        opcion = input("\n👉 Elige una opción (1-4): ").strip()
        
        if opcion == '1':
            probar_imagen_especifica(model, class_names, MODELOS_DIR)
        elif opcion == '2':
            probar_imagen_dataset(model, class_names, MODELOS_DIR)
        elif opcion == '3':
            ver_estadisticas(model, class_names, MODELOS_DIR)
        elif opcion == '4':
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción no válida")

def probar_imagen_especifica(model, class_names, MODELOS_DIR):
    """Prueba con una imagen específica que ingresa el usuario"""
    ruta = input("\n📂 Ruta de la imagen: ").strip()
    
    if not os.path.exists(ruta):
        print("❌ Archivo no encontrado")
        return
    
    try:
        # Cargar y preparar imagen
        img = image.load_img(ruta, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Predecir
        print("🤔 Analizando...")
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Resultados
        mostrar_resultados(predictions, class_names, img, ruta)
        
    except Exception as e:
        print(f"❌ Error procesando imagen: {e}")

def probar_imagen_dataset(model, class_names, MODELOS_DIR):
    """Prueba con una imagen aleatoria del dataset"""
    dataset_dir = MODELOS_DIR.parent / "dataset"
    
    if not dataset_dir.exists():
        print("❌ No se encuentra el dataset")
        return
    
    import random
    from pathlib import Path
    
    # Buscar todas las imágenes
    imagenes = []
    especies = []
    
    for especie_dir in dataset_dir.iterdir():
        if especie_dir.is_dir():
            for img_file in especie_dir.glob("*.jpg"):
                imagenes.append(img_file)
                especies.append(especie_dir.name)
            for img_file in especie_dir.glob("*.png"):
                imagenes.append(img_file)
                especies.append(especie_dir.name)
    
    if not imagenes:
        print("❌ No hay imágenes en el dataset")
        return
    
    # Seleccionar una aleatoria
    idx = random.randint(0, len(imagenes)-1)
    img_path = imagenes[idx]
    especie_real = especies[idx]
    
    print(f"\n📸 Imagen seleccionada: {img_path.name}")
    print(f"   Especie real: {especie_real}")
    
    try:
        # Cargar y preparar
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Predecir
        print("🤔 Analizando...")
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Mostrar resultados
        mostrar_resultados(predictions, class_names, img, str(img_path), especie_real)
        
    except Exception as e:
        print(f"❌ Error: {e}")

def mostrar_resultados(predictions, class_names, img, ruta, especie_real=None):
    """Muestra los resultados de la predicción"""
    
    # Mejor predicción
    idx = np.argmax(predictions)
    confianza = predictions[idx] * 100
    especie = class_names[str(idx)]
    
    print("\n" + "="*50)
    print("🎯 **RESULTADO DE IDENTIFICACIÓN**")
    print("="*50)
    
    if especie_real:
        print(f"📌 Especie real: {especie_real}")
        print(f"✅ Predicción:   {especie}")
        if especie.replace(" ", "_") == especie_real:
            print("   ✓ ¡ACIERTO!")
        else:
            print("   ✗ FALLO")
    else:
        print(f"🍄 Hongo identificado: {especie}")
    
    print(f"📊 Confianza: {confianza:.2f}%")
    
    # Barra de confianza
    barra = "█" * int(confianza/5) + "░" * (20 - int(confianza/5))
    print(f"   [{barra}]")
    
    # Top 5 predicciones
    top_5_idx = np.argsort(predictions)[-5:][::-1]
    print(f"\n📊 Top 5 predicciones:")
    for i, tidx in enumerate(top_5_idx, 1):
        esp = class_names[str(tidx)]
        conf = predictions[tidx] * 100
        if i == 1:
            print(f"   🏆 {i}. {conf:.1f}% - {esp} (GANADORA)")
        else:
            print(f"   {i}. {conf:.1f}% - {esp}")
    
    # Preguntar si mostrar imagen
    print("\n" + "-"*40)
    mostrar = input("¿Mostrar imagen? (s/n): ").strip().lower()
    
    if mostrar == 's':
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        titulo = f"Predicción: {especie}\nConfianza: {confianza:.1f}%"
        if especie_real:
            titulo = f"Real: {especie_real}\nPred: {especie}\nConf: {confianza:.1f}%"
        plt.title(titulo, fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def ver_estadisticas(model, class_names, MODELOS_DIR):
    """Muestra estadísticas del modelo"""
    print("\n📊 **ESTADÍSTICAS DEL MODELO**")
    print("="*50)
    
    # Información del modelo
    print(f"📦 Arquitectura:")
    print(f"   - Capas totales: {len(model.layers)}")
    
    # Contar parámetros
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_variables])
    
    print(f"   - Parámetros entrenables: {trainable_params:,}")
    print(f"   - Parámetros no entrenables: {non_trainable_params:,}")
    print(f"   - Total: {trainable_params + non_trainable_params:,}")
    
    print(f"\n🍄 Especies: {len(class_names)}")
    
    # Buscar métricas guardadas
    metricas_path = MODELOS_DIR / "metricas.json"
    if metricas_path.exists():
        with open(metricas_path, 'r') as f:
            metricas = json.load(f)
        
        print(f"\n📈 Métricas guardadas:")
        if 'accuracy_fase1' in metricas:
            print(f"   - Precisión fase 1: {metricas['accuracy_fase1']:.2%}")
        if 'accuracy_final' in metricas:
            print(f"   - Precisión final: {metricas['accuracy_final']:.2%}")

def probar_lote_imagenes(model, class_names, MODELOS_DIR):
    """Prueba con múltiples imágenes y calcula precisión"""
    dataset_dir = MODELOS_DIR.parent / "dataset"
    
    if not dataset_dir.exists():
        print("❌ No se encuentra el dataset")
        return
    
    print("\n📊 **PRUEBA POR LOTE**")
    print("="*50)
    
    resultados = []
    total = 0
    aciertos = 0
    
    for especie_dir in dataset_dir.iterdir():
        if not especie_dir.is_dir():
            continue
            
        especie_real = especie_dir.name
        imagenes = list(especie_dir.glob("*.jpg")) + list(especie_dir.glob("*.png"))
        
        # Probar hasta 10 imágenes por especie
        for img_path in imagenes[:10]:
            try:
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0
                
                predictions = model.predict(img_array, verbose=0)[0]
                idx = np.argmax(predictions)
                especie_pred = class_names[str(idx)]
                
                total += 1
                if especie_pred.replace(" ", "_") == especie_real:
                    aciertos += 1
                    
            except Exception as e:
                print(f"Error con {img_path.name}: {e}")
    
    if total > 0:
        print(f"\n✅ Resultados:")
        print(f"   - Total imágenes: {total}")
        print(f"   - Aciertos: {aciertos}")
        print(f"   - Precisión: {aciertos/total:.2%}")

if __name__ == "__main__":
    probar_modelo_final()