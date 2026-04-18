# app/ml/reentrenar_desde_sugerencias.py

import os
import json
import shutil
from pathlib import Path
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import FungiFinding

def recolectar_especies_para_reentrenar(min_sugerencias=3):
    """
    Recolecta imágenes de especies sugeridas por usuarios
    SOLO AGREGA NUEVAS IMÁGENES, NO BORRA LAS EXISTENTES
    """
    db = SessionLocal()
    
    try:
        from sqlalchemy import func
        
        # Buscar especies con suficientes sugerencias
        resultados = db.query(
            FungiFinding.user_suggestion,
            func.count(FungiFinding.id).label('count')
        ).filter(
            FungiFinding.user_suggestion.isnot(None),
            FungiFinding.user_suggestion != '',
            FungiFinding.user_suggestion != 'Desconocido'
        ).group_by(
            FungiFinding.user_suggestion
        ).having(
            func.count(FungiFinding.id) >= min_sugerencias
        ).all()
        
        print(f"🔍 Especies con {min_sugerencias}+ sugerencias:")
        
        for especie, count in resultados:
            print(f"\n   📌 {especie}: {count} imágenes totales")
            
            # Obtener todas las imágenes de esa especie
            imagenes = db.query(FungiFinding).filter(
                FungiFinding.user_suggestion == especie
            ).all()
            
            # Crear carpeta en dataset (si no existe)
            dataset_dir = Path("app/ml/dataset") / especie.replace(" ", "_")
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Contar imágenes ya existentes en la carpeta
            imagenes_existentes = set()
            for existing_file in dataset_dir.glob("*.jpg"):
                imagenes_existentes.add(existing_file.stem)
            
            imagenes_nuevas = 0
            
            # SOLO AGREGAR IMÁGENES QUE NO EXISTEN
            for img in imagenes:
                if img.image_url:
                    src = Path(".") / img.image_url.lstrip('/')
                    if src.exists():
                        # Usar el ID como nombre único
                        nombre_archivo = f"{img.id}.jpg"
                        dst = dataset_dir / nombre_archivo
                        
                        # Solo copiar si no existe
                        if not dst.exists():
                            shutil.copy2(src, dst)
                            imagenes_nuevas += 1
                            print(f"      ✅ Nueva imagen agregada: {nombre_archivo}")
            
            print(f"   📊 Resumen: {imagenes_nuevas} imágenes nuevas agregadas (total en carpeta: {len(list(dataset_dir.glob('*.jpg')))})")
        
        return resultados
        
    finally:
        db.close()

def reentrenar_modelo():
    """
    Reentrena el modelo con las nuevas imágenes (SOLO SI HAY NUEVAS)
    """
    print("\n🔄 Verificando si hay nuevas imágenes para reentrenar...")
    
    # Verificar si hay cambios en el dataset
    dataset_path = Path("app/ml/dataset")
    if not dataset_path.exists():
        print("⚠️ No hay dataset para reentrenar")
        return
    
    # Contar imágenes totales
    total_imagenes = 0
    for especie_dir in dataset_path.iterdir():
        if especie_dir.is_dir():
            total_imagenes += len(list(especie_dir.glob("*.jpg")))
    
    print(f"📊 Total de imágenes en dataset: {total_imagenes}")
    
    if total_imagenes < 50:
        print("⚠️ Pocas imágenes para reentrenar (mínimo 50 recomendadas)")
        respuesta = input("¿Aún así quieres reentrenar? (s/n): ")
        if respuesta.lower() != 's':
            print("❌ Reentrenamiento cancelado")
            return
    
    print("\n🚀 Iniciando reentrenamiento del modelo...")
    
    # Llamar al script de entrenamiento
    import subprocess
    result = subprocess.run(
        ["python", "app/ml/train_model_completo.py"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
    else:
        print("✅ Modelo reentrenado exitosamente")
        
        # Marcar las imágenes como usadas para entrenamiento (opcional)
        # Podrías agregar un campo 'used_for_training' en la tabla

def listar_estado_dataset():
    """
    Muestra el estado actual del dataset
    """
    dataset_path = Path("app/ml/dataset")
    if not dataset_path.exists():
        print("❌ No existe carpeta de dataset")
        return
    
    print("\n📊 ESTADO ACTUAL DEL DATASET:")
    print("-" * 50)
    
    total = 0
    for especie_dir in dataset_path.iterdir():
        if especie_dir.is_dir():
            count = len(list(especie_dir.glob("*.jpg")))
            total += count
            print(f"   🍄 {especie_dir.name}: {count} imágenes")
    
    print("-" * 50)
    print(f"   Total: {total} imágenes")

if __name__ == "__main__":
    print("=" * 50)
    print("🍄 SISTEMA DE REENTRENAMIENTO DE HONGOS")
    print("=" * 50)
    
    # 1. Mostrar estado actual
    listar_estado_dataset()
    
    # 2. Recolectar especies sugeridas (solo agrega nuevas)
    print("\n📥 Recolectando especies sugeridas por usuarios...")
    especies = recolectar_especies_para_reentrenar(min_sugerencias=3)
    
    if especies:
        print(f"\n📊 Se encontraron {len(especies)} especies con suficientes sugerencias")
        
        # 3. Preguntar si reentrenar
        respuesta = input("\n¿Reentrenar modelo con las nuevas imágenes? (s/n): ")
        if respuesta.lower() == 's':
            reentrenar_modelo()
        else:
            print("❌ Reentrenamiento cancelado")
    else:
        print("\n❌ No hay suficientes sugerencias para reentrenar")
        print("   Se necesitan al menos 3 sugerencias de la misma especie")