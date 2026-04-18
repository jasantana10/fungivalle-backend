# app/ml/preparar_dataset.py
import os
import shutil
import zipfile
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def preparar_dataset():
    """
    Prepara el dataset de hongos desde un archivo ZIP
    """
    print("🍄 PREPARADOR DE DATASET")
    print("="*50)
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATASET_DIR = BASE_DIR / "app" / "ml" / "dataset"
    
    # Crear directorio si no existe
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Dataset será guardado en: {DATASET_DIR}")
    
    # Buscar archivo ZIP
    zip_path = input("\n📦 Ruta del archivo ZIP del dataset: ").strip()
    
    if not os.path.exists(zip_path):
        print("❌ Archivo no encontrado")
        return
    
    # Extraer ZIP
    print("🔄 Extrayendo archivos...")
    temp_dir = BASE_DIR / "temp_dataset"
    temp_dir.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Buscar carpetas de especies
    especies_encontradas = []
    
    for root, dirs, files in os.walk(temp_dir):
        # Buscar carpetas que contengan imágenes
        imagenes = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if imagenes and len(imagenes) > 5:  # Mínimo 5 imágenes
            nombre_carpeta = os.path.basename(root)
            
            # Preguntar si es una especie
            print(f"\n📁 Encontrada carpeta: {nombre_carpeta}")
            print(f"   Contiene {len(imagenes)} imágenes")
            
            resp = input("   ¿Es una especie de hongo? (s/n): ").strip().lower()
            
            if resp == 's':
                nombre_especie = input("   Nombre científico (ej: Amanita_muscaria): ").strip()
                if nombre_especie:
                    # Crear carpeta destino
                    destino = DATASET_DIR / nombre_especie.replace(" ", "_")
                    destino.mkdir(exist_ok=True)
                    
                    # Copiar imágenes
                    for i, img in enumerate(imagenes[:150]):  # Máx 150 por especie
                        src = os.path.join(root, img)
                        dst = destino / f"{nombre_especie}_{i}.jpg"
                        shutil.copy2(src, dst)
                    
                    especies_encontradas.append(nombre_especie)
                    print(f"   ✅ {len(imagenes[:150])} imágenes copiadas")
    
    # Limpiar
    shutil.rmtree(temp_dir)
    
    print("\n" + "="*50)
    print("✅ PROCESO COMPLETADO")
    print(f"   Especies preparadas: {len(especies_encontradas)}")
    for e in especies_encontradas:
        print(f"   - {e}")
    print(f"   Ubicación: {DATASET_DIR}")
    print("="*50)

if __name__ == "__main__":
    preparar_dataset()