# app/routes/fungi_findings.py

from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
from typing import Optional, List
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import FungiFinding, ImageEmbedding, UserSpecies, FungiSpecies
from app.ml.data_augmentation import augment_images_for_species
import uuid
import shutil
import os
import subprocess
import sys
import importlib
import app.routes.hongos as hongos_module

router = APIRouter()

UPLOAD_DIR = Path("uploads/findings")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Cargar modelo para extraer características (embedding)
embedding_model = None

def load_embedding_model():
    global embedding_model
    if embedding_model is None:
        try:
            import tensorflow as tf
            base_model = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3),
                pooling='avg'
            )
            embedding_model = base_model
            print("✅ Modelo de embeddings cargado (TensorFlow)")
        except ImportError:
            print("⚠️ TensorFlow no instalado. Embeddings deshabilitados.")
            return None
    return embedding_model

def extract_embedding(image_path):
    """Extrae vector de características de la imagen"""
    model = load_embedding_model()
    if model is None:
        # Devolver un vector de ceros si no hay modelo (MobileNetV2 tiene 1280 dimensiones)
        return [0.0] * 1280

    try:
        import tensorflow as tf
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        embedding = model.predict(img_array, verbose=0)[0]
        return embedding.tolist()
    except Exception as e:
        print(f"❌ Error extrayendo embedding: {e}")
        return [0.0] * 1280


@router.post("/guardar")
async def guardar_hallazgo(
    file: UploadFile = File(...),
    species_name: str = Form(...),
    confidence_score: float = Form(...),
    user_id: int = Form(...),
    user_notes: Optional[str] = Form(None),
    user_suggestion: Optional[str] = Form(None),
    location_name: Optional[str] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Guarda un hallazgo en la tabla fungi_findings
    """
    try:
        # Validar que el archivo sea una imagen
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        # Guardar la imagen
        file_extension = file.filename.split('.')[-1]
        file_name = f"{uuid.uuid4()}.{file_extension}"
        file_path = UPLOAD_DIR / file_name
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        image_url = f"/uploads/findings/{file_name}"
        
        # Determinar el nombre final de la especie
        final_species_name = species_name
        if user_suggestion and species_name == "Desconocido":
            final_species_name = user_suggestion
        
        # Extraer embedding
        embedding = extract_embedding(file_path)
        
        # Crear el hallazgo usando SQLAlchemy
        new_finding = FungiFinding(
            user_id=user_id,
            species_name=final_species_name,
            confidence_score=confidence_score,
            user_notes=user_notes,
            user_suggestion=user_suggestion,
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            image_url=image_url,
            embedding=json.dumps(embedding),
            date_found=datetime.now(),
            created_at=datetime.now()
        )
        
        db.add(new_finding)
        db.commit()
        db.refresh(new_finding)
        
        print(f"✅ Hallazgo guardado con ID: {new_finding.id}")
        print(f"   Especie: {final_species_name}")
        print(f"   Usuario: {user_id}")
        print(f"   Confianza: {confidence_score}")
        
        return {
            "success": True,
            "id": new_finding.id,
            "species_name": final_species_name,
            "message": "Hallazgo guardado exitosamente"
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recientes")
async def obtener_hallazgos_recientes(
    user_id: int,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Obtiene los hallazgos más recientes del usuario
    """
    try:
        findings = db.query(FungiFinding).filter(
            FungiFinding.user_id == user_id
        ).order_by(
            FungiFinding.date_found.desc()
        ).limit(limit).all()
        
        # Convertir a diccionario para JSON
        result = []
        for f in findings:
            result.append({
                "id": f.id,
                "user_id": f.user_id,
                "species_name": f.species_name,
                "confidence_score": f.confidence_score,
                "user_notes": f.user_notes,
                "user_suggestion": f.user_suggestion,
                "location_name": f.location_name,
                "latitude": f.latitude,
                "longitude": f.longitude,
                "image_url": f.image_url,
                "date_found": f.date_found.isoformat() if f.date_found else None,
                "created_at": f.created_at.isoformat() if f.created_at else None
            })
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/especies-pendientes")
async def especies_pendientes(
    min_votes: int = 3,
    db: Session = Depends(get_db)
):
    """
    Obtiene especies sugeridas por usuarios (para reentrenamiento)
    """
    try:
        from sqlalchemy import func
        
        # Agrupar por user_suggestion
        results = db.query(
            FungiFinding.user_suggestion,
            func.count(FungiFinding.id).label('count'),
            func.max(FungiFinding.created_at).label('last_seen')
        ).filter(
            FungiFinding.user_suggestion.isnot(None),
            FungiFinding.user_suggestion != ''
        ).group_by(
            FungiFinding.user_suggestion
        ).order_by(
            func.count(FungiFinding.id).desc()
        ).all()
        
        return [
            {
                "user_suggestion": r.user_suggestion,
                "count": r.count,
                "last_seen": r.last_seen.isoformat() if r.last_seen else None
            }
            for r in results
        ]
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/estado-reentrenamiento")
async def estado_reentrenamiento(
    db: Session = Depends(get_db)
):
    """
    Devuelve el estado actual para reentrenamiento
    """
    try:
        from sqlalchemy import func
        from pathlib import Path
        
        # Especies pendientes con sugerencias
        especies = db.query(
            FungiFinding.user_suggestion,
            func.count(FungiFinding.id).label('count')
        ).filter(
            FungiFinding.user_suggestion.isnot(None),
            FungiFinding.user_suggestion != '',
            FungiFinding.user_suggestion != 'Desconocido'
        ).group_by(
            FungiFinding.user_suggestion
        ).order_by(
            func.count(FungiFinding.id).desc()
        ).all()
        
        # Contar imágenes en dataset actual
        dataset_dir = Path("app/ml/dataset")
        total_imagenes = 0
        especies_dataset = []
        
        if dataset_dir.exists():
            for especie_dir in dataset_dir.iterdir():
                if especie_dir.is_dir():
                    count = len(list(especie_dir.glob("*.jpg")))
                    if count > 0:
                        total_imagenes += count
                        especies_dataset.append({
                            "nombre": especie_dir.name,
                            "imagenes": count
                        })
        
        return {
            "especies_pendientes": [
                {"nombre": e[0], "sugerencias": e[1]}
                for e in especies if e[1] >= 3
            ],
            "especies_dataset": especies_dataset,
            "total_imagenes_dataset": total_imagenes,
            "recomendacion": "Listo para reentrenar" if any(e[1] >= 3 for e in especies) else "Faltan sugerencias"
        }
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reentrenar")
async def reentrenar_modelo_admin(
    min_sugerencias: int = 3,
    db: Session = Depends(get_db)
):
    """
    Endpoint para reentrenar el modelo con especies sugeridas.
    Incluye DATA AUGMENTATION para generar 150 imágenes por especie.
    SOLO ADMIN puede ejecutar esto.
    """
    try:
        import subprocess
        import sys
        import shutil
        from pathlib import Path
        from sqlalchemy import func
        
        print("🔄 Iniciando proceso de reentrenamiento con DATA AUGMENTATION...")
        
        # 1. Recolectar especies con suficientes sugerencias
        especies = db.query(
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
        
        if not especies:
            return {
                "success": False,
                "message": f"No hay especies con {min_sugerencias}+ sugerencias",
                "especies_encontradas": 0
            }
        
        print(f"📊 Especies encontradas: {len(especies)}")
        
        # 2. Preparar dataset con DATA AUGMENTATION
        dataset_dir = Path("app/ml/dataset")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        especies_procesadas = []
        total_imagenes_final = 0
        
        for especie, count in especies:
            print(f"\n🍄 Procesando especie: {especie}")
            
            # Carpeta para esta especie en el dataset
            especie_nombre_limpio = especie.replace(" ", "_")
            especie_dir = dataset_dir / especie_nombre_limpio
            especie_dir.mkdir(parents=True, exist_ok=True)
            
            # Obtener imágenes originales de esta especie
            imagenes = db.query(FungiFinding).filter(
                FungiFinding.user_suggestion == especie
            ).all()
            
            if not imagenes:
                print(f"   ⚠️ No hay imágenes para {especie}")
                continue
            
            # Copiar imágenes originales a una carpeta temporal
            temp_dir = especie_dir / "originales_temp"
            temp_dir.mkdir(exist_ok=True)
            
            imagenes_originales = 0
            for img in imagenes:
                if img.image_url:
                    src = Path(".") / img.image_url.lstrip('/')
                    if src.exists():
                        dst = temp_dir / f"{img.id}.jpg"
                        shutil.copy2(src, dst)
                        imagenes_originales += 1
            
            print(f"   📸 Imágenes originales encontradas: {imagenes_originales}")
            
            # Generar imágenes aumentadas (hasta 150 totales)
            imagenes_generadas = augment_images_for_species(
                temp_dir, 
                especie_dir, 
                target_count=150
            )
            
            # Limpiar temporal
            shutil.rmtree(temp_dir)
            
            # Contar imágenes finales
            imagenes_finales = len(list(especie_dir.glob("*.jpg")))
            total_imagenes_final += imagenes_finales
            
            especies_procesadas.append({
                "nombre": especie,
                "sugerencias": count,
                "imagenes_originales": imagenes_originales,
                "imagenes_finales": imagenes_finales,
                "imagenes_generadas": imagenes_finales - imagenes_originales
            })
            
            print(f"   ✅ {especie}: {imagenes_finales} imágenes totales (+{imagenes_finales - imagenes_originales} generadas)")
        
        if total_imagenes_final == 0:
            return {
                "success": False,
                "message": "No se pudieron generar imágenes para el reentrenamiento",
                "especies_procesadas": especies_procesadas
            }
        
        # 3. Verificar si tenemos suficientes imágenes
        print(f"\n📊 Total de imágenes en dataset: {total_imagenes_final}")
        
        if total_imagenes_final < 50:
            return {
                "success": False,
                "message": f"No hay suficientes imágenes para reentrenar. Total: {total_imagenes_final} (mínimo 50)"
            }
        
        # 4. Ejecutar reentrenamiento
        print("🚀 Iniciando reentrenamiento del modelo...")
        
        # Obtener la ruta del script de entrenamiento
        script_path = Path("app/ml/train_model_completo.py")
        
        if not script_path.exists():
            return {
                "success": False,
                "message": f"No se encuentra el script de entrenamiento en {script_path}",
                "especies_procesadas": especies_procesadas,
                "total_imagenes": total_imagenes_final
            }
        
        # Ejecutar el script de entrenamiento
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hora máximo
        )
        
        if result.returncode != 0:
            return {
                "success": False,
                "message": "Error en el reentrenamiento",
                "error": result.stderr,
                "especies_procesadas": especies_procesadas,
                "total_imagenes": total_imagenes_final
            }
        
        return {
            "success": True,
            "message": f"✅ Modelo reentrenado exitosamente con {total_imagenes_final} imágenes",
            "especies_procesadas": especies_procesadas,
            "total_imagenes": total_imagenes_final,
            "output": result.stdout[-1000:]  # Últimas 1000 líneas
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "message": "El reentrenamiento tomó demasiado tiempo (más de 1 hora)"
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
#importlib.reload(hongos_module)
    
@router.get("/diagnostico")
async def diagnosticar_reentrenamiento(
    db: Session = Depends(get_db)
):
    """
    Diagnóstico del sistema de reentrenamiento
    """
    from pathlib import Path
    
    dataset_dir = Path("app/ml/dataset")
    script_path = Path("app/ml/train_model_completo.py")
    modelo_path = Path("app/ml/models/modelo_finetuned.keras")
    
    # Contar especies y imágenes en dataset
    especies_dataset = []
    total_imagenes = 0
    
    if dataset_dir.exists():
        for especie_dir in dataset_dir.iterdir():
            if especie_dir.is_dir():
                count = len(list(especie_dir.glob("*.jpg")))
                if count > 0:
                    total_imagenes += count
                    especies_dataset.append({
                        "nombre": especie_dir.name,
                        "imagenes": count
                    })
    
    return {
        "dataset_existe": dataset_dir.exists(),
        "total_imagenes_dataset": total_imagenes,
        "especies_dataset": especies_dataset,
        "script_entrenamiento_existe": script_path.exists(),
        "modelo_existe": modelo_path.exists(),
        "modelo_path": str(modelo_path) if modelo_path.exists() else None,
        "recomendacion": "Reentrenar" if total_imagenes > 0 else "No hay imágenes"
    }

@router.delete("/eliminar/{finding_id}")
async def eliminar_hallazgo(
    finding_id: int,
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Elimina un hallazgo de la base de datos
    """
    try:
        # Buscar el hallazgo
        finding = db.query(FungiFinding).filter(
            FungiFinding.id == finding_id,
            FungiFinding.user_id == user_id
        ).first()
        
        if not finding:
            raise HTTPException(status_code=404, detail="Hallazgo no encontrado")
        
        # Eliminar la imagen física si existe
        if finding.image_url:
            image_path = Path(".") / finding.image_url.lstrip('/')
            if image_path.exists():
                os.remove(image_path)
                print(f"🗑️ Imagen eliminada: {image_path}")
        
        # Eliminar de la base de datos
        db.delete(finding)
        db.commit()
        
        return {
            "success": True,
            "message": "Hallazgo eliminado exitosamente"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))