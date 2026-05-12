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
from PIL import Image
import uuid
import shutil
import os

router = APIRouter()

UPLOAD_DIR = Path("uploads/findings")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Cargar modelo para extraer características (embedding) - VERSIÓN LIGERA SIN TENSORFLOW
def extract_simple_embedding(image_path):
    """Extrae vector de características simple sin TensorFlow"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((64, 64))  # Tamaño pequeño para embedding ligero
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array.flatten().tolist()  # Vector de 64*64*3 = 12288 dimensiones
    except Exception as e:
        print(f"❌ Error extrayendo embedding simple: {e}")
        return [0.0] * 12288

def extract_embedding(image_path):
    """Extrae vector de características de la imagen (usa método simple para reducir RAM)"""
    return extract_simple_embedding(image_path)

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
        
        # Extraer embedding (ligero, sin TensorFlow)
        embedding = extract_embedding(file_path)
        
        # Buscar especie por nombre
        species = db.query(FungiSpecies).filter(
            (FungiSpecies.scientific_name == final_species_name) | 
            (FungiSpecies.common_name == final_species_name)
        ).first()
        
        species_id = species.id if species else None
        
        # Crear el hallazgo usando SQLAlchemy
        new_finding = FungiFinding(
            user_id=user_id,
            species_id=species_id,
            species_name=final_species_name,
            confidence_score=confidence_score,
            user_notes=user_notes,
            user_suggestion=user_suggestion,
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            image_url=image_url,
            embedding=json.dumps(embedding),  # Guardar como JSON
            is_verified=0,
            date_found=datetime.now()
        )
        
        db.add(new_finding)
        db.commit()
        db.refresh(new_finding)
        
        # Guardar embedding en tabla separada (opcional, para compatibilidad)
        new_embedding = ImageEmbedding(
            finding_id=new_finding.id,
            embedding=json.dumps(embedding),
            species_name=final_species_name
        )
        db.add(new_embedding)
        db.commit()
        
        return {
            "success": True,
            "message": "Hallazgo guardado exitosamente",
            "finding_id": new_finding.id,
            "image_url": image_url
        }
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error guardando hallazgo: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.get("/usuario/{user_id}")
def get_hallazgos_usuario(
    user_id: int, 
    skip: int = 0, 
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Obtiene todos los hallazgos de un usuario"""
    findings = db.query(FungiFinding).filter(
        FungiFinding.user_id == user_id
    ).order_by(FungiFinding.created_at.desc()).offset(skip).limit(limit).all()
    
    return {
        "success": True,
        "findings": findings
    }

@router.get("/{finding_id}")
def get_hallazgo(finding_id: int, db: Session = Depends(get_db)):
    """Obtiene un hallazgo por su ID"""
    finding = db.query(FungiFinding).filter(FungiFinding.id == finding_id).first()
    
    if not finding:
        raise HTTPException(status_code=404, detail="Hallazgo no encontrado")
    
    return {
        "success": True,
        "finding": finding
    }
