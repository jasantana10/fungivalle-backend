from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import base64

from app import schemas, crud
from app.database import get_db
from app.auth import get_current_user
from app.models import User

router = APIRouter(prefix="/fungi", tags=["hongos"])

# Especies
@router.get("/species", response_model=List[schemas.FungiSpeciesResponse])
def get_species(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return crud.get_fungi_species(db, skip=skip, limit=limit)

@router.get("/species/{species_id}", response_model=schemas.FungiSpeciesResponse)
def get_species_by_id(species_id: int, db: Session = Depends(get_db)):
    species = crud.get_fungi_species_by_id(db, species_id=species_id)
    if not species:
        raise HTTPException(status_code=404, detail="Especie no encontrada")
    return species

@router.get("/species/search", response_model=List[schemas.FungiSpeciesResponse])
def search_species(query: str, db: Session = Depends(get_db)):
    if len(query) < 2:
        raise HTTPException(status_code=400, detail="La búsqueda debe tener al menos 2 caracteres")
    return crud.search_fungi_species(db, query=query)

# Identificación
@router.post("/identify", response_model=schemas.IdentificationResponse)
async def identify_fungi(
    file: UploadFile = File(None),
    image_base64: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    image_data = None
    
    if file:
        # Leer imagen del archivo
        image_data = await file.read()
    elif image_base64:
        # Decodificar base64
        try:
            if "," in image_base64:
                header, data = image_base64.split(",", 1)
            else:
                data = image_base64
            image_data = base64.b64decode(data)
        except:
            raise HTTPException(status_code=400, detail="Formato base64 inválido")
    else:
        raise HTTPException(status_code=400, detail="Se requiere una imagen")
    
    # Identificar con IA
    result = crud.identify_fungi_from_image(db, image_data)
    
    return schemas.IdentificationResponse(
        success=True,
        species_id=result.get("species_id"),
        confidence_score=result.get("confidence_score"),
        suggested_name=result.get("suggested_name"),
        message="Identificación completada"
    )

# Hallazgos
@router.post("/findings", response_model=schemas.FungiFindingResponse)
def create_finding(
    finding: schemas.FungiFindingCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return crud.create_fungi_finding(db=db, finding=finding, user_id=current_user.id)

@router.get("/findings/my", response_model=List[schemas.FungiFindingResponse])
def get_my_findings(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    findings = crud.get_user_findings(db, user_id=current_user.id, skip=skip, limit=limit)
    return findings

@router.get("/findings/recent", response_model=List[schemas.FungiFindingResponse])
def get_recent_findings(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    return crud.get_all_findings(db, skip=skip, limit=limit)

@router.get("/findings/{finding_id}", response_model=schemas.FungiFindingResponse)
def get_finding_by_id(
    finding_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    finding = crud.get_finding_by_id(db, finding_id=finding_id)
    if not finding:
        raise HTTPException(status_code=404, detail="Hallazgo no encontrado")
    if finding.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="No autorizado")
    return finding