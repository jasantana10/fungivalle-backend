# app/routes/profile.py

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
import base64

from app import schemas, crud, models
from app.database import get_db
from app.auth import get_current_user
from app.utils.profile_images import save_profile_image, delete_profile_image

router = APIRouter(prefix="/profile", tags=["perfil"])


@router.get("/me", response_model=schemas.UserProfileResponse)
async def get_my_profile(
    current_user: dict = Depends(get_current_user),  # ← Cambiar a dict
    db: Session = Depends(get_db)
):
    """Obtener perfil del usuario actual"""
    # Obtener el usuario completo de la base de datos
    user = db.query(models.User).filter(models.User.id == current_user['id']).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    return user


@router.put("/update", response_model=schemas.UserProfileResponse)
async def update_profile(
    profile_data: schemas.UserProfileUpdate,
    current_user: dict = Depends(get_current_user),  # ← Cambiar a dict
    db: Session = Depends(get_db)
):
    """Actualizar perfil del usuario"""
    
    updated_user = crud.update_user_profile(
        db, 
        user_id=current_user['id'],  # ← Usar ['id']
        profile_data=profile_data.dict(exclude_unset=True)
    )
    
    if not updated_user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    return updated_user


@router.post("/upload-image", response_model=schemas.UploadProfileImageResponse)
async def upload_profile_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),  # ← Cambiar a dict
    db: Session = Depends(get_db)
):
    """Subir imagen de perfil"""
    
    try:
        # Guardar imagen
        image_url = save_profile_image(current_user['id'], file)  # ← Usar ['id']
        
        # Actualizar en base de datos
        updated_user = crud.update_profile_image(db, current_user['id'], image_url)
        
        if not updated_user:
            raise HTTPException(status_code=400, detail="Error al actualizar imagen")
        
        return {
            "message": "Imagen de perfil actualizada",
            "image_url": image_url,
            "success": True
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar imagen: {str(e)}")


@router.delete("/remove-image")
async def remove_profile_image(
    current_user: dict = Depends(get_current_user),  # ← Cambiar a dict
    db: Session = Depends(get_db)
):
    """Eliminar imagen de perfil"""
    
    # Obtener el usuario completo
    user = db.query(models.User).filter(models.User.id == current_user['id']).first()
    
    if not user or not user.profile_image:
        raise HTTPException(status_code=400, detail="No hay imagen de perfil")
    
    # Eliminar archivo físico
    delete_profile_image(user.profile_image)
    
    # Actualizar en base de datos
    user.profile_image = None
    db.commit()
    
    return {"message": "Imagen de perfil eliminada", "success": True}


@router.post("/change-password")
async def change_password(
    request: schemas.ChangePasswordRequest,
    current_user: dict = Depends(get_current_user),  # ← Cambiar a dict
    db: Session = Depends(get_db)
):
    """Cambiar contraseña"""
    
    user, message = crud.change_user_password(
        db,
        user_id=current_user['id'],  # ← Usar ['id']
        current_password=request.current_password,
        new_password=request.new_password
    )
    
    if not user:
        raise HTTPException(status_code=400, detail=message)
    
    return {"message": message, "success": True}


@router.post("/change-email")
async def change_email(
    request: schemas.ChangeEmailRequest,
    current_user: dict = Depends(get_current_user),  # ← Cambiar a dict
    db: Session = Depends(get_db)
):
    """Cambiar email"""
    
    user, message = crud.change_user_email(
        db,
        user_id=current_user['id'],  # ← Usar ['id']
        new_email=request.new_email,
        password=request.password
    )
    
    if not user:
        raise HTTPException(status_code=400, detail=message)
    
    # TODO: Enviar email de verificación al nuevo correo
    
    return {"message": message, "success": True}


@router.post("/toggle-dark-mode")
async def toggle_dark_mode(
    dark_mode: bool = Form(...),
    current_user: dict = Depends(get_current_user),  # ← Cambiar a dict
    db: Session = Depends(get_db)
):
    """Activar/desactivar modo oscuro"""
    
    updated_user = crud.update_user_profile(
        db,
        user_id=current_user['id'],  # ← Usar ['id']
        profile_data={"dark_mode": dark_mode}
    )
    
    if not updated_user:
        raise HTTPException(status_code=400, detail="Error al actualizar modo oscuro")
    
    return {
        "message": "Modo oscuro actualizado",
        "dark_mode": dark_mode,
        "success": True
    }