from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import secrets

from app import schemas, crud, models
from app.database import get_db
from app.utils.email_service import send_password_reset_email

router = APIRouter(prefix="/auth", tags=["recuperación de contraseña"])

@router.post("/forgot-password", response_model=schemas.PasswordResetResponse)
async def forgot_password(
    request: schemas.ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Solicitar recuperación de contraseña"""
    
    # Verificar si el usuario existe
    user = crud.get_user_by_email(db, request.email)
    if not user:
        # Por seguridad, no revelar si el email existe o no
        return {
            "message": "Si el email existe, recibirás un enlace para recuperar tu contraseña",
            "success": True
        }
    
    # Crear token de recuperación
    token = crud.create_password_reset_token(db, request.email)
    
    # Enviar email en segundo plano
    background_tasks.add_task(
        send_password_reset_email,
        email_to=request.email,
        token=token,
        username=user.full_name
    )
    
    return {
        "message": "Si el email existe, recibirás un enlace para recuperar tu contraseña",
        "success": True
    }

@router.post("/verify-reset-token", response_model=schemas.PasswordResetResponse)
async def verify_reset_token(
    request: schemas.VerifyTokenRequest,
    db: Session = Depends(get_db)
):
    """Verificar si un token de recuperación es válido"""
    
    reset_token = crud.verify_password_reset_token(db, request.token)
    
    if not reset_token:
        raise HTTPException(
            status_code=400,
            detail="Token inválido o expirado"
        )
    
    return {
        "message": "Token válido",
        "success": True
    }

@router.post("/reset-password", response_model=schemas.PasswordResetResponse)
async def reset_password(
    request: schemas.ResetPasswordRequest,
    db: Session = Depends(get_db)
):
    """Restablecer contraseña con token"""
    
    # Verificar token
    reset_token = crud.verify_password_reset_token(db, request.token)
    
    if not reset_token:
        raise HTTPException(
            status_code=400,
            detail="Token inválido o expirado"
        )
    
    # Actualizar contraseña
    user = crud.update_user_password(db, reset_token.email, request.new_password)
    
    if not user:
        raise HTTPException(
            status_code=404,
            detail="Usuario no encontrado"
        )
    
    # Marcar token como usado
    crud.use_password_reset_token(db, request.token)
    
    return {
        "message": "Contraseña actualizada exitosamente",
        "success": True
    }