from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app import schemas
from app.database import get_db
from app import crud

router = APIRouter(prefix="/security", tags=["pregunta de seguridad"])

@router.get("/questions", response_model=schemas.AvailableQuestionsResponse)
async def get_security_questions(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Obtener lista de preguntas de seguridad disponibles"""
    questions = crud.get_security_questions(db, skip=skip, limit=limit)
    
    return {
        "questions": questions,
        "total": len(questions)
    }

@router.post("/set-question", response_model=schemas.PasswordResetResponse)
async def set_security_question(
    request: schemas.SetSecurityQuestion,
    db: Session = Depends(get_db)
):
    """Establecer o actualizar pregunta de seguridad"""
    
    user, message = crud.set_user_security_question(
        db, 
        user_id=request.user_id,
        question_id=request.security_question_id,
        answer=request.security_answer,
        current_password=request.current_password
    )
    
    if not user:
        raise HTTPException(status_code=400, detail=message)
    
    return {
        "message": message,
        "success": True
    }

@router.get("/get-question/{email}", response_model=schemas.UserSecurityInfo)
async def get_security_question(
    email: str,
    db: Session = Depends(get_db)
):
    """Obtener pregunta de seguridad de un usuario"""
    
    result = crud.get_user_security_question_info(db, email)
    
    if not result:
        # Por seguridad, no revelar si el usuario existe
        return {
            "security_question_id": None,
            "security_question_text": None,
            "has_security_question": False
        }
    
    return {
        "security_question_id": result["question_id"],
        "security_question_text": result["question_text"],
        "has_security_question": result["has_question_set"]
    }

@router.post("/verify-answer", response_model=schemas.PasswordResetResponse)
async def verify_security_answer(
    request: schemas.VerifySecurityAnswer,
    db: Session = Depends(get_db)
):
    """Verificar respuesta de seguridad"""
    
    is_valid, message = crud.verify_security_answer(
        db, 
        email=request.email,
        answer=request.security_answer
    )
    
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)
    
    return {
        "message": "Respuesta verificada correctamente",
        "success": True
    }

@router.post("/reset-password", response_model=schemas.PasswordResetResponse)
async def reset_password_with_question(
    request: schemas.ResetPasswordWithQuestion,
    db: Session = Depends(get_db)
):
    """Restablecer contraseña usando pregunta de seguridad"""
    
    user, message = crud.reset_password_with_security(
        db,
        email=request.email,
        answer=request.security_answer,
        new_password=request.new_password
    )
    
    if not user:
        raise HTTPException(status_code=400, detail=message)
    
    return {
        "message": message,
        "success": True
    }

@router.get("/user-has-question/{user_id}", response_model=schemas.UserSecurityInfo)
async def check_user_has_security_question(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Verificar si un usuario tiene pregunta de seguridad configurada"""
    
    user = crud.get_user_by_id(db, user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    question_text = None
    if user.security_question_id and user.security_question:
        question_text = user.security_question.question_text
    
    return {
        "security_question_id": user.security_question_id,
        "security_question_text": question_text,
        "has_security_question": bool(user.security_question_id and user.security_answer)
    }