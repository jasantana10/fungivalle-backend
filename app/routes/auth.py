from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.auth import get_current_user
from app import models

from app import schemas, crud
from app.database import get_db
from app.auth import authenticate_user, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter(prefix="/auth", tags=["autenticación"])

@router.post("/register", response_model=schemas.UserResponse)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Verificar si el usuario ya existe
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="El correo ya está registrado"
        )
    
    # Validar pregunta de seguridad
    if user.security_answer and not user.security_question_id:
        raise HTTPException(
            status_code=400,
            detail="Debe seleccionar una pregunta de seguridad si proporciona una respuesta"
        )
    
    if user.security_question_id and not user.security_answer:
        raise HTTPException(
            status_code=400,
            detail="Debe proporcionar una respuesta de seguridad si selecciona una pregunta"
        )
    
    # Si proporciona pregunta, verificar que exista
    if user.security_question_id:
        question = crud.get_security_question_by_id(db, user.security_question_id)
        if not question or not question.is_active:
            raise HTTPException(
                status_code=400,
                detail="Pregunta de seguridad no válida"
            )
    
    # Crear usuario
    try:
        created_user = crud.create_user(db=db, user=user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return created_user

@router.post("/login", response_model=schemas.Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Autenticar usuario
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Correo o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Crear token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=schemas.UserResponse)
def read_users_me(
    current_user: models.User = Depends(get_current_user)
):
    return current_user
