from sqlalchemy.orm import Session
from sqlalchemy import desc
from sqlalchemy import and_
from typing import List, Optional
from datetime import datetime
from datetime import datetime, timedelta
import base64
import os
import secrets
from uuid import uuid4

from app import models, schemas
from app.auth import get_password_hash

# User CRUD
def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    
    # Encriptar respuesta de seguridad si se proporciona
    hashed_answer = None
    if user.security_answer:
        hashed_answer = get_password_hash(user.security_answer)
    
    db_user = models.User(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password,
        security_question=user.security_question,
        security_answer=hashed_answer
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def get_user_by_id(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

# Fungi Species CRUD
def create_fungi_species(db: Session, species: schemas.FungiSpeciesCreate):
    db_species = models.FungiSpecies(**species.dict())
    db.add(db_species)
    db.commit()
    db.refresh(db_species)
    return db_species

def get_fungi_species(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.FungiSpecies).offset(skip).limit(limit).all()

def get_fungi_species_by_id(db: Session, species_id: int):
    return db.query(models.FungiSpecies).filter(models.FungiSpecies.id == species_id).first()

def search_fungi_species(db: Session, query: str):
    return db.query(models.FungiSpecies).filter(
        (models.FungiSpecies.scientific_name.ilike(f"%{query}%")) |
        (models.FungiSpecies.common_name.ilike(f"%{query}%")) |
        (models.FungiSpecies.local_name.ilike(f"%{query}%"))
    ).all()

# Fungi Findings CRUD
def create_fungi_finding(db: Session, finding: schemas.FungiFindingCreate, user_id: int):
    # Guardar imagen si viene en base64
    image_url = None
    if finding.image_base64:
        try:
            # Extraer datos base64
            if "," in finding.image_base64:
                header, data = finding.image_base64.split(",", 1)
            else:
                data = finding.image_base64
            
            image_data = base64.b64decode(data)
            filename = f"fungi_{uuid4().hex}.jpg"
            
            # Guardar en carpeta uploads
            upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
            fungi_dir = os.path.join(upload_dir, "fungi_images")
            os.makedirs(fungi_dir, exist_ok=True)
            
            filepath = os.path.join(fungi_dir, filename)
            with open(filepath, "wb") as f:
                f.write(image_data)
            
            image_url = f"/uploads/fungi_images/{filename}"
        except Exception as e:
            print(f"Error al guardar imagen: {e}")
    
    db_finding = models.FungiFinding(
        user_id=user_id,
        species_id=finding.species_id,
        latitude=finding.latitude,
        longitude=finding.longitude,
        location_name=finding.location_name,
        user_notes=finding.user_notes,
        image_url=image_url,
        date_found=datetime.utcnow()
    )
    
    db.add(db_finding)
    db.commit()
    db.refresh(db_finding)
    return db_finding

def get_user_findings(db: Session, user_id: int, skip: int = 0, limit: int = 50):
    return db.query(models.FungiFinding).filter(
        models.FungiFinding.user_id == user_id
    ).order_by(desc(models.FungiFinding.created_at)).offset(skip).limit(limit).all()

def get_all_findings(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.FungiFinding).order_by(
        desc(models.FungiFinding.created_at)
    ).offset(skip).limit(limit).all()

def get_finding_by_id(db: Session, finding_id: int):
    return db.query(models.FungiFinding).filter(models.FungiFinding.id == finding_id).first()

# Función temporal para identificación (sin IA todavía)
def identify_fungi_from_image(db: Session, image_data: bytes):
    # TODO: Integrar modelo de IA aquí
    # Por ahora, busca especies similares por nombre común
    
    # Ejemplo: si la imagen tiene "amanita" en el nombre
    possible_species = db.query(models.FungiSpecies).filter(
        models.FungiSpecies.common_name.ilike("%amanita%")
    ).first()
    
    if possible_species:
        return {
            "species_id": possible_species.id,
            "confidence_score": 0.75,
            "suggested_name": possible_species.scientific_name
        }
    
    return {
        "species_id": None,
        "confidence_score": 0.0,
        "suggested_name": "Especie no identificada"
    }

# Password Reset CRUD functions
def create_password_reset_token(db: Session, email: str):
    """Crea un token de recuperación de contraseña"""
    
    # Eliminar tokens anteriores para este email
    db.query(models.PasswordResetToken).filter(
        models.PasswordResetToken.email == email
    ).delete()
    
    # Crear nuevo token
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=1)
    
    reset_token = models.PasswordResetToken(
        email=email,
        token=token,
        expires_at=expires_at
    )
    
    db.add(reset_token)
    db.commit()
    db.refresh(reset_token)
    
    return token

def verify_password_reset_token(db: Session, token: str):
    """Verifica si un token es válido"""
    reset_token = db.query(models.PasswordResetToken).filter(
        and_(
            models.PasswordResetToken.token == token,
            models.PasswordResetToken.expires_at > datetime.utcnow(),
            models.PasswordResetToken.is_used == False
        )
    ).first()
    
    return reset_token

def use_password_reset_token(db: Session, token: str):
    """Marca un token como usado"""
    reset_token = db.query(models.PasswordResetToken).filter(
        models.PasswordResetToken.token == token
    ).first()
    
    if reset_token:
        reset_token.is_used = True
        db.commit()
    
    return reset_token

def update_user_password(db: Session, email: str, new_password: str):
    """Actualiza la contraseña de un usuario"""
    user = get_user_by_email(db, email)
    if user:
        user.hashed_password = get_password_hash(new_password)
        user.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(user)
    return user

# Profile CRUD functions
def update_user_profile(db: Session, user_id: int, profile_data: dict):
    """Actualiza el perfil del usuario"""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    
    if not user:
        return None
    
    # Actualizar campos permitidos
    allowed_fields = ['full_name', 'phone', 'bio', 'location', 'dark_mode']
    
    for field in allowed_fields:
        if field in profile_data and profile_data[field] is not None:
            setattr(user, field, profile_data[field])
    
    user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)
    
    return user

def update_profile_image(db: Session, user_id: int, image_url: str):
    """Actualiza la imagen de perfil del usuario"""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    
    if not user:
        return None
    
    # Eliminar imagen anterior si existe
    if user.profile_image:
        from app.utils.profile_images import delete_profile_image
        delete_profile_image(user.profile_image)
    
    user.profile_image = image_url
    user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)
    
    return user

def change_user_password(db: Session, user_id: int, current_password: str, new_password: str):
    """Cambia la contraseña del usuario"""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    
    if not user:
        return None, "Usuario no encontrado"
    
    # Verificar contraseña actual
    if not verify_password(current_password, user.hashed_password):
        return None, "Contraseña actual incorrecta"
    
    # Actualizar contraseña
    user.hashed_password = get_password_hash(new_password)
    user.updated_at = datetime.utcnow()
    db.commit()
    
    return user, "Contraseña actualizada exitosamente"

def change_user_email(db: Session, user_id: int, new_email: str, password: str):
    """Cambia el email del usuario"""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    
    if not user:
        return None, "Usuario no encontrado"
    
    # Verificar contraseña
    if not verify_password(password, user.hashed_password):
        return None, "Contraseña incorrecta"
    
    # Verificar si el nuevo email ya existe
    existing_user = get_user_by_email(db, new_email)
    if existing_user and existing_user.id != user_id:
        return None, "El email ya está en uso"
    
    # Actualizar email
    user.email = new_email
    user.email_verified = False  # Requiere nueva verificación
    user.updated_at = datetime.utcnow()
    db.commit()
    
    return user, "Email actualizado exitosamente. Por favor verifica tu nuevo email."

# Importar verify_password si no existe
from app.auth import get_password_hash, verify_password

# ====== FUNCIONES PARA PREGUNTA DE SEGURIDAD ======

def set_user_security_question(db: Session, user_id: int, question: str, answer: str, current_password: str):
    """Establecer pregunta de seguridad para un usuario"""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    
    if not user:
        return None, "Usuario no encontrado"
    
    # Verificar contraseña actual
    if not verify_password(current_password, user.hashed_password):
        return None, "Contraseña actual incorrecta"
    
    # Encriptar la respuesta
    hashed_answer = get_password_hash(answer)
    
    # Actualizar campos
    user.security_question_id = question
    user.security_answer = hashed_answer
    db.commit()
    db.refresh(user)
    
    return user, "Pregunta de seguridad establecida correctamente"

def get_user_security_question(db: Session, email: str):
    """Obtener pregunta de seguridad de un usuario"""
    user = db.query(models.User).filter(models.User.email == email).first()
    
    if not user:
        return None
    
    return {
        "security_question": user.security_question_id,
        "has_question_set": bool(user.security_question_id and user.security_answer)
    }

def verify_security_answer(db: Session, email: str, answer: str):
    """Verificar respuesta de seguridad"""
    user = db.query(models.User).filter(models.User.email == email).first()
    
    if not user or not user.security_answer:
        return False, "Usuario no encontrado o no tiene pregunta configurada"
    
    # Verificar respuesta
    if verify_password(answer, user.security_answer):
        return True, "Respuesta correcta"
    else:
        return False, "Respuesta incorrecta"

def reset_password_with_security(db: Session, email: str, answer: str, new_password: str):
    """Restablecer contraseña usando pregunta de seguridad"""
    user = db.query(models.User).filter(models.User.email == email).first()
    
    if not user or not user.security_answer:
        return None, "Usuario no encontrado o no tiene pregunta configurada"
    
    # Verificar respuesta
    if not verify_password(answer, user.security_answer):
        return None, "Respuesta incorrecta"
    
    # Actualizar contraseña
    user.hashed_password = get_password_hash(new_password)
    user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)
    
    return user, "Contraseña actualizada exitosamente"

def create_user_with_security(db: Session, user: schemas.UserCreate):
    """Crear usuario con pregunta de seguridad opcional"""
    hashed_password = get_password_hash(user.password)
    
    # Encriptar respuesta de seguridad si se proporciona
    hashed_answer = None
    if user.security_answer:
        hashed_answer = get_password_hash(user.security_answer)
    
    db_user = models.User(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password,
        security_question=user.security_question_id,
        security_answer=hashed_answer
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# ====== CRUD PARA PREGUNTAS DE SEGURIDAD ======

def get_security_questions(db: Session, skip: int = 0, limit: int = 20):
    """Obtener lista de preguntas de seguridad activas"""
    try:
        from app.models import SecurityQuestion
        
        # Versión SIN filtro para que funcione YA
        preguntas = db.query(SecurityQuestion).offset(skip).limit(limit).all()
        
        print(f"✅ Preguntas encontradas: {len(preguntas)}")
        return preguntas
        
    except Exception as e:
        print(f"❌ Error en get_security_questions: {e}")
        return []

def get_security_question_by_id(db: Session, question_id: int):
    """Obtener pregunta de seguridad por ID"""
    return db.query(models.SecurityQuestion).filter(
        models.SecurityQuestion.id == question_id
    ).first()

def set_user_security_question(db: Session, user_id: int, question_id: int, answer: str, current_password: str):
    """Establecer pregunta de seguridad para un usuario"""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    
    if not user:
        return None, "Usuario no encontrado"
    
    # Verificar contraseña actual
    if not verify_password(current_password, user.hashed_password):
        return None, "Contraseña actual incorrecta"
    
    # Verificar que la pregunta existe
    question = get_security_question_by_id(db, question_id)
    if not question or not question.is_active:
        return None, "Pregunta de seguridad no válida"
    
    # Encriptar la respuesta
    hashed_answer = get_password_hash(answer)
    
    # Actualizar campos
    user.security_question_id = question_id
    user.security_answer = hashed_answer
    db.commit()
    db.refresh(user)
    
    return user, "Pregunta de seguridad establecida correctamente"

def get_user_security_question_info(db: Session, email: str):
    """Obtener información de pregunta de seguridad de un usuario"""
    user = db.query(models.User).filter(models.User.email == email).first()
    
    if not user or not user.security_question_id:
        return None
    
    # Obtener el texto de la pregunta
    question = get_security_question_by_id(db, user.security_question_id)
    
    return {
        "question_id": user.security_question_id,
        "question_text": question.question_text if question else None,
        "has_question_set": bool(user.security_question_id and user.security_answer)
    }

def verify_security_answer(db: Session, email: str, answer: str):
    """Verificar respuesta de seguridad"""
    user = db.query(models.User).filter(models.User.email == email).first()
    
    if not user or not user.security_answer:
        return False, "Usuario no encontrado o no tiene pregunta configurada"
    
    # Verificar respuesta
    if verify_password(answer, user.security_answer):
        return True, "Respuesta correcta"
    else:
        return False, "Respuesta incorrecta"

def reset_password_with_security(db: Session, email: str, answer: str, new_password: str):
    """Restablecer contraseña usando pregunta de seguridad"""
    user = db.query(models.User).filter(models.User.email == email).first()
    
    if not user or not user.security_answer:
        return None, "Usuario no encontrado o no tiene pregunta configurada"
    
    # Verificar respuesta
    if not verify_password(answer, user.security_answer):
        return None, "Respuesta incorrecta"
    
    # Actualizar contraseña
    user.hashed_password = get_password_hash(new_password)
    user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(user)
    
    return user, "Contraseña actualizada exitosamente"

# Modificar la función create_user para aceptar pregunta
def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    
    # Verificar si se proporciona pregunta de seguridad
    security_question_id = None
    hashed_answer = None
    
    if user.security_answer and user.security_question_id:
        # Verificar que la pregunta existe y está activa
        question = get_security_question_by_id(db, user.security_question_id)
        if not question or not question.is_active:
            raise ValueError("Pregunta de seguridad no válida")
        
        security_question_id = user.security_question_id
        hashed_answer = get_password_hash(user.security_answer)
    
    db_user = models.User(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password,
        security_question_id=security_question_id,
        security_answer=hashed_answer
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user