# app/schemas.py

from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional, List

# User Schemas
class UserBase(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=3, max_length=100)

class UserCreate(UserBase):
    password: str = Field(..., min_length=6)
    security_question_id: Optional[int] = None
    security_answer: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(..., max_length=72)

class UserResponse(UserBase):
    id: int
    is_active: bool
    email_verified: bool = False
    is_admin: bool = False
    profile_image: Optional[str] = None
    phone: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# Fungi Species Schemas
class FungiSpeciesBase(BaseModel):
    scientific_name: str
    common_name: Optional[str] = None
    local_name: Optional[str] = None
    description: Optional[str] = None
    habitat: Optional[str] = None
    season: Optional[str] = None
    edible: Optional[str] = None
    toxicity_level: Optional[str] = None

class FungiSpeciesCreate(FungiSpeciesBase):
    pass

class FungiSpeciesResponse(FungiSpeciesBase):
    id: int
    image_url: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

# Finding Schemas
class FungiFindingBase(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_name: Optional[str] = None
    user_notes: Optional[str] = None

class FungiFindingCreate(FungiFindingBase):
    species_id: Optional[int] = None
    image_base64: Optional[str] = None

class FungiFindingResponse(FungiFindingBase):
    id: int
    user_id: int
    species_id: Optional[int]
    species_name: Optional[str] = None  # ← AGREGAR para nombres sugeridos
    user_suggestion: Optional[str] = None  # ← AGREGAR
    date_found: datetime
    confidence_score: Optional[float] = None
    image_url: Optional[str] = None
    is_verified: bool
    created_at: datetime
    species: Optional[FungiSpeciesResponse] = None
    
    class Config:
        from_attributes = True

# Identification Response
class IdentificationResponse(BaseModel):
    success: bool
    species_id: Optional[int] = None
    confidence_score: Optional[float] = None
    suggested_name: Optional[str] = None
    message: str

# Password Reset Schemas
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=6)

class VerifyTokenRequest(BaseModel):
    token: str

class PasswordResetResponse(BaseModel):
    message: str
    success: bool

# Profile Schemas
class UserProfileBase(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    dark_mode: Optional[bool] = None

class UserProfileUpdate(UserProfileBase):
    pass

class UserProfileResponse(BaseModel):
    id: int
    email: str
    full_name: str
    profile_image: Optional[str] = None
    phone: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    dark_mode: bool = False
    email_verified: bool = False
    is_admin: bool = False  # ← AGREGAR is_admin
    is_active: bool = True
    created_at: datetime
    
    class Config:
        from_attributes = True

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6)

class ChangeEmailRequest(BaseModel):
    new_email: EmailStr
    password: str

class UploadProfileImageResponse(BaseModel):
    message: str
    image_url: str
    success: bool

# ====== SCHEMAS PARA PREGUNTA DE SEGURIDAD ======

class SecurityQuestionBase(BaseModel):
    question_text: str

class SecurityQuestionCreate(SecurityQuestionBase):
    pass

class SecurityQuestionResponse(SecurityQuestionBase):
    id: int
    is_active: bool
    
    class Config:
        from_attributes = True

# Para establecer pregunta después
class SetSecurityQuestion(BaseModel):
    user_id: int
    security_question_id: int = Field(..., gt=0)
    security_answer: str = Field(..., min_length=2)
    current_password: str

class UserSecurityInfo(BaseModel):
    security_question_id: Optional[int]
    security_question_text: Optional[str]
    has_security_question: bool

# Para recuperación
class SecurityQuestionForUser(BaseModel):
    question_id: int
    question_text: str

class VerifySecurityAnswer(BaseModel):
    email: EmailStr
    security_answer: str

class ResetPasswordWithQuestion(BaseModel):
    email: EmailStr
    security_answer: str
    new_password: str = Field(..., min_length=6)

# Para listar preguntas disponibles
class AvailableQuestionsResponse(BaseModel):
    questions: List[SecurityQuestionResponse]
    total: int