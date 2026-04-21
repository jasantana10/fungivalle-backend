# app/models.py

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from pydantic import BaseModel
from typing import Optional
from app.database import Base


# ========== MODELOS DE LA BASE DE DATOS ==========

class SecurityQuestion(Base):
    __tablename__ = "security_questions"
    
    id = Column(Integer, primary_key=True, index=True)
    question_text = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relaciones
    users = relationship("User", back_populates="security_question")


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    security_question_id = Column(Integer, ForeignKey("security_questions.id"), nullable=True)
    security_answer = Column(String(255), nullable=True)
    profile_image = Column(String(255), nullable=True)
    phone = Column(String(20), nullable=True)
    bio = Column(Text, nullable=True)
    location = Column(String(100), nullable=True)
    dark_mode = Column(Boolean, default=False)
    email_verified = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False) 
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    findings = relationship("FungiFinding", back_populates="user", cascade="all, delete-orphan")
    security_question = relationship("SecurityQuestion", back_populates="users")


class FungiSpecies(Base):
    __tablename__ = "fungi_species"
    
    id = Column(Integer, primary_key=True, index=True)
    scientific_name = Column(String(100), unique=True, nullable=False)
    common_name = Column(String(100))
    local_name = Column(String(100))
    description = Column(Text)
    habitat = Column(Text)
    season = Column(String(50))
    edible = Column(String(20))
    toxicity_level = Column(String(20))
    image_url = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relaciones
    findings = relationship("FungiFinding", back_populates="species")


class FungiFinding(Base):
    __tablename__ = "fungi_findings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    species_id = Column(Integer, ForeignKey("fungi_species.id"), nullable=True)
    species_name = Column(String(150))
    confidence_score = Column(Float)
    user_notes = Column(Text)
    user_suggestion = Column(String(150))
    location_name = Column(String(200))
    latitude = Column(Float)
    longitude = Column(Float)
    image_url = Column(String(500))
    embedding = Column(Text)
    is_verified = Column(Integer, default=0)
    date_found = Column(DateTime, default=datetime.now)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relaciones
    user = relationship("User", back_populates="findings")
    species = relationship("FungiSpecies", back_populates="findings")


class ImageEmbedding(Base):
    __tablename__ = "image_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    finding_id = Column(Integer, ForeignKey("fungi_findings.id"), nullable=False)
    embedding = Column(Text)
    species_name = Column(String(150))
    created_at = Column(DateTime, default=datetime.now)
    
    # Relaciones
    finding = relationship("FungiFinding", backref="embedding_ref", uselist=False)


class UserSpecies(Base):
    __tablename__ = "user_species"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(150), unique=True)
    vote_count = Column(Integer, default=1)
    first_seen_at = Column(DateTime, default=datetime.now)
    last_seen_at = Column(DateTime, default=datetime.now)
    status = Column(String(20), default="pending")


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), nullable=False, index=True)
    token = Column(String(255), unique=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    is_used = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# ========== PYDANTIC MODELS PARA API ==========

class FungiFindingBase(BaseModel):
    user_id: int
    species_id: Optional[int] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_name: Optional[str] = None
    date_found: Optional[datetime] = None
    confidence_score: float
    user_notes: Optional[str] = None
    image_url: str
    is_verified: bool = False

    class Config:
        from_attributes = True


class FungiFindingCreate(FungiFindingBase):
    pass


class FungiFindingResponse(FungiFindingBase):
    id: int
    created_at: Optional[datetime] = None
    species_name: Optional[str] = None
    user_suggestion: Optional[str] = None


class UserBase(BaseModel):
    email: str
    full_name: str

    class Config:
        from_attributes = True


class UserCreate(UserBase):
    password: str
    security_question_id: Optional[int] = None
    security_answer: Optional[str] = None


class UserResponse(UserBase):
    id: int
    profile_image: Optional[str] = None
    phone: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    is_active: bool
    created_at: datetime


class FungiSpeciesBase(BaseModel):
    scientific_name: str
    common_name: Optional[str] = None
    local_name: Optional[str] = None
    description: Optional[str] = None
    habitat: Optional[str] = None
    season: Optional[str] = None
    edible: Optional[str] = None
    toxicity_level: Optional[str] = None

    class Config:
        from_attributes = True


class FungiSpeciesResponse(FungiSpeciesBase):
    id: int
    image_url: Optional[str] = None
    created_at: datetime