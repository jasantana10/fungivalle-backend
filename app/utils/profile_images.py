import os
import shutil
from uuid import uuid4
from fastapi import UploadFile
from PIL import Image
import io

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
PROFILE_IMAGES_DIR = os.path.join(UPLOAD_DIR, "profile_images")

# Crear directorios si no existen
os.makedirs(PROFILE_IMAGES_DIR, exist_ok=True)

def save_profile_image(user_id: int, image_file: UploadFile) -> str:
    """
    Guarda una imagen de perfil y devuelve la URL
    
    Args:
        user_id: ID del usuario
        image_file: Archivo de imagen subido
    
    Returns:
        URL de la imagen guardada
    """
    
    # Validar tipo de archivo
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    file_ext = os.path.splitext(image_file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise ValueError("Tipo de archivo no permitido")
    
    # Generar nombre único
    filename = f"user_{user_id}_{uuid4().hex}{file_ext}"
    filepath = os.path.join(PROFILE_IMAGES_DIR, filename)
    
    # Leer y procesar imagen
    image_data = image_file.file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Convertir a RGB si es necesario
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionar a tamaño máximo 500x500
    image.thumbnail((500, 500), Image.Resampling.LANCZOS)
    
    # Guardar imagen optimizada
    image.save(filepath, 'JPEG', quality=85, optimize=True)
    
    # Devolver URL relativa
    return f"/uploads/profile_images/{filename}"

def delete_profile_image(image_url: str):
    """
    Elimina una imagen de perfil
    
    Args:
        image_url: URL de la imagen a eliminar
    """
    if image_url:
        # Extraer nombre de archivo de la URL
        filename = image_url.split("/")[-1]
        filepath = os.path.join(PROFILE_IMAGES_DIR, filename)
        
        if os.path.exists(filepath):
            os.remove(filepath)

def get_default_profile_image() -> str:
    """
    Devuelve la URL de la imagen de perfil por defecto
    """
    return "/uploads/default_profile.png"