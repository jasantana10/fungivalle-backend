import os
from uuid import uuid4
from PIL import Image
import io

def save_image(image_data: bytes, filename: str, subfolder: str = "") -> str:
    """
    Guarda una imagen en el sistema de archivos
    
    Args:
        image_data: Bytes de la imagen
        filename: Nombre del archivo
        subfolder: Subcarpeta dentro de uploads
    
    Returns:
        URL relativa de la imagen
    """
    # Crear directorios si no existen
    upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
    target_dir = os.path.join(upload_dir, subfolder) if subfolder else upload_dir
    os.makedirs(target_dir, exist_ok=True)
    
    # Guardar imagen
    filepath = os.path.join(target_dir, filename)
    with open(filepath, "wb") as f:
        f.write(image_data)
    
    # Devolver ruta relativa
    if subfolder:
        return f"/uploads/{subfolder}/{filename}"
    return f"/uploads/{filename}"

def process_image_for_ai(image_data: bytes):
    """
    Procesa imagen para el modelo de IA
    
    Args:
        image_data: Bytes de la imagen
    
    Returns:
        Imagen procesada para el modelo
    """
    # Abrir imagen con PIL
    image = Image.open(io.BytesIO(image_data))
    
    # Convertir a RGB si es necesario
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionar si es muy grande (máx 1024px)
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Guardar en buffer
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=85)
    img_byte_arr = img_byte_arr.getvalue()
    
    return img_byte_arr

def is_valid_image(image_data: bytes, max_size_mb: int = 10) -> bool:
    """
    Verifica si la imagen es válida
    
    Args:
        image_data: Bytes de la imagen
        max_size_mb: Tamaño máximo en MB
    
    Returns:
        bool: True si la imagen es válida
    """
    # Verificar tamaño
    if len(image_data) > max_size_mb * 1024 * 1024:
        return False
    
    # Verificar que sea una imagen válida
    try:
        Image.open(io.BytesIO(image_data))
        return True
    except:
        return False