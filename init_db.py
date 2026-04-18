# init_db.py
import sys
import os

# Agregar la carpeta app al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import engine, Base
from app import models

print("Creando tablas en la base de datos...")

# Crear todas las tablas
Base.metadata.create_all(bind=engine)

print("✅ Tablas creadas exitosamente!")
print("Base de datos: fungivalle_db")
print("Tablas creadas:")
for table in Base.metadata.tables.keys():
    print(f"  - {table}")