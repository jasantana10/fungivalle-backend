import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import engine, Base
from app import models

print("Actualizando tablas de la base de datos...")

# Esto agregará las nuevas columnas a las tablas existentes
Base.metadata.create_all(bind=engine)

print("✅ Base de datos actualizada exitosamente!")
print("Tablas actualizadas:")
for table in Base.metadata.tables.keys():
    print(f"  - {table}")