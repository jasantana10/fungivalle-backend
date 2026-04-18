# seed_data.py
import sys
import os
from sqlalchemy.orm import Session

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal, engine
from app import models
from app.auth import get_password_hash

def create_initial_data():
    db = SessionLocal()
    
    try:
        # Crear especies de hongos de Valledupar (ejemplo)
        species_data = [
            {
                "scientific_name": "Amanita muscaria",
                "common_name": "Fly Agaric",
                "local_name": "Seta de las moscas",
                "description": "Hongo venenoso con sombrero rojo y puntos blancos",
                "habitat": "Bosques de coníferas y caducifolios",
                "season": "Verano-Otoño",
                "edible": "venenoso",
                "toxicity_level": "alto"
            },
            {
                "scientific_name": "Cantharellus cibarius",
                "common_name": "Chanterelle",
                "local_name": "Rebozuelo",
                "description": "Hongo comestible muy apreciado, color amarillo dorado",
                "habitat": "Bosques húmedos",
                "season": "Verano",
                "edible": "comestible",
                "toxicity_level": "ninguno"
            },
            {
                "scientific_name": "Pleurotus ostreatus",
                "common_name": "Oyster Mushroom",
                "local_name": "Seta de ostra",
                "description": "Hongo comestible que crece en troncos",
                "habitat": "Troncos de árboles",
                "season": "Todo el año",
                "edible": "comestible",
                "toxicity_level": "ninguno"
            }
        ]
        
        for data in species_data:
            # Verificar si ya existe
            existing = db.query(models.FungiSpecies).filter_by(
                scientific_name=data["scientific_name"]
            ).first()
            
            if not existing:
                species = models.FungiSpecies(**data)
                db.add(species)
                print(f"✅ Especie creada: {data['scientific_name']}")
        
        db.commit()
        
        # Crear usuario admin de prueba
        admin_email = "admin@fungivalle.com"
        existing_user = db.query(models.User).filter_by(email=admin_email).first()
        
        if not existing_user:
            admin_user = models.User(
                full_name="Administrador FungiValle",
                email=admin_email,
                hashed_password=get_password_hash("admin123")
            )
            db.add(admin_user)
            db.commit()
            print("✅ Usuario admin creado")
            print(f"   Email: {admin_email}")
            print(f"   Password: admin123")
        
        print("\n🎉 Datos iniciales insertados correctamente!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_initial_data()