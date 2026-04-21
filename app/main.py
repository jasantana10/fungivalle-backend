from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path

from app.database import engine, Base
from app.routes import auth, fungi, password_reset
from app.routes import profile
from app.routes import security_auth
from app.routes import hongos, fungi_findings

# Crear tablas en la base de datos (con manejo de errores para evitar que el servidor no suba)
try:
    Base.metadata.create_all(bind=engine)
    print("✅ Tablas de base de datos verificadas/creadas")
except Exception as e:
    print(f"⚠️ Error al conectar o crear tablas: {e}")
    print("El servidor intentará continuar, pero las funciones de DB podrían fallar.")

app = FastAPI(
    title="FungiValle API",
    description="API para identificación de hongos en Valledupar",
    version="1.0.0"
)

# Servir archivos estáticos de uploads
uploads_path = Path("uploads")
if uploads_path.exists():
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
    print("✅ Servidor de archivos estáticos configurado en /uploads")

# Configurar CORS para Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En desarrollo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos
if not os.path.exists("uploads"):
    os.makedirs("uploads")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Incluir rutas
app.include_router(auth.router)
app.include_router(fungi.router)
app.include_router(password_reset.router)
app.include_router(profile.router)
app.include_router(security_auth.router)
app.include_router(hongos.router, prefix="/api/v1")
app.include_router(fungi_findings.router, prefix="/api/v1/hallazgos", tags=["hallazgos"])

@app.get("/")
def read_root():
    return {
        "message": "Bienvenido a FungiValle API",
        "version": "1.0.0",
        "endpoints": {
            "auth": ["/auth/register", "/auth/login"],
            "fungi": ["/fungi/species", "/fungi/identify", "/fungi/findings"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)