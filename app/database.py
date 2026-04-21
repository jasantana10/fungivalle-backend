from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# Configuración desde .env
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:@localhost/fungivalle_db")

# Ajuste para SQLAlchemy 1.4+ con PostgreSQL (Supabase/Render usan postgres://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Configuración del motor según la base de datos
connect_args = {}
if "mysql" in DATABASE_URL:
    # MySQL specific args if any
    pass
elif "postgresql" in DATABASE_URL:
    # PostgreSQL specific args (required for Supabase pooler)
    connect_args = {"sslmode": "require"}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()