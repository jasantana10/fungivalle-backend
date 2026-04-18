@echo off
echo ========================================
echo    INICIANDO BACKEND FUNGIVALLE
echo ========================================
echo.
echo Activando entorno virtual...
call venv\Scripts\activate.bat

echo.
echo Instalando dependencias si es necesario...
pip install -r requirements.txt

echo.
echo Iniciando servidor FastAPI...
echo API disponible en: http://localhost:8000
echo Documentacion: http://localhost:8000/docs
echo.

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause