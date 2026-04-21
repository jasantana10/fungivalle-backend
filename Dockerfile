# Usar Python 3.11 como base
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema (necesarias para TensorFlow)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libhdf5-dev \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt primero (para aprovechar caché de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]