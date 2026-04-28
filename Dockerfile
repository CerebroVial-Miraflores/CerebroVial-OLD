FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema para OpenCV, psycopg2, librerías nativas
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python primero (aprovecha el cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY . .

# Hace que 'src' sea importable como paquete raíz
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.vision.presentation.api:app", "--host", "0.0.0.0", "--port", "8000"]
