# =============================================================
# LAB 6 — frontend-iris/Dockerfile
# Imagen Docker para el frontend Gradio de la aplicación Iris
# =============================================================

# Imagen base oficial de Python, versión slim para reducir tamaño
FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar e instalar dependencias ANTES que el código fuente
# Así la capa de pip se cachea aunque cambie el código
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Puerto estándar de Gradio
EXPOSE 7860

# Variable de entorno con la URL del backend.
# Se sobreescribirá en docker-compose.yml con el nombre de servicio ('backend'),
# que es la URL válida dentro de la red interna de Docker Compose.
ENV BACKEND_URL=http://localhost:8000

# Gradio lanza con server_name="0.0.0.0" en gradio_app.py para ser accesible
# desde fuera del contenedor
CMD ["python", "gradio_app.py"]
