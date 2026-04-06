# LAB 6 — Despliegue con Docker: MLOps-Capstone-Project

**Asignatura:** MLOps & AI BI — Sesión 6
**Alumno:** Guillermo
**Modalidad:** Individual

---

## Descripción

En este laboratorio se containeriza la aplicación completa del proyecto MLOps-Capstone, compuesta por:

- **Backend** — API FastAPI que descarga el modelo Iris desde Hugging Face Hub y sirve predicciones en `POST /predict`.
- **Frontend** — Interfaz Gradio que recibe los inputs del usuario, llama al backend y muestra la predicción.

Ambos servicios se orquestan con Docker Compose y arrancan con un único comando, aislados del sistema operativo host y reproducibles en cualquier máquina con Docker instalado.

---

## Estructura del proyecto

```
Entrega_Lab6/
├── MLOps-Capstone-Project/
│   ├── backend-iris/
│   │   ├── main.py               ← API FastAPI con health check robusto
│   │   ├── requirements.txt
│   │   └── Dockerfile            ← Python 3.11-slim + HEALTHCHECK
│   ├── frontend-iris/
│   │   ├── gradio_app.py         ← Interfaz Gradio
│   │   ├── requirements.txt
│   │   └── Dockerfile            ← Python 3.11-slim, puerto 7860
│   ├── docker-compose.yml        ← Orquestación con condition: service_healthy
│   └── verify_stack.sh           ← Script de verificación end-to-end
└── README.md                     ← Este fichero
```

---

## Decisiones de diseño

### Dockerfile del backend (`backend-iris/Dockerfile`)

- Imagen base `python:3.11-slim` para minimizar el tamaño de la imagen.
- **Orden de capas optimizado**: `COPY requirements.txt` + `RUN pip install` antes de `COPY . .`, para que la capa de dependencias se cachee aunque cambie el código fuente.
- Se instala `curl` (necesario para el `HEALTHCHECK`).
- **`HEALTHCHECK`** configurado con `--start-period=30s` para dar tiempo a que el modelo se descargue de Hugging Face Hub antes de comenzar los checks.

```dockerfile
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

### Dockerfile del frontend (`frontend-iris/Dockerfile`)

- Imagen base `python:3.11-slim`.
- Puerto `7860` (estándar de Gradio).
- Variable de entorno `BACKEND_URL=http://localhost:8000` por defecto, sobreescrita en docker-compose.yml con el nombre de servicio interno.

### Health check robusto en `main.py`

El endpoint `/health` devuelve:
- **HTTP 200** — cuando el modelo está completamente cargado en memoria.
- **HTTP 503** — mientras el modelo aún se está descargando de Hugging Face Hub.

Esto permite que `HEALTHCHECK` y `condition: service_healthy` funcionen correctamente.

### `docker-compose.yml`

Se usa `condition: service_healthy` en el `depends_on` del frontend:

```yaml
depends_on:
  backend:
    condition: service_healthy
```

Esto garantiza que el frontend solo arranca cuando el backend supera el `HEALTHCHECK`, evitando errores de conexión durante el inicio del stack.

---

## Cómo ejecutar

### Prerrequisitos

- Docker Desktop instalado y en ejecución.
- Token de Hugging Face Hub (si el repositorio del modelo es privado).

### Configuración del token (opcional)

Crea un fichero `.env` en la raíz de `MLOps-Capstone-Project/`:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

> Este fichero está en `.gitignore` y no se sube al repositorio.

### Arranque del stack completo

```bash
cd MLOps-Capstone-Project
docker compose up --build -d
```

Accede a:
- **Backend (docs interactivas):** http://localhost:8000/docs
- **Frontend (interfaz Gradio):** http://localhost:7860

### Verificación automática

```bash
chmod +x verify_stack.sh
./verify_stack.sh
```

### Parada del stack

```bash
docker compose down
```

---

## Ejercicios realizados

### Ejercicio 1 — Backend en solitario

Se construye y arranca el contenedor del backend de forma independiente. Se verifica el endpoint `GET /health` y `POST /predict`:

```bash
cd backend-iris
docker build -t iris-backend .
docker run -d -p 8000:8000 --name iris-backend iris-backend
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

### Ejercicio 2 — Frontend en solitario

Se construye y arranca el contenedor del frontend apuntando al backend del Ejercicio 1:

```bash
cd frontend-iris
docker build -t iris-frontend .
docker run -d -p 7860:7860 -e BACKEND_URL=http://host.docker.internal:8000 iris-frontend
```

### Ejercicio 3 — Stack completo con Docker Compose

```bash
docker compose up --build -d
docker compose ps
```

Ambos servicios aparecen en estado `running (healthy)`. El frontend se comunica con el backend usando el nombre de servicio `backend` dentro de la red interna.

### Ejercicio 4 — Verificación automática end-to-end

```bash
./verify_stack.sh
```

---

## Respuestas a las preguntas

### Pregunta 1 — `--host 0.0.0.0` vs `--host 127.0.0.1`

Cada contenedor Docker tiene su propia interfaz de red loopback. Si Uvicorn arranca con `--host 127.0.0.1` (o sin especificar, que es el valor por defecto), solo escucha conexiones que provengan del interior del propio contenedor. El mapeo de puertos de Docker (`-p 8000:8000`) reenvía el tráfico del host al contenedor, pero llega como una conexión *externa* desde el punto de vista del contenedor. Si el proceso solo escucha en loopback, esas conexiones externas son rechazadas.

Con `--host 0.0.0.0`, Uvicorn escucha en todas las interfaces de red del contenedor (incluyendo la interfaz conectada a la red Docker), por lo que el tráfico reenviado desde el host llega correctamente al proceso.

### Pregunta 2 — `BACKEND_URL=http://backend:8000` vs `http://localhost:8000`

Dentro de un contenedor, `localhost` (o `127.0.0.1`) apunta a la interfaz loopback *del propio contenedor*, no al backend que corre en otro contenedor. Docker Compose crea automáticamente una red bridge privada para todos los servicios del mismo `docker-compose.yml`. En esa red, Docker implementa un servidor DNS interno que resuelve el nombre de cada servicio (en este caso `backend`) a la IP del contenedor correspondiente. Por eso `http://backend:8000` es la URL correcta dentro de la red interna, mientras que `http://localhost:8000` no tiene conexión con el backend y produce un error de conexión rechazada.

### Pregunta 3 — Caché de capas y orden de instrucciones

Docker construye las imágenes de forma incremental: cada instrucción genera una capa inmutable. Si una capa no ha cambiado desde la última construcción, Docker la reutiliza de la caché sin ejecutarla de nuevo. Las capas se invalidan en cascada: si una capa cambia, todas las capas posteriores deben reconstruirse.

Si se hiciera `COPY . .` antes del `RUN pip install`, cualquier cambio en el código fuente (por ejemplo, una línea en `main.py`) invalidaría la capa de `COPY . .`, forzando a recalcular la capa de `pip install` aunque `requirements.txt` no haya cambiado. Esto puede suponer varios minutos de espera en cada iteración de desarrollo.

Con el orden correcto (`COPY requirements.txt` → `RUN pip install` → `COPY . .`), la capa de pip solo se invalida cuando cambia `requirements.txt`. Un cambio en `main.py` solo invalida la última capa `COPY . .`, que es prácticamente instantánea.

**Situaciones habituales:**
- *Cambiar una línea en `main.py`*: con el orden correcto, pip se cachea; con el orden incorrecto, pip se reinstala entero.
- *Añadir una dependencia nueva en `requirements.txt`*: en ambos casos pip se recalcula, pero con el orden correcto solo se recalcula desde esa capa.

### Pregunta 4 — Gestión de secretos en producción

**Riesgo de la solución con `.env`:** si el fichero `.env` se sube accidentalmente al repositorio (por un error en `.gitignore`), el token queda expuesto públicamente. Además, el token aparece en el historial de comandos del shell y en los logs de Docker si se imprime por error.

**Alternativas:**

1. **Variable de entorno del shell:** exportar el token en la sesión del terminal antes de ejecutar `docker compose up` (`export HF_TOKEN=...`). Es sencillo y no deja rastro en disco, pero el token desaparece al cerrar la sesión y no es adecuado para entornos CI/CD automatizados.

2. **Docker Secrets (modo Swarm):** permiten montar secretos como ficheros en `/run/secrets/` dentro del contenedor, sin que aparezcan en variables de entorno ni en el historial de capas de la imagen. Son adecuados para entornos de producción en clústeres Docker Swarm, pero no están disponibles en Docker Compose estándar (solo en modo Swarm).

3. **Servicios externos de gestión de secretos (AWS Secrets Manager, HashiCorp Vault, etc.):** el secreto nunca sale del servicio gestor; la aplicación solicita el token en tiempo de ejecución mediante una llamada autenticada. Es la solución más robusta para producción cloud a gran escala, con rotación automática, auditoría y control de acceso granular.

**Elección por entorno:**
- **Desarrollo local:** fichero `.env` con `.gitignore` es suficiente.
- **CI/CD (GitHub Actions, GitLab CI):** variables de entorno cifradas del propio CI.
- **Producción en Swarm o Kubernetes:** Docker Secrets o Kubernetes Secrets.
- **Producción cloud empresarial:** AWS Secrets Manager, Azure Key Vault o HashiCorp Vault.

---

## Ejercicio de programación — Health check y arranque ordenado

### Fase de razonamiento previo

**Escenario 1 (modelo pequeño, < 1s):** `depends_on: started` es suficiente; el modelo carga antes de que el frontend envíe la primera petición. No hay problema en la práctica.

**Escenario 2 (descarga desde HF Hub, latencia variable):** `depends_on: started` no es suficiente. El contenedor del backend aparece como "iniciado" en cuanto arranca uvicorn, pero el modelo puede tardar 10-60 segundos en descargarse. El frontend arranca, intenta conectarse e inmediatamente recibe errores 503. Con `condition: service_healthy` el frontend espera a que el HEALTHCHECK confirme que el modelo está en memoria.

**Escenario 3 (modelo grande, compilación/warmup de decenas de segundos):** el `start-period` del HEALTHCHECK debe configurarse con un valor suficientemente alto (ej. `--start-period=120s`) para evitar que Docker marque el contenedor como `unhealthy` antes de que termine la carga. La lógica de `/health` devuelve 503 mientras carga y 200 cuando está listo.

**Pseudocódigo de la solución:**

```
# backend/main.py
model_state = {"loaded": False, "model": None}

on_startup():
    lanzar_hilo(load_model_background)

load_model_background():
    model = descargar_y_cargar_modelo()
    model_state["loaded"] = True

GET /health:
    si model_state["loaded"]:
        return HTTP 200 {"status": "ok", "model_loaded": True}
    sino:
        return HTTP 503 {"status": "not_ready", "model_loaded": False}

# Dockerfile backend:
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3
    CMD curl -f http://localhost:8000/health || exit 1

# docker-compose.yml:
depends_on:
  backend:
    condition: service_healthy
```

Esta combinación garantiza que el frontend solo intenta conectarse cuando el modelo está completamente disponible, independientemente del tiempo de carga.
