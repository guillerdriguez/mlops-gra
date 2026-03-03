"""
app-iris-ct: Continuous Training extension for ML-FastAPI-Docker
================================================================
Extends app-iris with:
  - POST /train      → reentrenamiento incremental con nuevas muestras
  - GET  /model/info → versión activa, métricas, historial
  - POST /predict    → inferencia (igual que app-iris, con versión activa)
  - GET  /health     → estado del servicio

[MOD] Añadido: Política de activación configurable para /train:
  - policy: "any_improvement" | "min_delta" | "per_class_f1"
  - min_delta (para min_delta)
  - target_class (para per_class_f1)
  - Guardamos en el historial la policy usada y el motivo (decision_reason)
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Literal  # [MOD] Literal para policy

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score  # [MOD] f1_score para per_class_f1
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "model_active.joblib"
HISTORY_PATH = MODELS_DIR / "training_history.json"

# ---------------------------------------------------------------------------
# Esquemas Pydantic
# ---------------------------------------------------------------------------

class IrisSample(BaseModel):
    sepal_length: float = Field(..., example=5.1, description="Longitud del sépalo (cm)")
    sepal_width: float  = Field(..., example=3.5, description="Anchura del sépalo (cm)")
    petal_length: float = Field(..., example=1.4, description="Longitud del pétalo (cm)")
    petal_width: float  = Field(..., example=0.2, description="Anchura del pétalo (cm)")


class LabeledSample(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float  = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float  = Field(..., example=0.2)
    label: int = Field(..., ge=0, le=2, example=0,
                       description="0=setosa, 1=versicolor, 2=virginica")


class TrainRequest(BaseModel):
    samples: List[LabeledSample] = Field(
        ..., min_items=5,
        description="Nuevas muestras etiquetadas para reentrenamiento (mínimo 5)"
    )
    retrain_from_scratch: bool = Field(
        False,
        description="Si True, ignora datos anteriores y entrena solo con las muestras enviadas"
    )

    # -----------------------------------------------------------------------
    # [MOD] Política configurable (ejercicio de programación)
    # -----------------------------------------------------------------------
    policy: Literal["any_improvement", "min_delta", "per_class_f1"] = Field(
        "any_improvement",
        description="Política de activación del modelo"
    )
    min_delta: float = Field(
        0.02,
        ge=0.0,
        description="Mejora mínima requerida (solo para policy=min_delta). Ej: 0.02 = +2 puntos"
    )
    target_class: Optional[int] = Field(
        None,
        ge=0,
        le=2,
        description="Clase objetivo para policy=per_class_f1 (0/1/2)"
    )


class PredictResponse(BaseModel):
    prediction: int
    class_name: str
    model_version: str


class TrainResponse(BaseModel):
    status: str
    model_version: str
    accuracy_new: float
    accuracy_previous: Optional[float]
    model_updated: bool
    message: str


class ModelInfo(BaseModel):
    active_version: str
    trained_at: str
    accuracy: float
    n_training_samples: int
    algorithm: str
    history: List[dict]


# ---------------------------------------------------------------------------
# Utilidades de persistencia
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}


def load_history() -> List[dict]:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []


def save_history(history: List[dict]):
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def get_active_model_meta():
    history = load_history()
    if not history:
        return None

    # modelo activo = última versión activada
    for entry in reversed(history):
        if entry.get("activated", False):
            return entry

    # fallback por compatibilidad
    return history[0]


# ---------------------------------------------------------------------------
# Bootstrap: si no existe modelo, lo entrenamos con el dataset original
# ---------------------------------------------------------------------------

def bootstrap_model():
    """Entrena un modelo base con el dataset Iris completo al arrancar."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    iris = load_iris()
    X, y = iris.data, iris.target

    # [NOTA] random_state fijo => bootstrap reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = float(accuracy_score(y_test, clf.predict(X_test)))

    version = "v1.0-base"
    joblib.dump(clf, MODEL_PATH)

    history = [{
        "version": version,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "accuracy": round(accuracy, 4),
        "n_training_samples": len(X_train),
        "algorithm": "LogisticRegression",
        "source": "bootstrap (iris dataset completo)",
        "status": "activado",
        "activated": True,

        # [MOD] Campos nuevos para consistencia con las políticas
        "policy": "bootstrap",
        "policy_params": {},
        "decision_reason": "bootstrap inicial",
        "f1_per_class": None
    }]
    save_history(history)
    print(f"[bootstrap] Modelo base creado → versión={version}, accuracy={accuracy:.4f}")


# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Iris Continuous Training API",
    description=(
        "Extensión MLOps de app-iris. Sirve predicciones y permite reentrenar "
        "el modelo con nuevas muestras etiquetadas, registrando el historial de versiones."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    if not MODEL_PATH.exists():
        bootstrap_model()
    else:
        meta = get_active_model_meta()
        if meta:
            print(f"[startup] Modelo activo cargado → versión={meta['version']}, "
                  f"accuracy={meta['accuracy']}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Sistema"])
def health():
    meta = get_active_model_meta()
    return {
        "status": "ok",
        "active_model_version": meta["version"] if meta else "none",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inferencia"])
def predict(sample: IrisSample):
    """
    Realiza una predicción con el modelo activo.
    Devuelve la clase predicha, su nombre y la versión del modelo usado.
    """
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Modelo no disponible. Llama primero a /train.")

    clf = joblib.load(MODEL_PATH)
    X = np.array([[
        sample.sepal_length,
        sample.sepal_width,
        sample.petal_length,
        sample.petal_width
    ]])
    pred = int(clf.predict(X)[0])
    meta = get_active_model_meta()

    return PredictResponse(
        prediction=pred,
        class_name=CLASS_NAMES[pred],
        model_version=meta["version"] if meta else "unknown"
    )


@app.post("/train", response_model=TrainResponse, tags=["Entrenamiento"])
def train(request: TrainRequest):
    """
    Reentrena el modelo con las nuevas muestras enviadas.

    - Si `retrain_from_scratch=False` (por defecto), las nuevas muestras se añaden
      al dataset de entrenamiento anterior (si existe) y se reentrena sobre el total.
    - Si `retrain_from_scratch=True`, solo se usan las muestras enviadas.

    [MOD] Gate configurable:
      - any_improvement: accuracy_new >= accuracy_previous (comportamiento original)
      - min_delta: accuracy_new >= accuracy_previous + min_delta
      - per_class_f1: solo activa si mejora la F1 de una clase concreta
    """

    # 1. Preparar nuevas muestras
    new_X = np.array([[s.sepal_length, s.sepal_width, s.petal_length, s.petal_width]
                      for s in request.samples])
    new_y = np.array([s.label for s in request.samples])

    # 2. Recuperar métricas del modelo activo
    history = load_history()
    active_meta = get_active_model_meta()
    previous_accuracy = active_meta.get("accuracy") if active_meta else None
    previous_f1_per_class = active_meta.get("f1_per_class") if active_meta else None  # [MOD]

    # 3. Construir dataset de entrenamiento
    data_file = MODELS_DIR / "accumulated_data.joblib"

    if not request.retrain_from_scratch and data_file.exists():
        saved = joblib.load(data_file)
        X_train = np.vstack([saved["X"], new_X])
        y_train = np.concatenate([saved["y"], new_y])
        source = f"incremental (+{len(new_X)} muestras nuevas, {len(saved['X'])} anteriores)"
    else:
        X_train, y_train = new_X, new_y
        source = f"desde cero ({len(new_X)} muestras)"

    # 4. Necesitamos al menos 2 clases para entrenar
    if len(np.unique(y_train)) < 2:
        raise HTTPException(
            status_code=422,
            detail="El dataset de entrenamiento debe contener al menos 2 clases distintas."
        )

    # 5. Entrenar nuevo modelo
    clf_new = LogisticRegression(max_iter=300, random_state=42)

    # Evaluación: si hay suficientes datos, usamos split; si no, evaluamos en train
    if len(X_train) >= 20:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train  # [MOD] stratify para estabilidad
        )
        clf_new.fit(X_tr, y_tr)
        y_pred = clf_new.predict(X_val)

        accuracy_new = float(accuracy_score(y_val, y_pred))
        f1_per_class_new = f1_score(y_val, y_pred, average=None, labels=[0, 1, 2]).tolist()  # [MOD]
        eval_note = f"validación con {len(X_val)} muestras"
    else:
        clf_new.fit(X_train, y_train)
        y_pred = clf_new.predict(X_train)

        accuracy_new = float(accuracy_score(y_train, y_pred))
        f1_per_class_new = f1_score(y_train, y_pred, average=None, labels=[0, 1, 2]).tolist()  # [MOD]
        eval_note = "evaluación en train (dataset pequeño, < 20 muestras)"

    accuracy_new = round(accuracy_new, 4)
    f1_per_class_new = [round(v, 4) for v in f1_per_class_new]  # [MOD] redondeo simple

    # 6. Decidir si activar el nuevo modelo según policy
    policy = request.policy  # [MOD]
    decision_reason = ""     # [MOD]

    if policy == "any_improvement":
        model_updated = (previous_accuracy is None) or (accuracy_new >= previous_accuracy)
        if previous_accuracy is None:
            decision_reason = "any_improvement: primer modelo"
        else:
            decision_reason = f"any_improvement: {accuracy_new:.4f} >= {previous_accuracy:.4f}"

    elif policy == "min_delta":
        delta = request.min_delta
        if previous_accuracy is None:
            model_updated = True
            decision_reason = f"min_delta: primer modelo (delta={delta:.4f})"
        else:
            model_updated = accuracy_new >= (previous_accuracy + delta)
            decision_reason = f"min_delta: {accuracy_new:.4f} >= {previous_accuracy:.4f} + {delta:.4f}"

    elif policy == "per_class_f1":
        if request.target_class is None:
            raise HTTPException(status_code=422, detail="policy=per_class_f1 requiere target_class (0/1/2).")

        k = int(request.target_class)
        if previous_f1_per_class is None:
            # Si venimos de un historial viejo sin f1_per_class, activamos para empezar a comparar.
            model_updated = True
            decision_reason = f"per_class_f1: primer modelo para comparar clase {k}"
        else:
            model_updated = f1_per_class_new[k] > float(previous_f1_per_class[k])
            decision_reason = f"per_class_f1: f1_clase_{k} {f1_per_class_new[k]:.4f} > {float(previous_f1_per_class[k]):.4f}"

    else:
        # Por si llega algo raro (aunque Literal ya lo limita)
        raise HTTPException(status_code=422, detail="policy no válida.")

    version = f"v{len(history) + 1}.0-{uuid.uuid4().hex[:6]}"
    status = "activado" if model_updated else "rechazado"

    if model_updated:
        joblib.dump(clf_new, MODEL_PATH)
        joblib.dump({"X": X_train, "y": y_train}, data_file)

        message = (
            f"Nuevo modelo activado. Accuracy {accuracy_new:.4f} "
            f"{'(primer modelo)' if previous_accuracy is None else f'>= anterior ({previous_accuracy:.4f})'}"
        )
        # [MOD] añado razón resumida para que quede registrado también “por qué”
        message += f" | {decision_reason}"
    else:
        message = (
            f"Modelo NO activado. Accuracy {accuracy_new:.4f} "
            f"< anterior ({previous_accuracy:.4f}). El modelo activo se mantiene sin cambios."
        )
        message += f" | {decision_reason}"

    # 7. Registrar en historial (también los rechazados)
    history.append({
        "version": version,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "accuracy": accuracy_new,
        "n_training_samples": len(X_train),
        "algorithm": "LogisticRegression",
        "source": source,
        "eval_note": eval_note,
        "status": status,
        "activated": model_updated,

        # [MOD] Guardado de política y motivos
        "policy": policy,
        "policy_params": {
            "min_delta": request.min_delta if policy == "min_delta" else None,
            "target_class": request.target_class if policy == "per_class_f1" else None,
        },
        "decision_reason": decision_reason,
        "f1_per_class": f1_per_class_new
    })
    save_history(history)

    return TrainResponse(
        status=status,
        model_version=version,
        accuracy_new=accuracy_new,
        accuracy_previous=previous_accuracy,
        model_updated=model_updated,
        message=message
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Modelo"])
def model_info():
    """
    Devuelve información del modelo activo y el historial completo de entrenamientos.
    """
    history = load_history()
    if not history:
        raise HTTPException(status_code=404, detail="No hay ningún modelo entrenado aún.")

    # El modelo activo es el último con activated=True
    active_entries = [h for h in history if h.get("activated", False)]
    active = active_entries[-1] if active_entries else history[0]

    return ModelInfo(
        active_version=active["version"],
        trained_at=active["trained_at"],
        accuracy=active["accuracy"],
        n_training_samples=active["n_training_samples"],
        algorithm=active["algorithm"],
        history=history
    )


@app.delete("/model/history", tags=["Modelo"])
def reset_history():
    """
    [CUIDADO] Elimina el historial y el modelo activo.
    Útil para pruebas y demostración.
    """
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    data_file = MODELS_DIR / "accumulated_data.joblib"
    if data_file.exists():
        data_file.unlink()

    bootstrap_model()
    return {"status": "ok", "message": "Historial eliminado y modelo base restaurado."}