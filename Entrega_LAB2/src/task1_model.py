"""
TASK 1: Predecir fallos de productos
=====================================
Usamos un RandomForest con preprocesamiento de datos.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ── 1. CARGAR DATOS ──────────────────────────────────────────────────────────
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

print(f"Train: {train.shape} | Test: {test.shape}")
print(f"Tasa de fallos en train: {train['failure'].mean():.1%}")

# ── 2. PREPROCESAMIENTO ──────────────────────────────────────────────────────
# Separamos el target antes de tocar nada
y = train["failure"]

# Juntamos train y test para hacer el mismo preprocesamiento a ambos
# (esto se llama "pipeline consistente")
train_ids = train["id"]
test_ids  = test["id"]

train_data = train.drop(columns=["id", "failure"])
test_data  = test.drop(columns=["id"])

combined = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)

# Codificamos las columnas categóricas (texto → números)
# attribute_0 y attribute_1 son strings como "material_7"
for col in ["attribute_0", "attribute_1"]:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

# product_code también es categórico
le_code = LabelEncoder()
combined["product_code"] = le_code.fit_transform(combined["product_code"].astype(str))

# Rellenar valores nulos con la mediana de cada columna
# (estrategia simple pero efectiva para datos numéricos)
combined = combined.fillna(combined.median(numeric_only=True))

# Separamos de nuevo train y test
n_train = len(train_data)
X_train = combined.iloc[:n_train]
X_test  = combined.iloc[n_train:]

print(f"\nFeatures usadas: {X_train.shape[1]}")
print(f"Valores nulos restantes: {X_train.isnull().sum().sum()}")

# ── 3. ENTRENAR MODELO ───────────────────────────────────────────────────────
# RandomForest: ensemble de muchos árboles de decisión.
# Parámetros clave:
#   n_estimators: cuántos árboles construye
#   max_depth: profundidad máxima de cada árbol (evita overfitting)
#   random_state: semilla para reproducibilidad

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1  # usa todos los cores del CPU
)

# Validación cruzada: dividimos train en 5 partes, entrenamos en 4 y 
# evaluamos en 1, rotando. Así estimamos el rendimiento real del modelo.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y, cv=cv, scoring="roc_auc")

print(f"\n── Resultados Validación Cruzada (5-fold) ──")
print(f"AUC por fold: {[round(s, 4) for s in cv_scores]}")
print(f"AUC medio:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
# AUC = 0.5 es aleatorio, AUC = 1.0 es perfecto

# Entrenamos con TODOS los datos de train para hacer las predicciones finales
model.fit(X_train, y)

# ── 4. IMPORTANCIA DE VARIABLES ──────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=X_train.columns)
print(f"\n── Top 10 variables más importantes ──")
print(importances.sort_values(ascending=False).head(10).round(4))

# ── 5. PREDICCIONES FINALES ───────────────────────────────────────────────────
# predict_proba devuelve probabilidades [prob_clase_0, prob_clase_1]
# Nos quedamos con la probabilidad de fallo (clase 1)
predictions = model.predict_proba(X_test)[:, 1]

output = pd.DataFrame({
    "id": test_ids,
    "failure": predictions
})

output.to_csv("outputs/predictions.csv", index=False)
print(f"\n✅ Predicciones guardadas en outputs/predictions.csv")
print(f"Distribución de probabilidades predichas:")
print(output["failure"].describe().round(3))