"""
TASK 2: Detectar Data Drift
=====================================
Si train y test son indistinguibles → no hay drift → bien.
Si un modelo puede separarlos fácilmente → hay drift → problema.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── 1. CARGAR Y PREPARAR DATOS ───────────────────────────────────────────────
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

# Eliminamos columnas que el enunciado indica que no son features del producto
train_clean = train.drop(columns=["id", "failure"])  # quitamos failure también
test_clean  = test.drop(columns=["id"])

# Etiquetamos: 0 = viene de train, 1 = viene de test
train_clean["origin"] = 0
test_clean["origin"]  = 1

# ── 2. COMBINAR EN UN SOLO DATAFRAME ─────────────────────────────────────────
combined = pd.concat([train_clean, test_clean], axis=0).reset_index(drop=True)

# Separamos el target (origin) de las features
y        = combined["origin"]
features = combined.drop(columns=["origin"])

# ── 3. PREPROCESAMIENTO (igual que en Task 1) ────────────────────────────────
for col in ["attribute_0", "attribute_1"]:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col].astype(str))

le_code = LabelEncoder()
features["product_code"] = le_code.fit_transform(features["product_code"].astype(str))

features = features.fillna(features.median(numeric_only=True))

print(f"Dataset combinado: {features.shape}")
print(f"Filas de train (label=0): {(y==0).sum()}")
print(f"Filas de test  (label=1): {(y==1).sum()}")

# ── 4. ENTRENAR MODELO DETECTOR DE DRIFT ────────────────────────────────────
# La pregunta es: ¿puede un modelo aprender a distinguir train de test?
# Si puede → los datasets son diferentes (drift)
# Si no puede (AUC ≈ 0.5) → son homogéneos (sin drift)

model_drift = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model_drift, features, y, cv=cv, scoring="roc_auc")

print(f"\n── Resultados: ¿Podemos distinguir train de test? ──")
print(f"AUC por fold: {[round(s, 4) for s in cv_scores]}")
print(f"AUC medio:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 5. INTERPRETACIÓN ────────────────────────────────────────────────────────
auc = cv_scores.mean()

print("\n── INTERPRETACIÓN ──────────────────────────────────────────")
if auc > 0.80:
    print(f"⚠️  AUC = {auc:.3f} → DRIFT SIGNIFICATIVO detectado.")
    print("   El modelo distingue fácilmente train de test.")
    print("   Esto significa que los datos son muy diferentes entre sí.")
    print("   Tu modelo de Task 1 probablemente tendrá peor rendimiento real.")
elif auc > 0.60:
    print(f"⚡ AUC = {auc:.3f} → DRIFT MODERADO detectado.")
    print("   Hay algunas diferencias entre train y test.")
    print("   Conviene revisar qué variables cambian más.")
else:
    print(f"✅ AUC = {auc:.3f} → SIN DRIFT significativo.")
    print("   Train y test son indistinguibles.")
    print("   Los datasets son homogéneos. Tu modelo debería generalizar bien.")

# ── 6. ¿QUÉ VARIABLES DELATAN EL DRIFT? ─────────────────────────────────────
model_drift.fit(features, y)
importances = pd.Series(model_drift.feature_importances_, index=features.columns)

print("\n── Variables que más cambian entre train y test ──")
print(importances.sort_values(ascending=False).head(10).round(4))

# ── 7. ESTADÍSTICAS COMPARATIVAS ─────────────────────────────────────────────
print("\n── Comparación de medias: Train vs Test ──")
numeric_cols = features.select_dtypes(include='number').columns
train_means = features.loc[y==0, numeric_cols].mean()
test_means  = features.loc[y==1, numeric_cols].mean()

comparison = pd.DataFrame({
    "media_train": train_means,
    "media_test":  test_means,
    "diferencia_%": ((test_means - train_means) / train_means.abs() * 100).round(1)
}).sort_values("diferencia_%", key=abs, ascending=False)

print(comparison.head(10).round(3))