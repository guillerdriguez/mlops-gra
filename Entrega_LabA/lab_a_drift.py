"""
LabA – Análisis de deriva de datos con Evidently AI
====================================================
Dataset: Titanic (seaborn)

Se generan 12 condiciones de división (train/val/test):
  - 2 opciones de estratificación (con / sin)
  - 3 proporciones: 60/20/20, 90/5/5, 98/1/1
  - 2 semillas aleatorias: 42, 123

Para cada condición se generan 2 reports HTML:
  - <condición>_val.html   (train como referencia, val como check)
  - <condición>_test.html  (train como referencia, test como check)

Total: 12 × 2 = 24 reports HTML
"""

from pathlib import Path
from itertools import product

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.metrics import ValueDrift

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

REPORTS_DIR = Path("reports")
DATA_DIR = Path("data")
REPORTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

TARGET_COL = "survived"

SPLIT_RATIOS = [
    (0.60, 0.20, 0.20),  # 60/20/20
    (0.90, 0.05, 0.05),  # 90/5/5
    (0.98, 0.01, 0.01),  # 98/1/1
]

SEEDS = [42, 123]

STRATIFY_OPTIONS = [True, False]

# ---------------------------------------------------------------------------
# Cargar y preprocesar el dataset Titanic
# ---------------------------------------------------------------------------

def load_titanic() -> pd.DataFrame:
    """Carga el dataset Titanic desde seaborn y realiza preprocesamiento básico."""
    df = sns.load_dataset("titanic")

    # Seleccionar columnas relevantes y descartar duplicadas/complejas
    # 'deck' se excluye porque tiene ~77% de valores nulos, lo que genera
    # columnas vacías en splits muy pequeños (98/1/1) y rompe Evidently.
    keep_cols = [
        "survived", "pclass", "sex", "age", "sibsp", "parch",
        "fare", "embarked", "class", "who", "adult_male",
        "embark_town", "alone"
    ]
    df = df[keep_cols].copy()

    # Convertir booleanos a int para Evidently
    df["adult_male"] = df["adult_male"].astype(int)
    df["alone"] = df["alone"].astype(int)

    return df


# ---------------------------------------------------------------------------
# Definición del esquema para Evidently
# ---------------------------------------------------------------------------

NUMERICAL_COLS = ["age", "sibsp", "parch", "fare"]
CATEGORICAL_COLS = ["pclass", "sex", "embarked", "class", "who", "adult_male", "embark_town", "alone", "survived"]

titanic_schema = DataDefinition(
    numerical_columns=NUMERICAL_COLS,
    categorical_columns=CATEGORICAL_COLS,
)


# ---------------------------------------------------------------------------
# Funciones de split
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide df en train/val/test con las proporciones indicadas."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Los ratios deben sumar 1"

    strat_col = df[TARGET_COL] if stratify else None

    # Primera división: train vs (val+test)
    val_test_ratio = val_ratio + test_ratio
    df_train, df_valtest = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=strat_col,
    )

    # Segunda división: val vs test dentro del conjunto val+test
    relative_test = test_ratio / val_test_ratio
    strat_valtest = df_valtest[TARGET_COL] if stratify else None

    # Evitar error cuando el conjunto es muy pequeño para estratificar
    if stratify and strat_valtest is not None:
        counts = strat_valtest.value_counts()
        if counts.min() < 2:
            strat_valtest = None  # degradar a no estratificado

    df_val, df_test = train_test_split(
        df_valtest,
        test_size=relative_test,
        random_state=seed,
        stratify=strat_valtest,
    )

    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# Función para construir un report de Evidently
# ---------------------------------------------------------------------------

def build_report() -> Report:
    """Construye el report de drift con DataDriftPreset + ValueDrift por columna."""
    metrics = [
        DataDriftPreset(),
        DataSummaryPreset(),
    ]
    for col in NUMERICAL_COLS:
        metrics.append(ValueDrift(column=col))
    for col in CATEGORICAL_COLS:
        metrics.append(ValueDrift(column=col))

    return Report(metrics, include_tests=True)


def run_report(
    train_df: pd.DataFrame,
    check_df: pd.DataFrame,
    output_path: Path,
) -> dict:
    """Ejecuta el report y guarda el HTML. Devuelve métricas resumidas."""
    ref_ds = Dataset.from_pandas(train_df, data_definition=titanic_schema)
    cur_ds = Dataset.from_pandas(check_df, data_definition=titanic_schema)

    report = build_report()
    result = report.run(reference_data=ref_ds, current_data=cur_ds)
    result.save_html(str(output_path))

    # Extraer fracción de columnas con drift del resultado
    drift_info = _extract_drift_fraction(result)
    drift_str = f"{drift_info:.4f}" if not pd.isna(drift_info) else "N/A"
    print(f"  [OK] {output_path.name}  →  drift fraction: {drift_str}")
    return {"drift_fraction": drift_info}


def _extract_drift_fraction(snapshot) -> float:
    """Extrae la fracción de columnas con drift detectado del Snapshot de Evidently."""
    try:
        mr = snapshot.metric_results
        for v in mr.values():
            display = getattr(v, "display_name", "")
            if "Count of Drifted Columns" in display and hasattr(v, "share"):
                share_obj = v.share
                if hasattr(share_obj, "value"):
                    return float(share_obj.value)
    except Exception:
        pass
    return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Cargando dataset Titanic...")
    df = load_titanic()
    print(f"  Shape: {df.shape}")

    # Guardar dataset completo
    df.to_csv(DATA_DIR / "titanic.csv", index=False)

    summary_rows = []

    conditions = list(product(STRATIFY_OPTIONS, SPLIT_RATIOS, SEEDS))
    print(f"\nGenerando reports para {len(conditions)} condiciones × 2 splits = {len(conditions)*2} reports\n")

    for stratify, (train_r, val_r, test_r), seed in conditions:
        ratio_label = f"{int(train_r*100)}-{int(val_r*100)}-{int(test_r*100)}"
        strat_label = "strat" if stratify else "nostrat"
        cond_name = f"{ratio_label}_{strat_label}_seed{seed}"

        print(f"\n── Condición: {cond_name}")

        # Split
        train_df, val_df, test_df = split_dataset(df, train_r, val_r, test_r, seed, stratify)
        print(f"   train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        # Reports
        val_path = REPORTS_DIR / f"{cond_name}_val.html"
        test_path = REPORTS_DIR / f"{cond_name}_test.html"

        val_metrics = run_report(train_df, val_df, val_path)
        test_metrics = run_report(train_df, test_df, test_path)

        summary_rows.append({
            "Condición": cond_name,
            "Estratificación": "Sí" if stratify else "No",
            "Split ratio": f"{int(train_r*100)}/{int(val_r*100)}/{int(test_r*100)}",
            "Semilla": seed,
            "N train": len(train_df),
            "N val": len(val_df),
            "N test": len(test_df),
            "Drift val (fracción columnas)": round(val_metrics["drift_fraction"], 4) if not pd.isna(val_metrics["drift_fraction"]) else "N/A",
            "Drift test (fracción columnas)": round(test_metrics["drift_fraction"], 4) if not pd.isna(test_metrics["drift_fraction"]) else "N/A",
        })

    # Guardar tabla resumen en CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(DATA_DIR / "drift_summary.csv", index=False)
    print("\n\n=== TABLA RESUMEN ===")
    print(summary_df.to_string(index=False))
    print(f"\nResumen guardado en {DATA_DIR / 'drift_summary.csv'}")
    print(f"Reports guardados en {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
