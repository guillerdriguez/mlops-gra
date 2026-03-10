import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from mlflow.models import infer_signature


TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "LAB4_Diabetes"
REGISTERED_MODEL_NAME = "diabetes_classifier"

RANDOM_STATE = 42


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def build_candidates():
    candidates = {
        "logreg": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
            ]),
            "param_grid": {
                "model__C": [0.01, 0.1, 1, 10],
                "model__class_weight": [None, "balanced"],
                "model__solver": ["liblinear", "lbfgs"]
            }
        },
        "random_forest": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(random_state=RANDOM_STATE))
            ]),
            "param_grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5],
                "model__class_weight": [None, "balanced"]
            }
        },
        "svm": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVC(random_state=RANDOM_STATE))
            ]),
            "param_grid": {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
                "model__class_weight": [None, "balanced"]
            }
        }
    }
    return candidates


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # Split 70/20/10
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=0.10,
        random_state=RANDOM_STATE,
        stratify=y
    )

    val_size_relative = 20 / 90  # para que el total quede 70/20/10
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size_relative,
        random_state=RANDOM_STATE,
        stratify=y_trainval
    )

    print("Tamaños:")
    print(f"Train: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test: {len(X_test)}")

    # Para GridSearch usando train como entrenamiento y val como validación fija
    X_search = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_search = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    test_fold = [-1] * len(X_train) + [0] * len(X_val)
    predefined_split = PredefinedSplit(test_fold=test_fold)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1"
    }

    candidates = build_candidates()
    all_results = []

    for model_name, config in candidates.items():
        print(f"\n===== Entrenando {model_name} =====")

        grid = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["param_grid"],
            scoring=scoring,
            refit="recall",
            cv=predefined_split,
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_search, y_search)
        best_model = grid.best_estimator_

        y_val_pred = best_model.predict(X_val)
        val_metrics = compute_metrics(y_val, y_val_pred)

        cm = confusion_matrix(y_val, y_val_pred)
        tn, fp, fn, tp = cm.ravel()

        result = {
            "candidate_family": model_name,
            "best_params": grid.best_params_,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1_score": val_metrics["f1_score"],
            "val_true_negatives": int(tn),
            "val_false_positives": int(fp),
            "val_false_negatives": int(fn),
            "val_true_positives": int(tp),
            "selection_reason": ""
        }

        all_results.append((result, best_model, X_train))

    # Orden clínicamente razonable:
    # primero recall, luego F1, luego precision, luego accuracy
    all_results.sort(
        key=lambda item: (
            item[0]["val_recall"],
            item[0]["val_f1_score"],
            item[0]["val_precision"],
            item[0]["val_accuracy"]
        ),
        reverse=True
    )

    # Registrar solo finalistas: máximo 3, que en este caso coinciden con las 3 familias
    rows_for_csv = []

    for rank, (result, model, X_signature) in enumerate(all_results, start=1):
        if rank == 1:
            result["selection_reason"] = (
                "Modelo prioritario por mayor recall en validación, criterio principal "
                "en contexto médico para reducir falsos negativos."
            )
        elif rank == 2:
            result["selection_reason"] = (
                "Modelo alternativo con buen equilibrio entre recall y F1, útil para "
                "comparar compromiso entre sensibilidad y precisión."
            )
        else:
            result["selection_reason"] = (
                "Modelo registrado como referencia comparativa por pertenecer a una familia "
                "distinta y aportar una alternativa metodológica."
            )

        with mlflow.start_run(run_name=f"finalist_{rank}_{result['candidate_family']}"):
            mlflow.log_param("candidate_family", result["candidate_family"])
            mlflow.log_param("data_split", "70_train_20_validation_10_test")
            mlflow.log_param("selection_priority", "recall_then_f1_then_precision_then_accuracy")
            mlflow.log_param("best_params", json.dumps(result["best_params"]))

            mlflow.log_metric("val_accuracy", result["val_accuracy"])
            mlflow.log_metric("val_precision", result["val_precision"])
            mlflow.log_metric("val_recall", result["val_recall"])
            mlflow.log_metric("val_f1_score", result["val_f1_score"])
            mlflow.log_metric("val_true_negatives", result["val_true_negatives"])
            mlflow.log_metric("val_false_positives", result["val_false_positives"])
            mlflow.log_metric("val_false_negatives", result["val_false_negatives"])
            mlflow.log_metric("val_true_positives", result["val_true_positives"])

            mlflow.set_tag("selection_reason", result["selection_reason"])

            signature = infer_signature(X_signature, model.predict(X_signature))

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_signature.head(3),
                registered_model_name=REGISTERED_MODEL_NAME
            )

            row = {
                "rank": rank,
                "registered_run_name": f"finalist_{rank}_{result['candidate_family']}",
                "candidate_family": result["candidate_family"],
                "best_params": json.dumps(result["best_params"]),
                "val_accuracy": round(result["val_accuracy"], 4),
                "val_precision": round(result["val_precision"], 4),
                "val_recall": round(result["val_recall"], 4),
                "val_f1_score": round(result["val_f1_score"], 4),
                "val_false_negatives": result["val_false_negatives"],
                "val_false_positives": result["val_false_positives"],
                "selection_reason": result["selection_reason"],
                "model_uri": model_info.model_uri
            }
            rows_for_csv.append(row)

    results_df = pd.DataFrame(rows_for_csv)
    results_df.to_csv("results/registered_models_summary.csv", index=False)

    print("\nResumen de modelos registrados:")
    print(results_df)


if __name__ == "__main__":
    main()