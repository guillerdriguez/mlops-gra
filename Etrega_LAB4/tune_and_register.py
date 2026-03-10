import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

EXPERIMENT_NAME = "LAB4_Diabetes"

def get_model_grids():
    return {
        "logreg": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000))
            ]),
            "params": {
                "model__C": [0.01, 0.1, 1, 10],
                "model__class_weight": [None, "balanced"],
                "model__solver": ["liblinear", "lbfgs"]
            }
        },
        "rf": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(random_state=42))
            ]),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5],
                "model__class_weight": [None, "balanced"]
            }
        },
        "svc": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVC())
            ]),
            "params": {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
                "model__class_weight": [None, "balanced"]
            }
        }
    }

def main():
    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # Split 70/20/10 estratificado
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )

    val_relative_size = 20 / 90  # porque train+val es 90%
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_relative_size,
        random_state=42,
        stratify=y_trainval
    )

    # Unimos train y val para usar PredefinedSplit
    X_search = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_search = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    split_index = [-1] * len(X_train) + [0] * len(X_val)
    predefined_split = PredefinedSplit(test_fold=split_index)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1"
    }

    mlflow.set_experiment(EXPERIMENT_NAME)

    results_summary = []

    model_grids = get_model_grids()

    for model_name, cfg in model_grids.items():
        with mlflow.start_run(run_name=f"grid_{model_name}") as run:
            grid = GridSearchCV(
                estimator=cfg["pipeline"],
                param_grid=cfg["params"],
                scoring=scoring,
                refit="recall",
                cv=predefined_split,
                n_jobs=-1,
                verbose=1
            )

            grid.fit(X_search, y_search)

            best_model = grid.best_estimator_

            y_val_pred = best_model.predict(X_val)

            metrics = {
                "val_accuracy": accuracy_score(y_val, y_val_pred),
                "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
                "val_recall": recall_score(y_val, y_val_pred, zero_division=0),
                "val_f1_score": f1_score(y_val, y_val_pred, zero_division=0)
            }

            mlflow.log_param("candidate_family", model_name)
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)

            # Guarda también las métricas del mejor CV interno
            mlflow.log_metric("best_cv_recall", grid.best_score_)

            signature = mlflow.models.infer_signature(X_train, best_model.predict(X_train))
            model_info = mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(3),
                registered_model_name="diabetes_classifier"
            )

            results_summary.append({
                "run_id": run.info.run_id,
                "model_name": model_name,
                **grid.best_params_,
                **metrics,
                "model_uri": model_info.model_uri
            })

    results_df = pd.DataFrame(results_summary)
    results_df = results_df.sort_values(
        by=["val_recall", "val_f1_score", "val_precision", "val_accuracy"],
        ascending=False
    )
    results_df.to_csv("registered_candidates_summary.csv", index=False)
    print(results_df)

if __name__ == "__main__":
    main()