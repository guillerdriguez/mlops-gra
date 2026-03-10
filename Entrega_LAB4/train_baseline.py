import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models import infer_signature


TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "LAB4_Diabetes"
REGISTERED_MODEL_NAME = "diabetes_classifier"


def main():
    # Configuración MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Cargar datos
    df = pd.read_csv("diabetes.csv")

    # Separar variables
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # Split simple para experimento de prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Pipeline baseline
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    with mlflow.start_run(run_name="baseline_logistic_regression"):
        # Entrenamiento
        pipeline.fit(X_train, y_train)

        # Predicción
        y_pred = pipeline.predict(X_test)

        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Logging de parámetros
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("imputer_strategy", "median")
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Logging de métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Firma del modelo
        signature = infer_signature(X_train, pipeline.predict(X_train))

        # Registro del modelo
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name=REGISTERED_MODEL_NAME
        )

        print("Modelo entrenado y registrado correctamente.")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Model URI: {model_info.model_uri}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")


if __name__ == "__main__":
    main()