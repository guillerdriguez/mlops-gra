import requests
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

API_URL = "http://127.0.0.1:8000/invocations"

def predict_batch(df_features: pd.DataFrame):
    payload = {
        "dataframe_split": {
            "columns": df_features.columns.tolist(),
            "data": df_features.values.tolist()
        }
    }

    response = requests.post(API_URL, json=payload, timeout=60)
    response.raise_for_status()
    preds = response.json()

    if isinstance(preds, dict) and "predictions" in preds:
        return preds["predictions"]
    return preds

def main():
    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # Reproducir exactamente el split 70/20/10 usado en training
    from sklearn.model_selection import train_test_split

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )

    y_pred = predict_batch(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("=== MÉTRICAS SOBRE TEST VÍA API ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    main()