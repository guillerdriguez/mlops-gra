import json
import requests
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


API_URL = "http://127.0.0.1:8000/invocations"
RANDOM_STATE = 42


def predict_via_api(df_features: pd.DataFrame):
    payload = {
        "dataframe_split": {
            "columns": df_features.columns.tolist(),
            "data": df_features.values.tolist()
        }
    }

    response = requests.post(API_URL, json=payload, timeout=60)
    response.raise_for_status()

    result = response.json()

    if isinstance(result, dict) and "predictions" in result:
        return result["predictions"]

    return result


def main():
    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # Reproducimos exactamente el split 70/20/10 usado en tune_register.py
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=RANDOM_STATE,
        stratify=y
    )

    y_pred = predict_via_api(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("=== TEST METRICS VIA API ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    results = {
        "model_served": "diabetes_classifier version 3",
        "api_url": API_URL,
        "test_size": len(X_test),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist()
    }

    with open("results/test_metrics_from_api.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    with open("results/test_metrics_from_api.md", "w", encoding="utf-8") as f:
        f.write("# Test Metrics Obtained Through MLflow API\n\n")
        f.write(f"Served model: `{results['model_served']}`\n\n")
        f.write(f"API endpoint: `{API_URL}`\n\n")
        f.write(f"Test samples: `{results['test_size']}`\n\n")
        f.write("## Metrics\n\n")
        f.write(f"- Accuracy: {results['accuracy']}\n")
        f.write(f"- Precision: {results['precision']}\n")
        f.write(f"- Recall: {results['recall']}\n")
        f.write(f"- F1-score: {results['f1_score']}\n\n")
        f.write("## Confusion Matrix\n\n")
        f.write(f"`{results['confusion_matrix']}`\n")

    print("\nResults saved in:")
    print("- results/test_metrics_from_api.json")
    print("- results/test_metrics_from_api.md")


if __name__ == "__main__":
    main()