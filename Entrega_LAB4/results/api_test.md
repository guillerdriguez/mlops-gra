# Prueba de despliegue del modelo mediante API

El modelo entrenado fue desplegado localmente utilizando MLflow.

## Comando utilizado

mlflow models serve -m "models:/diabetes_classifier/1" -p 8000 --no-conda

## Salida del servidor

Uvicorn running on http://127.0.0.1:8000

## Petición de prueba

curl -X POST http://127.0.0.1:8000/invocations \
-H "Content-Type: application/json" \
-d '{
"dataframe_split": {
"columns": ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"],
"data": [[6,148,72,35,0,33.6,0.627,50]]
}
}'

## Respuesta obtenida

[1]

## Conclusión

La API devolvió correctamente una predicción para los datos enviados, confirmando que el modelo se desplegó y funciona correctamente mediante el servicio de inferencia de MLflow.