# Métricas del modelo obtenidas a través de la API de MLflow

Modelo servido: diabetes_classifier versión 3

Endpoint utilizado: http://127.0.0.1:8001/invocations

Número de muestras evaluadas: 77

## Métricas obtenidas

Accuracy: 0.7403  
Precision: 0.6061  
Recall: 0.7407  
F1-score: 0.6667  

## Matriz de confusión

[[37, 13],  
 [7, 20]]

## Interpretación

El modelo identifica correctamente la mayoría de los casos positivos de diabetes, alcanzando un recall cercano al 74 %. Esto significa que detecta aproximadamente tres de cada cuatro pacientes diabéticos presentes en el conjunto de test.

Aunque la precisión es moderada, el modelo mantiene un equilibrio razonable entre sensibilidad y número de falsos positivos, lo que resulta aceptable para un sistema de detección preliminar.