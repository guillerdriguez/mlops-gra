Durante este trabajo se desarrolló un pipeline completo de experimentación y despliegue de modelos de clasificación utilizando MLflow. A partir del dataset PIMA Diabetes se entrenaron varios modelos de aprendizaje automático con el objetivo de predecir la presencia de diabetes en pacientes a partir de variables clínicas.

El proceso comenzó con un experimento de prueba en el que se entrenó un modelo baseline de regresión logística, se registró en MLflow y se desplegó mediante una API local para verificar su funcionamiento. Este paso permitió validar el flujo completo de entrenamiento, registro y servicio del modelo.

Posteriormente se llevaron a cabo experimentos completos utilizando una partición estratificada del dataset en tres subconjuntos: entrenamiento (70 %), validación (20 %) y test (10 %). Sobre el conjunto de entrenamiento y validación se aplicó una búsqueda de hiperparámetros mediante GridSearchCV para tres familias de modelos: Logistic Regression, Random Forest y Support Vector Machine. La selección de los modelos finalistas se realizó considerando múltiples métricas (accuracy, precision, recall y F1-score), priorizando el recall de la clase positiva debido a la naturaleza médica del problema, donde los falsos negativos tienen mayor impacto.

Los modelos seleccionados se registraron en MLflow Model Registry, permitiendo gestionar diferentes versiones del modelo y facilitar su despliegue posterior. Entre ellos, el modelo Random Forest presentó el mejor equilibrio entre sensibilidad y precisión, por lo que fue elegido como modelo final para su evaluación en el conjunto de test.

El modelo final fue desplegado mediante la API de MLflow y evaluado enviando todo el conjunto de test a través del endpoint /invocations. Este procedimiento permite reproducir un escenario de inferencia real, en el que las predicciones se obtienen a través de un servicio de modelo en lugar de realizarse directamente en el entorno de entrenamiento.

Los resultados obtenidos en el conjunto de test fueron:

Accuracy: 0.7403
Precision: 0.6061
Recall: 0.7407
F1-score: 0.6667

La matriz de confusión mostró que el modelo identifica correctamente la mayoría de los casos positivos, con un recall cercano al 74 %. Aunque la precisión es moderada, el modelo mantiene un compromiso razonable entre sensibilidad y número de falsos positivos, lo que resulta adecuado para un sistema de cribado preliminar.

En conjunto, el trabajo demuestra cómo MLflow permite estructurar de forma reproducible todo el ciclo de vida de un modelo de machine learning: experimentación, seguimiento de métricas, registro de modelos, despliegue mediante API y evaluación final en un entorno de inferencia.
