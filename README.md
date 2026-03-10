Inicio del repositorio de la asignatura MLOps. 

# Entrega 1

# Entrega 2
Para la entrega 2, en /Entrega_LAB2/notebooks hay un jupyter con la solucion de ambas task, incluyendo EDA. Ademas en /Entrega_LAB2/src estan ambas soluciones por separado. 

# Entrega 3
En la carpeta ENTREGA_LAB3/entregables se encuentra todo el material solicitado en la práctica.

Los ejercicios realizados antes de modificar el main (el archivo original es main_1.py) están identificados como “ejercicioX…txt”, junto con el documento PREGUNTAS.PDF, donde se incluyen las respuestas teóricas.

El archivo actual main.py corresponde a la versión modificada para el ejercicio de programación (política de activación configurable). Todo lo que comienza por “ejProgram…” está relacionado con esta parte, incluyendo el fichero de historial generado y el documento con las respuestas asociadas.

Además, el razonamiento previo exigido antes de implementar la modificación del código se encuentra en el archivo de texto correspondiente dentro de la misma carpeta.

# Entrega 4

## Descripción

En esta práctica se desarrolla un pipeline completo de experimentación, registro, despliegue y evaluación de modelos de machine learning utilizando **MLflow**.

El objetivo es entrenar modelos de clasificación capaces de predecir la presencia de diabetes a partir del dataset **PIMA Diabetes**, comparar distintos algoritmos y desplegar el modelo final mediante una API para evaluar su rendimiento en un escenario de inferencia real.

---

# Dataset

Se utiliza el dataset **PIMA Indians Diabetes Database**, que contiene información clínica de pacientes y una variable objetivo binaria que indica la presencia o ausencia de diabetes.

Variables principales del dataset:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Variable objetivo:

- **Outcome** (0 = no diabetes, 1 = diabetes)

---

# Flujo de trabajo

El proceso seguido en la práctica se divide en varias etapas.

## 1. Experimento de prueba

Se entrena un modelo baseline utilizando **Regresión Logística** para validar el flujo de trabajo completo:

- entrenamiento del modelo
- registro de métricas en MLflow
- registro del modelo en MLflow Model Registry
- despliegue del modelo mediante API

Las métricas obtenidas en este modelo baseline fueron:

| Métrica | Valor |
|------|------|
| Accuracy | 0.7338 |
| Precision | 0.6032 |
| Recall | 0.7037 |
| F1-score | 0.6496 |

---

## 2. Experimentos completos

Posteriormente se realizaron experimentos completos utilizando una partición del dataset:

- **70 % entrenamiento**
- **20 % validación**
- **10 % test**

Se compararon tres familias de modelos:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

Para cada modelo se aplicó **GridSearchCV** para optimizar hiperparámetros.

La selección de los mejores modelos se realizó considerando varias métricas:

- Accuracy
- Precision
- Recall
- F1-score

Dado que se trata de un problema médico, se priorizó el **recall de la clase positiva**, ya que los falsos negativos tienen mayor impacto clínico.

---

## 3. Registro de modelos en MLflow

Los modelos finalistas se registraron en **MLflow Model Registry**, permitiendo gestionar distintas versiones del modelo y facilitar su despliegue posterior.

Los resultados de validación se encuentran en: