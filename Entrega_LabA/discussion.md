# LabA – Análisis de Deriva del Dato con Evidently AI

## Dataset

Se utiliza el dataset del Titanic (891 pasajeros, 13 columnas), disponible a través de seaborn. La variable objetivo es `survived`. Las columnas incluyen variables numéricas (`age`, `fare`, `sibsp`, `parch`) y categóricas (`pclass`, `sex`, `embarked`, `class`, `who`, `adult_male`, `embark_town`, `alone`). La columna `deck` fue excluida por tener un 77% de valores nulos, lo que provocaba errores en Evidently AI con splits muy pequeños (98/1/1).

## Condiciones experimentales

Se generaron **12 condiciones** de división (train/val/test), combinando:

- **Estratificación**: sí / no (sobre la variable `survived`)
- **Proporciones de split**: 60/20/20 · 90/5/5 · 98/1/1
- **Semilla aleatoria**: 42 · 123

Para cada condición se generaron 2 reports de deriva HTML (val y test), totalizando **24 reports**.

---

## Tabla resumen de resultados

| Condición | Estratif. | Split ratio | Semilla | N train | N val | N test | Drift val | Drift test |
|---|:-:|:-:|:-:|--:|--:|--:|:-:|:-:|
| 60-20-20_strat_seed42 | Sí | 60/20/20 | 42 | 534 | 178 | 179 | **0%** | **0%** |
| 60-20-20_strat_seed123 | Sí | 60/20/20 | 123 | 534 | 178 | 179 | **7.69%** | **0%** |
| 90-5-5_strat_seed42 | Sí | 90/5/5 | 42 | 801 | 45 | 45 | **0%** | **0%** |
| 90-5-5_strat_seed123 | Sí | 90/5/5 | 123 | 801 | 45 | 45 | **0%** | **15.38%** |
| 98-1-1_strat_seed42 | Sí | 98/1/1 | 42 | 873 | 9 | 9 | **0%** | **0%** |
| 98-1-1_strat_seed123 | Sí | 98/1/1 | 123 | 873 | 9 | 9 | **15.38%** | **0%** |
| 60-20-20_nostrat_seed42 | No | 60/20/20 | 42 | 534 | 178 | 179 | **0%** | **0%** |
| 60-20-20_nostrat_seed123 | No | 60/20/20 | 123 | 534 | 178 | 179 | **0%** | **0%** |
| 90-5-5_nostrat_seed42 | No | 90/5/5 | 42 | 801 | 45 | 45 | **15.38%** | **0%** |
| 90-5-5_nostrat_seed123 | No | 90/5/5 | 123 | 801 | 45 | 45 | **0%** | **7.69%** |
| 98-1-1_nostrat_seed42 | No | 98/1/1 | 42 | 873 | 9 | 9 | **0%** | **0%** |
| 98-1-1_nostrat_seed123 | No | 98/1/1 | 123 | 873 | 9 | 9 | **0%** | **7.69%** |

*Drift: fracción de columnas con deriva detectada (umbral por defecto de Evidently, p-value < 0.05 para numéricas, chi-cuadrado para categóricas).*

---

## Discusión

### Efecto del tamaño del split

El parámetro que más claramente condiciona la aparición de deriva es el **tamaño de los conjuntos de validación y test**. Con el split 60/20/20, val y test contienen ~178-179 muestras cada uno; con 90/5/5, ~45; y con 98/1/1, apenas 9. A medida que los conjuntos se reducen, la varianza muestral aumenta enormemente: con sólo 9 observaciones, es perfectamente posible que una columna categórica como `pclass` o `sex` muestre una distribución muy alejada de la del conjunto de entrenamiento, simplemente por azar.

En ese sentido, los resultados muestran un patrón coherente: la deriva detectada aparece preferentemente en condiciones con splits pequeños (90/5/5 y 98/1/1) y casi nunca en el split más equilibrado (60/20/20). Esto sugiere que **gran parte de la deriva observada es estadística (ruido muestral)** y no una señal real de cambio de distribución.

### Efecto de la estratificación

La estratificación sobre `survived` garantiza que la proporción de supervivientes se mantiene igual en train, val y test. Sin embargo, esto sólo afecta directamente a la variable objetivo; el resto de columnas continúan siendo asignadas de forma aleatoria dentro de cada estrato.

Los resultados muestran que la estratificación **no elimina la posibilidad de detectar deriva**: aparecen casos con deriva tanto con estratificación como sin ella. Esto es esperable porque las demás columnas no están estratificadas. Lo que sí aporta la estratificación es una mayor robustez en la distribución de la variable objetivo, lo cual es importante para la evaluación de modelos, aunque no supone una garantía de ausencia de deriva en las features.

Un patrón notable es que, en el split 60/20/20, el caso con estratificación y semilla 123 presenta deriva en val (7.69%), mientras que sin estratificación y misma semilla no hay deriva. Esto puede parecer contraintuitivo, pero se explica porque al estratificar se reordenan los índices de los subconjuntos, generando splits con composición ligeramente diferente en las variables no estratificadas.

### Efecto de la semilla aleatoria

Los resultados difieren entre semilla 42 y semilla 123 para el mismo ratio y estratificación. Esto ilustra que **la semilla puede ser determinante en splits pequeños**: con muy pocas muestras, la elección concreta de qué observaciones van a val/test puede cambiar radicalmente la distribución observada. En splits grandes (60/20/20), el efecto de la semilla es mínimo; en splits extremos (98/1/1), la semilla puede hacer la diferencia entre detectar o no detectar deriva.

### Asimetría val vs. test

En la mayoría de condiciones, cuando se detecta deriva sólo lo hace en uno de los dos conjuntos (val o test), raramente en ambos. Esto es consistente con la explicación de ruido muestral: con muestras pequeñas, es un evento aleatorio que sólo uno de los dos splits caiga en una región estadísticamente "lejana" del entrenamiento.

---

## Conclusión general

En este experimento, la deriva detectada por Evidently AI entre los conjuntos de entrenamiento y validación/test del dataset Titanic es, en la gran mayoría de los casos, un **artefacto del tamaño reducido de los conjuntos de evaluación** y no una deriva real de los datos. Cuando los splits son suficientemente grandes (60/20/20), la deriva es prácticamente inexistente o marginal. Cuando los splits se reducen hasta el 1%, la varianza muestral dispara falsos positivos.

Desde el punto de vista práctico, estos resultados subrayan varias lecciones importantes:

1. **Un split demasiado pequeño es un riesgo**: conjuntos de val/test con pocas decenas de muestras son estadísticamente poco representativos y pueden generar señales de deriva espurias o, al contrario, no detectar deriva real.

2. **La estratificación es necesaria pero no suficiente**: estratificar sobre la variable objetivo garantiza balanceo de clases pero no asegura representatividad en todas las features.

3. **La semilla importa (especialmente con muestras pequeñas)**: en producción, es buena práctica reportar resultados promediados sobre múltiples semillas para distinguir señal de ruido.

4. **Para datasets pequeños como el Titanic**, el ratio 60/20/20 con estratificación es la opción más robusta: minimiza la varianza muestral y garantiza que train y val/test sean estadísticamente comparables.
