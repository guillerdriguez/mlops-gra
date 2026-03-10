SVM (rank 1)
Accuracy 0.753
Precision 0.603
Recall 0.870
F1 0.712
FN = 7
FP = 31

Random Forest (rank 2)
Accuracy 0.831
Precision 0.719
Recall 0.852
F1 0.780
FN = 8
FP = 18

Logistic Regression (rank 3)
Accuracy 0.779
Precision 0.647
Recall 0.815
F1 0.721
FN = 10
FP = 24

Análisis técnico:

El modelo SVM obtiene el mayor recall (0.87), lo que implica que detecta más pacientes con diabetes y produce menos falsos negativos (7). En un contexto médico esto es un criterio relevante, ya que un falso negativo implica no detectar un caso potencial de enfermedad.

Sin embargo, el SVM genera un número considerable de falsos positivos (31), lo que reduce la precisión y podría implicar un mayor número de pacientes enviados innecesariamente a pruebas adicionales.

El modelo Random Forest presenta el mejor equilibrio global entre métricas. Aunque su recall (0.852) es ligeramente inferior al del SVM, mantiene un valor elevado mientras reduce significativamente los falsos positivos (18 frente a 31). Además obtiene el mejor F1-score (0.7797) y la mayor accuracy.

Por tanto, desde una perspectiva de compromiso entre sensibilidad diagnóstica y estabilidad del modelo, Random Forest aparece como una opción muy sólida.
