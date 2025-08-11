# 📂 Carpeta `models/`

Esta carpeta almacena los **modelos entrenados** del prototipo de clasificación de poses con [`Entrenamiento.py`](../scripts/Entrenamiento.py), así como un archivo json auxiliar necesarios para mapear las predicciones a sus etiquetas reales.

---

## 📁 Estructura interna

```bash
models/
├── best_model_0.keras      # Modelo entrenado (mejor epoca del entrenamiento 0)
├── best_model_1.keras      
├── best_model_n.keras     
├── mapeo_etiquetas.json    # Diccionario para traducir índices de clase a nombres de poses
└── README.md               # Este archivo
```
## 📋 Descripción de contenido
- **best_model_XX.keras:**
  - Archivos del modelo en formato Keras, guardados durante el entrenamiento cuando se alcanzó la mejor métrica de validación.
  - Incluyen arquitectura, pesos y configuración de entrenamiento.
  - El número *(XX)* corresponde al numero de entrenamiento donde se la mejor epoca del entrenamiento. Cada vez que se ejecuta [`Entrenamiento.py`](../scripts/Entrenamiento.py) se genera un nuevo *best_model_XX.keras*
- **mapeo_etiquetas.json:**
  - Archivo JSON que mapea cada índice numérico de salida del modelo a su etiqueta correspondiente (nombre de la pose).
