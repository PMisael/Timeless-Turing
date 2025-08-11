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
