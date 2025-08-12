# Clasificación Automática de Poses Corporales

Prototipo funcional para la **detección y clasificación automática de poses humanas** utilizando [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) y una red neuronal densa entrenada en TensorFlow.  

Este trabajo forma parte de mi **Proyecto Terminal** de Ingeniería Biomédica y constituye la **fase inicial** de un sistema más amplio para el análisis automatizado de **Movimientos Generales (GMA)** en neonatos, orientado a la **detección temprana de alteraciones neuromotoras**.  
En esta versión se utilizan videos de **adultos** como entorno de validación controlado para probar la viabilidad técnica del flujo completo.

---

## 🧠 Propósito

Validar un **pipeline reproducible** de adquisición, preprocesamiento, entrenamiento y predicción de poses, que posteriormente pueda adaptarse a datos de neonatos.  

Trabajar con adultos permite:
- Iterar y depurar rápidamente sin restricciones éticas o clínicas.
- Control total sobre las condiciones de grabación y etiquetado.
- Desarrollar una arquitectura y metodología lista para migrar a un contexto clínico.

---

## 📚 ¿Qué hace?
- **Captura de video**:
  - Uso de una librería de **RealSense** en Ubuntu para grabar con la cámara **[Intel RealSense D435](https://www.intel.la/content/www/xl/es/products/sku/128255/intel-realsense-depth-camera-d435/specifications.html)** mediante una interfaz gráfica (GUI).
  - Los videos se guardan en formato `.bag` (archivo nativo de RealSense con información de color y profundidad).

- **Preprocesamiento y eliminación de fondo**:
  - Conversión de `.bag` a `.mp4`.
  - Aplicación de filtrado por **rango de profundidad configurable**, eliminando el fondo y dejando únicamente al sujeto en primer plano.
  - Resultado: un video `.mp4` limpio, optimizado para análisis y entrenamiento.


- **Extracción de características**:
  - Obtención de coordenadas 3D `(x, y, z)` de **33 puntos clave** del cuerpo por frame con *MediaPipe Pose* (99 características totales por frame).
  - Generación de datasets CSV etiquetados para cada actividad/pose.

- **Preparación de dataset**:
  - Unificación de CSVs en un único archivo listo para Pandas.
  - División en *training*, *validation* y *test set*.

- **Entrenamiento**:
  - Red neuronal densa (MLP) con:
    - Batch Normalization
    - Dropout
    - Optimizador Adam
  - Entrenamiento supervisado multiclase.

- **Evaluación**:
  - Exactitud (*accuracy*), F1-score, matriz de confusión.
  - Comparativa entre modelos guardados.

- **Predicción**:
  - Videos (`.mp4`) y fotografías (`.jpeg`, `.png`).
  - Registro de predicciones con sello de tiempo para análisis temporal.

---

## 🗂️ Estructura del Proyecto
```bash

├── config/
│   └── README.md
│
├── data/
│   ├── csv/
│   │   ├── test_csv/
│   │   └── train_val_csv/
│   ├── fotos/
│   ├── processed_videos/ #Descarga desde Google Drive
│
├── models/ # Modelos entrenados (.keras) y mapeo de etiquetas (.json)
│
├── notebooks/
│   └── Paso_a_Paso.ipynb # Notebook explicando desde Preparacion de Dataset hasta Prueba de modelos
│
├── resultados/ # Directorio donde se guardaran los resultados del análisis de video
│
├── scripts/
│   ├── Preparar_dataset.py
│   ├── Entrenamiento.py
│   ├── Tester.py
│   └── bag2mp4.py
│
└── README.md
```
## 🛠️ Requisitos

    Python 3.10 o superior

    TensorFlow ≥ 2.15

    MediaPipe ≥ 0.10

    Pandas ≥ 2.0

    Scikit-learn ≥ 1.3

    NumPy ≥ 1.24
