# 📂 Carpeta `data/`

Esta carpeta contiene todos los **datos de entrada** utilizados en el prototipo de clasificación de poses, así como los archivos generados durante el preprocesamiento.

---

## 📁 Estructura interna

```bash
data/
├── csv/                  # Archivos CSV listos para entrenamiento, validación y prueba
│   ├── test_csv/         # CSVs del conjunto de prueba
│   └── train_val_csv/    # CSVs de entrenamiento y validación
│
├── fotos/                # Imágenes utilizadas para pruebas rápidas de predicción
│
├── processed_videos/     # Videos .mp4 listos para ser procesados por MediaPipe Pose
│   ├── PruebasReales/         # Videos .mp4 para probar el modelo
│   ├── test/         # Videos .mp4 para preparar el conjunto de prueba
│   └── train_val/    # Videos .mp4 para preparar el conjunto de entrenamiento y validación
│
└── README.md             # Este archivo
```
## 📋 Descripción de contenido
  - csv/ → Contiene los archivos generados a partir del procesamiento de videos. Esta carpeta y sus subcarmetas son creadas en [`Preparar_dataset.py`](../scripts/Preparar_dataset.py)
  - train_val_csv/ → CSVs para entrenamiento y validación del modelo.
  - test_csv/ → CSVs para evaluar el rendimiento final del modelo.
  - fotos/ → Imágenes individuales empleadas para pruebas rápidas con imagenes usando [`Tester.py`](../scripts/Tester.py).
  - processed_videos/ → Videos en formato .mp4 que ya han pasado por el filtrado de fondo mediante el script [`bag2mp4.py`](../scripts/bag2mp4.py).
    - Esta carpeta está disponible para descargar desde [Google Drive](https://drive.google.com/drive/folders/1kxPISQjgr6vX_-z4FhiXL4QVWJOGZun4?usp=sharing).
