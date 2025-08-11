# 🧩 Carpeta `scripts/`

Scripts principales del proyecto: **preprocesamiento**, **preparación de datos**, **entrenamiento** y **predicción**.  
Esta guía explica **qué hace cada script** y **cómo usarlo**.

---

## 📑 Tabla de contenidos
- [Requisitos y dependencias](#-requisitos-y-dependencias)
- [Flujo de trabajo (overview)](#-flujo-de-trabajo-overview)
- [1) `bag2mp4.py` — Conversión y filtrado de .bag](#1-bag2mp4py--conversión-y-filtrado-de-bag)
- [2) `MpPose.py` — Uso de Mediapipe Pose para extraer puntos del cuerpo](#2-mpposepy--uso-de-mediapipe-pose-para-extraer-puntos-del-cuerpo)
- [3) `Analizador.py` — Manipulador de imagenes y videos](#3-analizadorpy--manipulador-de-imagenes-y-videos)
- [4) `Preparar_dataset.py` — Construcción del dataset](#4-preparar_dataset.py--construcción-del-dataset)
- [5) `Entrenamiento.py` — Modelo y entrenamiento](#5-entrenamiento.py--modelo-y-entrenamiento)
- [6) `Tester.py` — Predicción en video/imagen](#6-tester.py--predicción-en-videoimagen)
- [Convenciones de nombres y rutas](#-convenciones-de-nombres-y-rutas)
- [FAQ (errores comunes)](#-faq-errores-comunes)

---

## 🔩 Requisitos y dependencias

- Python 3.10+
- TensorFlow ≥ 2.15
- MediaPipe ≥ 0.10
- OpenCV (`opencv-python`)
- NumPy, Pandas, scikit-learn
- **Intel RealSense SDK** (`pyrealsense2`) para `bag2mp4.py` (Ubuntu)
---
## 🧭 Flujo de trabajo (overview)
``` bash
(Grabación .bag)  →  bag2mp4.py  →  data/processed_videos/*.mp4
                              ↓
                       Preparar_dataset.py  →  data/csv/train_val_csv/*.csv, data/csv/test_csv/*.csv
                              ↓
                        Entrenamiento.py  →  models/best_model_XX.keras, models/mapeo_etiquetas.json
                              ↓
                            Tester.py  →  resultados/*.csv
```
---
## 1) bag2mp4.py — Conversión y filtrado de .bag
Propósito: convertir grabaciones de Intel RealSense (.bag) a .mp4 aplicando filtro por rango de profundidad para eliminar fondo y dejar solo al sujeto.
### 🧠 Qué hace
- Lee frames RGB + Depth de .bag.
- Enmascara pixeles fuera de [min, max].
- Exporta .mp4 limpio (sin fondo).
### ▶️ Uso (CLI)
- Parámetros clave:
  - --video.bag
  - --distancia mínima para el recorte (metros)
  - --distancia máxima para el recorte (metros)
**Uso rapido**
```bash
python bag2mp4.py video.bag \
                  --min 0.5 \
                  --max 1.0 \
```
**Nombre y ruta personalizada del video mp4 resultante**

```bash
python bag2mp4.py video.bag \
                  ruta/del/video_sin_fondo.mp4 \
                  --min 0.5 \
                  --max 1.0 \
```

La salida muestra la ruta del video sin fondo creado en formato mp4, el cual, por defecto tiene el mismo nombre que el archivo.bag procesado.

---

## 2) [MpPose.py](./MpPose.py) — Uso de Mediapipe Pose para extraer puntos del cuerpo

**Propósito:** Detectar puntos del cuerpo y extraer coordenadas 3D con MediaPipe Pose.
### 🧠 Qué hace
Implementa una clase auxiliar que envuelve la funcionalidad de MediaPipe Pose para trabajar de forma sencilla con detección de poses tanto en imágenes estáticas como en videos.

Sus principales funciones son:
  - Inicializar el modelo de MediaPipe Pose en modo imagen estática o en modo video, con model_complexity=2 para mayor precisión.
  - Procesar frames (`process()`) y, opcionalmente, dibujar la silueta y conexiones de los 33 puntos clave del cuerpo humano.
  - Contar puntos visibles (`cuenta_puntos()`) filtrando por un umbral mínimo de visibilidad, útil para descartar frames incompletos o con detección deficiente.
  - Extraer valores (x, y, z) (`extrae_valores()`) de cada punto detectado, devolviendo un vector de 99 características, y añadir etiqueta (label) si está en modo entrenamiento.
  - Generar nombres de columnas (`create_lists()`) para los datasets CSV, combinando el nombre anatómico del punto y sus coordenadas.
---
## 3) [Analizador.py](./Analizador.py) — Manipulador de imagenes y videos
**Propósito:** Servir como herramienta para la etapa de predicción y generación de datasets, permitiendo procesar videos e imágenes mediante MediaPipe Pose. 

### 🧠 Qué hace
Implementa dos clases auxiliares (Video e Imagen) que utilizan la clase MpPose para procesar videos e imágenes en la etapa de predicción y extracción de datos para entrenamiento.

Sus principales funciones son:
- Clase `Video`:
  - Inicializar (`__init__()`) → carga un video desde la ruta indicada, configura si está en modo entrenamiento (train=True) y prepara un detector de poses MpPose.
  - Predecir poses en video (`Prediccion()`) → procesa un video completo con un modelo entrenado:
    - Analiza cada cierto número de frames (step) para optimizar tiempo de ejecución.
    - Extrae coordenadas (x, y, z) con MpPose.
    - Predice la pose usando el modelo.
    - Guarda los resultados en un CSV con columnas: Segundo, Pose.
  - Extraer frames para dataset (`Extrae_frames()`) → procesa un video etiquetado para obtener coordenadas (x, y, z) de cada uno de los 33 puntos del cuerpo:
    - Solo guarda frames donde se detectan los 33 puntos.
    - Genera un CSV con 99 columnas de coordenadas + 1 columna label.
  - Guardar CSV (`guardaCSV()`) → exporta la lista de resultados (self.data) a un archivo CSV, asignando nombres de columnas dependiendo si es modo entrenamiento o predicción.
