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
- [4) `Preparar_dataset.py` — Construcción del dataset](#4-preparar_datasetpy--construcción-del-dataset)
- [5) `Entrenamiento.py` — Modelo y entrenamiento](#5-entrenamientopy--modelo-y-entrenamiento)
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
**Propósito:** convertir grabaciones de Intel RealSense (.bag) a .mp4 aplicando filtro por rango de profundidad para eliminar fondo y dejar solo al sujeto.
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
- `Clase Imagen`:
  - Inicializar (`__init__()`) → carga una imagen desde la ruta indicada y prepara MpPose en modo imagen estática.
  - Predecir pose en imagen (`Predice()`) → procesa la imagen para:
    - Detectar los 33 puntos clave del cuerpo.
    - Predecir la pose con el modelo entrenado.
    - Mostrar en pantalla la imagen con el nombre de la pose detectada, si se solicita (muestra=True).
---
## 4) [Preparar_dataset.py](./Preparar_dataset.py) — Construcción del dataset
**Propósito:** Automatizar la generación de datasets a partir de videos .mp4 ya preprocesados (sin fondo): crea carpetas de salida, extrae coordenadas (x,y,z) de los 33 puntos (99 features) usando [Analizador.Video](#3-analizadorpy--manipulador-de-imagenes-y-videos), genera CSVs por video, y, al final, produce un CSV unificado por partición (train/val y test).
### 🧠 Qué hace
Recorre los videos procesados, extrae automáticamente las coordenadas de todos los puntos detectados por MediaPipe Pose, guarda un CSV para cada video y finalmente combina los resultados en un único archivo por partición (entrenamiento/validación y prueba). Está diseñado para estructurar los datos de forma uniforme y lista para ser utilizada en el entrenamiento del modelo.
- `__init__()` → Inicializa rutas base (data/) y placeholders para data_test y data_train_val.
- `Crea_carpetas()` → Crea (si no existen) las rutas:
  - [data/csv/train_val_csv/](../data/csv/train_val_csv).
  - [data/csv/test_csv/](../data/csv/test_csv).
- `Extrae_frames()`:
  - Busca todos los .mp4 dentro de data/processed_videos/, ignorando la subcarpeta llamada PruebasReales.
  - La etiqueta se toma del nombre del archivo (path.stem.upper()).
  - Decide la carpeta de salida en función de la ubicación del video:
    - Si la subcarpeta en la que se encuentra se llama train_val → guarda en train_val_csv/
    - En cualquier otro caso → guarda en test_csv/
  - Para cada video:
    - Instancia `Video(path, train=True)`.
    - Llama a `Video.Extrae_frames(label, ruta_salida, step=5, reproduce=False)` para generar el CSV.
    - Solo guarda frames con los 33 puntos detectados con un `paso` de 5 frames.
- `Une_csvs()` → Para cada carpeta (train_val_csv/ y test_csv/):
  - Concatena todos los CSVs (excepto dataset_completo.csv cuando ya hay uno generado).
  - Elimina duplicados.
  - Guarda un CSV unificado llamado dataset_completo.csv en esa carpeta.
- `main()` → Ejecuta la secuencia completa:
  - Crea_carpetas()
  - Extrae_frames()
  - Une_csvs().
### ▶️ Uso (CLI)
El script puede ejecutarse directamente desde la terminal como paquete:
``` bash
python -m scripts.Preparar_dataset.py
```
### 🔍 Notas y consideraciones
- `step=5` en `Extrae_frames()` controla cada cuántos frames se extraen features (puedes ajustarlo dentro del script si necesitas más/menos densidad).
- Solo se guardan registros cuando MediaPipe detecta los 33 puntos (calidad ajustable).
- La etiqueta depende estrictamente del nombre de archivo; evita espacios raros y mantén una convención clara.
- `Une_csvs()` elimina duplicados (útil si hay solapamientos).

### ⚠️ Errores comunes
- No genera CSVs → Revisa que processed_videos/ tenga .mp4 fuera de PruebasReales/ y que la detección de puntos funcione (iluminación/encuadre).
- Todo cae en test_csv/ → Asegúrate de que los videos para entrenamiento estén bajo una carpeta padre llamada train_val.
- Clases inesperadas → El nombre del video define la etiqueta. Renombra los archivos si es necesario.
---
## 5) [Entrenamiento.py](.Entrenamiento.py) — Modelo y entrenamiento
**Propósito:** Entrenar y evaluar un clasificador multiclase a partir de las 99 características por frame, guardando el mejor modelo y registrando métricas y gráficas para análisis de desempeño.
### 🧠 Qué hace
Este módulo carga los CSV unificados (`dataset_completo.csv`) de train/val y test, convierte las etiquetas a one-hot, construye una red neuronal densa (MLP) con BatchNormalization y Dropout, entrena con early saving del mejor modelo (por val_loss), evalúa en el conjunto de prueba y grafica la historia de entrenamiento (accuracy y loss).
- `__init__()` → Inicializa dataframes de entrada, matrices X/Y para train/val/test, el modelo Keras y un LabelBinarizer para codificar etiquetas.
- `cargar_datos()` → Lee:
  - data/csv/train_val_csv/dataset_completo.csv
  - data/csv/test_csv/dataset_completo.csv
- `dividir_datos()`:
  - Separa features y labels en train/val; hace one-hot con LabelBinarizer.
  - Realiza train_test_split estratificado (test_size=0.20, random_state=42).
  - Prepara X_test/Y_test desde el CSV de test y codifica sus etiquetas.
  - Guarda el mapeo índice → clase en [models/mapeo_etiquetas.json](../models/mapeo_etiquetas.json).
- `construccion()`:
Crea un *Red Neuronal Densa*:
Input(n_features) → Dense(128, relu) → BatchNorm → Dropout(0.30) →
Dense(64, relu) → BatchNorm → Dropout(0.30) → Dense(n_classes, softmax)
Compila con Adam(lr=1e-3), categorical_crossentropy, accuracy.
- `entrenamiento()`:
  - Autonumera models/best_model_*.keras según los existentes.

Usa ModelCheckpoint(..., save_best_only=True, monitor='val_loss').

Entrena epochs=100, batch_size=256, validando con (X_val, Y_val).

Devuelve history (historial de métricas).

evaluacion()

Imprime Accuracy en test.

Calcula confusion_matrix y classification_report (usando poses_lb.classes_).

graficar_entrenamiento(history)
Muestra dos gráficas (accuracy y loss) con curvas de train y val.

getX_test_Y_test_poses_lb()
Devuelve (X_test, Y_test, poses_lb) para comparativas externas (útil en el notebook).

main()
Pipeline end-to-end: cargar_datos() → dividir_datos() → construccion() → entrenamiento() → evaluacion() → graficar_entrenamiento().
