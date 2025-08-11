# 🧩 Carpeta `scripts/`

Scripts principales del proyecto: **preprocesamiento**, **preparación de datos**, **entrenamiento** y **predicción**.  
Esta guía explica **qué hace cada script** y **cómo usarlo**.

---

## 📑 Tabla de contenidos
- [Requisitos y dependencias](#-requisitos-y-dependencias)
- [Flujo de trabajo (overview)](#-flujo-de-trabajo-overview)
- [Estructura de archivos](#-estructura-de-archivos)
- [1) `bag2mp4.py` — Conversión y filtrado de .bag](#1-bag2mp4py--conversión-y-filtrado-de-bag)
- [2) `MpPose.py` — Conversión y filtrado de .bag](#1-bag2mp4py--conversión-y-filtrado-de-bag)
- [3) `Analizador.py` — Conversión y filtrado de .bag](#1-bag2mp4py--conversión-y-filtrado-de-bag)
- [4) `Preparar_dataset.py` — Construcción del dataset](#2-preparar_datasetpy--construcción-del-dataset)
- [5) `Entrenamiento.py` — Modelo y entrenamiento](#3-entrenamientopy--modelo-y-entrenamiento)
- [6) `Tester.py` — Predicción en video/imagen](#4-testerpy--predicción-en-videoimagen)
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



# README.md anterior (ignora)
Instalación sugerida (ver también `requirements.txt`):
```bash
pip install -r requirements.txt



# Parámetros clave:

--video.bag

--distancia mínima para el recorte (metros)

--distancia máxima para el recorte (metros)

## Uso rápido
### Nombre por defecto (cambia extension .bag a .mp4)
```bash
python bag2mp4.py video.bag \
                  --min 0.5 \
                  --max 1.0 \
```

### Nombre y ruta personalizada del video mp4 resultante

```bash
python bag2mp4.py video.bag \
                  ruta/del/video_sin_fondo.mp4 \
                  --min 0.5 \
                  --max 1.0 \
```

La salida muestra la ruta del video sin fondo creado en formato mp4 el cual tiene el mismo nombre que el archivo.bag procesado.
