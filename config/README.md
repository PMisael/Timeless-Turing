## 📦 Librerías principales

El proyecto fue desarrollado en un entorno **conda** con las especificaciones de [spec-file.txt](./spec-file.txt) y las dependencias exactas están listadas en [`requirements.txt`](./requirements.txt).  
A continuación, se listan las más relevantes para el desarrollo:

- **TensorFlow** (`tensorflow`, `tensorboard`, `tensorflow-io-gcs-filesystem`) – Entrenamiento y despliegue de redes neuronales.
- **MediaPipe** (`mediapipe`) – Detección y seguimiento de puntos clave del cuerpo.
- **OpenCV** (`opencv-python`, `opencv-contrib-python`, `opencv-python-headless`) – Procesamiento y manipulación de imágenes y video.
- **Pandas** y **NumPy** – Manejo y análisis de datos.
- **Matplotlib** y **Plotly** – Visualización de datos y resultados.
- **Scikit-learn** y **Scikit-image** – Herramientas para análisis, métricas y procesamiento de imágenes.
- **RealSense** (`pyrealsense2`, `librealsense`) – Soporte para cámaras Intel RealSense.
---
## 🚀 Uso rápido de spec-file.txt y requirements.txt

``` bash
conda create --name nombre_entorno --file spec-file.txt
conda activate nombre_entorno
conda install pip
python -m pip install -r requirements.txt 
```
En este punto ya es posible ejecutar cualquier [Script](../scripts)
