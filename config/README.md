## 📦 Librerías principales

El proyecto fue desarrollado en un entorno **conda** y las dependencias exactas están listadas en [`requirements.txt`](./requirements.txt).  
A continuación, se listan las más relevantes para el desarrollo:

- **TensorFlow** (`tensorflow`, `tensorboard`, `tensorflow-io-gcs-filesystem`) – Entrenamiento y despliegue de redes neuronales.
- **MediaPipe** (`mediapipe`) – Detección y seguimiento de puntos clave del cuerpo.
- **OpenCV** (`opencv-python`, `opencv-contrib-python`, `opencv-python-headless`) – Procesamiento y manipulación de imágenes y video.
- **Pandas** y **NumPy** – Manejo y análisis de datos.
- **Matplotlib** y **Plotly** – Visualización de datos y resultados.
- **Scikit-learn** y **Scikit-image** – Herramientas para análisis, métricas y procesamiento de imágenes.
- **RealSense** (`pyrealsense2`, `librealsense`) – Soporte para cámaras Intel RealSense.
---
## ⚙️ Configuración de Visual Studio Code

El repositorio incluye una carpeta `.vscode` con la configuración de entorno y notebooks:

```json
{
    "python.envFile": "${workspaceFolder}/.env",
    "jupyter.envFile": "${workspaceFolder}/.env",
    "jupyter.notebookFileRoot": "${workspaceFolder}"
}
```
---
Esto asegura que:
  - Los notebooks de Jupyter usen las variables definidas en .env.
  - El entorno de Python sea consistente tanto en scripts como en notebooks.
  - La raíz para notebooks sea la carpeta del proyecto.
## 🚀 Uso rápido de requirements.txt

``` bash
conda create --name poses --file requirements.txt
conda activate poses
```
