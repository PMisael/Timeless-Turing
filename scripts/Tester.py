from keras.models import load_model
from scripts.Analizador import Video, Imagen
import json # Cargar el Json con mapeo de etiquetas - vectores One Hot
from pathlib import Path

class Tester:
    def __init__(self):
        self.model      =None
        self.media      =None
        self.etiquetas  =None
    #
    def Carga_modelo(self, model_path, etiquetas_json_path):
        with open(etiquetas_json_path, 'r') as f:
            self.etiquetas = json.load(f)
        self.model         = load_model(model_path)
    #
    def Predice_video(self, video_path: str, reproducir: bool):
        self.media         = Video(path=video_path,train=False)
        nombre_csv         = Path(video_path).stem
        salida_csv         = "resultados/"+nombre_csv+".csv"
        self.media.Prediccion(model            = self.model, 
                              mapeo_etiquetas  = self.etiquetas,
                              csv_path         = salida_csv,
                              step             = 3,
                              reproduce        = reproducir)
    #
    def Predice_imagen(self, image_path: str, muestra: bool):
        self.media         = Imagen(image_path)
        self.media.Predice(model            = self.model,
                           mapeo_etiquetas  = self.etiquetas,
                           muestra          = muestra)


def main():
    VIDEO_PATH          = "data/processed_videos/PruebasConSujetos/Sujeto3_2.mp4"
    PHOTO_PATH          = "data/fotos/ChatGPT_BrazosCruzados.png"
    MODEL_PATH          = "models/best_model_8.keras"
    ETIQUETAS_JSON_PATH = "models/mapeo_etiquetas.json"
    #
    tester=Tester()
    tester.Carga_modelo(MODEL_PATH, ETIQUETAS_JSON_PATH)
    tester.Predice_video(VIDEO_PATH, reproducir=True)
    #tester.Predice_imagen(PHOTO_PATH, muestra=True)
#
#
if __name__=='__main__':
    main()
