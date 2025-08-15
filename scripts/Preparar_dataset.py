import pandas as pd
from pathlib import Path
from scripts.Analizador import Video
#
class Preparar_dataset:
    def __init__(self):
        self.data           =Path("data").resolve()
        self.data_test      =None
        self.data_train_val =None
    #
    def Crea_carpetas(self):
        # Datos para Entrenamiento y Validación
        self.data_train_val = self.data/"csv/train_val_csv"
        Path(self.data_train_val).mkdir(parents=True, exist_ok=True)
        # Datos para Test
        self.data_test      = self.data/"csv/test_csv"
        Path(self.data_test).mkdir(parents=True, exist_ok=True)
    #
    def Extrae_frames(self):
        videos = [p.resolve() for p in (self.data/"processed_videos").rglob("*.mp4") if p.parent.stem!="PruebasConSujetos"]
        for video in videos:
            print(video,video.stem)
        #
        for path in videos:
            label = path.stem.upper()
            ruta_salida=str((self.data_train_val if path.parent.stem=='train_val' else self.data_test)/path.stem)+".csv"
            #
            print(f'\n\nProcesando video:  {str(path)} con etiqueta {label}, guardando en {ruta_salida}\n\n')
            video=Video(path, True)
            video.Extrae_frames(label,ruta_salida,step=5,reproduce=False)
    #
    def une_csvs(self):
        Salida    = Path("dataset_completo.csv")
        for carpeta in ([self.data_test,self.data_train_val]):
            csvs = [p for p in self.data.rglob("*.csv") if p.parent==carpeta and p.stem!=Salida.stem]
            df   = pd.concat((pd.read_csv(f) for f in csvs),ignore_index=True)
            df   = df.drop_duplicates().reset_index(drop=True)
            df.to_csv(carpeta/Salida, index=False, float_format="%.6f")
#
#
#
def main():
    dataset = Preparar_dataset()
    dataset.Crea_carpetas()
    dataset.Extrae_frames()
    dataset.une_csvs()

if __name__ == "__main__":
    main()
