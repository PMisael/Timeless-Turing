import cv2
import pandas as pd
import numpy as np
from scripts.MpPose import MpPose

class Video:
    def __init__(self, path: str, train: bool):
        self.path  = path
        self.train = train
        self.pose  = MpPose(image=False, Video_Train=self.train)
        self.frame = None
        self.data  = []
    #
    def Prediccion(self,model, mapeo_etiquetas, csv_path, step=30,reproduce=False):
        cap=cv2.VideoCapture(self.path)
        frame_rate=30
        frame_actual=0
        contador=0
        #
        while cap.grab():
            contador += 1
            frame_actual += 1
            if contador < step:
                continue
            #
            status,self.frame=cap.read()
            if not status:
                break
            #
            self.pose.process(self.frame,True)
            puntos=self.pose.cuenta_puntos(0.1)
            #
            if puntos==33 and contador>=step:
                X=np.array([self.pose.extrae_valores()])
                prediccion = model.predict(X)
                prediccion = (np.argmax(prediccion, axis=1))
                label      = mapeo_etiquetas[str(prediccion[0])]
                msg        = f'Pose: {label}'
                segundo    = frame_actual/frame_rate
                data=[segundo,label]
                self.data.append(data)
                contador   = 0
            else:
                msg        = 'Sujeto no detectado'
            if reproduce:
                cv2.putText(self.frame, msg , (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 4)
                cv2.imshow("frame",self.frame)
            #
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.guardaCSV(csv_path)


    #
    def Extrae_frames(self,label,csv_path,step=10,reproduce=False):
        cap=cv2.VideoCapture(self.path)
        contador=0
        #
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total de frames: {total_frames}")
        while cap.grab():
            contador += 1
            if contador < step:
                continue
            #
            status,self.frame=cap.read()
            if not status:
                break
            #
            self.pose.process(self.frame,True)
            puntos=self.pose.cuenta_puntos(0.1)
            #
            if puntos==33:
                self.data.append(self.pose.extrae_valores(label))
                contador = 0
            if reproduce:
                cv2.putText(self.frame, f"Puntos: {puntos}" , (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 4)
                cv2.imshow("frame",self.frame)
            #
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.guardaCSV(csv_path)
    #
    def guardaCSV(self,csv_path):
        if self.data!=[]:
            columns = self.pose.create_lists() if self.train else ['Segundo', 'Pose']
            df      = pd.DataFrame(self.data, columns=columns)
            df.to_csv(csv_path,index=False, float_format="%.6f")
            print(f'CSV guardado con {len(self.data)} datos')
#
#
class Imagen:
    def __init__(self, path):
        self.pose  = MpPose(image=True)
        self.frame = None
        self.path  = path
    # 
    def Predice(self,model, mapeo_etiquetas,muestra):
        self.frame = cv2.imread(self.path)
        self.pose.process(self.frame,True)
        puntos     = self.pose.cuenta_puntos(0.1)
        if puntos==33:
            X           = (self.pose.extrae_valores())
            x_new       = np.array([X])
            prediccion  = model.predict(x_new)
            prediccion  = (np.argmax(prediccion, axis=1))
            label       = mapeo_etiquetas[str(prediccion[0])]
            msg         = f'Pose: {label}'
        else:
            msg         = f'{puntos} Puntos detectados, se necesitan 33 para hacer una prediccion'
        #
        if muestra:
            cv2.putText(self.frame, msg , (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 4)
            #
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 800, 600) 
            cv2.imshow("frame",self.frame)
        #
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(msg)
        