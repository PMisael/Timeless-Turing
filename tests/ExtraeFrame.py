import mediapipe as mp
import cv2
import pandas as pd

class MpPose:
    def __init__(self,video=False):
        self.mp_drawing=mp.solutions.drawing_utils
        self.mp_pose=mp.solutions.pose
        self.video=video
        if self.video:
            self.model=self.mp_pose.Pose(model_complexity=2)
        else:
            self.model=self.mp_pose.Pose(static_image_mode=True,model_complexity=2)
            print("Analizando Imagen")
        self.results=None
    #
    def process(self,frame,draw=False):
        self.results=self.model.process(frame)
        if draw and (self.results.pose_landmarks is not None):
            self.draw_landmarks(frame)
    #
    def draw_landmarks(self,frame):
        self.mp_drawing.draw_landmarks(frame,self.results.pose_landmarks,self.mp_pose.POSE_CONNECTIONS)
    #
    def cuenta_puntos(self,min_visibility):
        contador=0
        if self.results.pose_landmarks is not None:
            for i in range(len(self.results.pose_landmarks.landmark)):
                if self.results.pose_landmarks.landmark[i].visibility>=min_visibility:
                    contador +=1
        return contador
    #
    def extrae_valores(self, label):
        row = [c for lm in self.results.pose_landmarks.landmark
                for c in (lm.x, lm.y, lm.z)]
        if self.video:
            row.append(label)
        return (row)
    #
    def create_lists(self):
        landmarks_names=[lm.name for lm in self.mp_pose.PoseLandmark]
        columns = [f'{n}_{c}' for n in landmarks_names for c in ["x","y","z"]]
        columns.append("label")
        return (columns)
#
#
class Video:
    def __init__(self, path):
        self.path=path
        self.pose  = MpPose()
        self.frame = None
        self.data  = []
    # 
    def process(self,label,csv_path,step=10,reproduce=False):
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
            columns=self.pose.create_lists()
            df=pd.DataFrame(self.data, columns=columns)
            df.to_csv(csv_path,index=False, float_format="%.6f")
            print(f'CSV guardado con {len(self.data)} fotogramas')
#
#
class IMAGEN:
    def __init__(self, path):
        self.pose  = MpPose()
        self.frame = None
        self.path  = path
    # 
    def extrae(self,muestra=False,label="desconocido"):
        self.frame=cv2.imread(self.path)
        self.pose.process(self.frame,True)
        puntos=self.pose.cuenta_puntos(0.1)
        #
        if muestra:
            cv2.putText(self.frame, f"Puntos: {puntos}" , (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 4)
            #
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 800, 600) 
            cv2.imshow("frame",self.frame)
        #
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if puntos==33:
            return(self.pose.extrae_valores(label))
        return(None)