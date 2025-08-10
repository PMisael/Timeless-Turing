import mediapipe as mp

class MpPose:
    def __init__(self,image=False, Video_Train=False):
        self.mp_drawing=mp.solutions.drawing_utils
        self.mp_pose=mp.solutions.pose
        self.train=Video_Train
        if image != True:
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
    def extrae_valores(self, label='desconocido'):
        row = [c for lm in self.results.pose_landmarks.landmark
                for c in (lm.x, lm.y, lm.z)]
        if self.train:
            row.append(label)
        return (row)
    #
    def create_lists(self):
        landmarks_names=[lm.name for lm in self.mp_pose.PoseLandmark]
        columns = [f'{n}_{c}' for n in landmarks_names for c in ["x","y","z"]]
        columns.append("label")
        return (columns)
#