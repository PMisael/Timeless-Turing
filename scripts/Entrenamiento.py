import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

import tensorflow as tf
from keras import layers,callbacks
from keras.models import Sequential, load_model
from keras.optimizers import Adam

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import json #Guardar mapeo de vectores One Hot

tf.config.list_physical_devices('GPU')


class Modelo:
    def __init__(self):
        self.df_train_val = None
        self.df_test      = None
        self.X_train      = None
        self.Y_train      = None
        self.X_val        = None
        self.Y_val        = None
        self.X_test       = None
        self.Y_test       = None
        self.model        = None
        self.poses_lb     = LabelBinarizer()
    #
    def cargar_datos(self):
        self.df_train_val   =pd.read_csv("data/csv/train_val_csv/dataset_completo.csv")
        self.df_test  =pd.read_csv("data/csv/test_csv/dataset_completo.csv")
    #
    def dividir_datos(self):
        # Entrenamiento 80% + Validacion 20%
        X_train_val         = self.df_train_val.drop(columns=['label'])
        Y_train_val         = self.df_train_val['label']
        Y_train_val_one_hot = self.poses_lb.fit_transform(Y_train_val)
        #
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
        X_train_val, Y_train_val_one_hot, test_size=0.20, stratify=Y_train_val, random_state=42)
        #
        # Test no necesita dividirse
        self.X_test         = self.df_test.drop(columns=['label'])
        Y_test              = self.df_test['label']
        self.Y_test         = self.poses_lb.fit_transform(Y_test)
        #
        # Guardar mapeo de etiquetas : onehot para predicciones
        mapeo={}
        for i,clase in enumerate(self.poses_lb.classes_):
            mapeo[i]=clase
        with open('models/mapeo_etiquetas.json', 'w') as f:
            json.dump(mapeo, f)
    #
    def construccion(self):
        n_features = self.X_train.shape[1]
        n_classes  = self.Y_train.shape[1]
        #
        self.model = Sequential([
            layers.Input( shape=(n_features,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.30),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.30),
            layers.Dense(n_classes, activation='softmax')
        ])
        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    #
    def entrenamiento(self):
        model_path = Path ("models").resolve()
        models = [p for p in model_path.rglob("best_model_*.keras")]
        model_name = str(model_path / f'best_model_{len(models)}.keras')


        checkpoint = callbacks.ModelCheckpoint(
            model_name, save_best_only=True, monitor='val_loss')

        history = self.model.fit(
            self.X_train, self.Y_train,
            epochs=100,
            batch_size=256,
            validation_data=(self.X_val, self.Y_val),
            callbacks=[checkpoint],
            verbose=2)
        return history
    #
    def evaluacion(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        print(f"Accuracy en test: {test_acc:.3f}")
        #
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.Y_test.argmax(1), y_pred.argmax(1))
        print(cm)
        print(classification_report(self.Y_test.argmax(1), y_pred.argmax(1), target_names=self.poses_lb.classes_))
    #
    def graficar_entrenamiento(self,history):
        fig1, axes = plt.subplots(1,2,figsize=(12,6))
        axes[0].plot(history.history['accuracy'])
        axes[0].plot(history.history['val_accuracy'])
        axes[0].set_title('Precision del modelo')
        axes[0].set_ylabel('Precision')
        axes[0].set_xlabel('Epoch')
        axes[0].legend(['Train', 'Test'], loc='upper left')

        #Funcion de perdida
        axes[1].plot(history.history['loss'])
        axes[1].plot(history.history['val_loss'])
        axes[1].set_title('Funcion de perdida del modelo')
        axes[1].set_ylabel('Funcion de perdida')
        axes[1].set_xlabel('Epoch')
        axes[1].legend(['Train', 'Val'], loc='upper left')

        plt.tight_layout()
        plt.show()
    #
    def getX_test_Y_test_poses_lb(self): #Util para evaluar modelos (Notebook)
        return (self.X_test,self.Y_test, self.poses_lb)
#
#
def main():
    model=Modelo()
    model.cargar_datos()
    model.dividir_datos()
    model.construccion()
    history=model.entrenamiento()
    model.evaluacion()
    model.graficar_entrenamiento(history)

#
if __name__ == "__main__":
    main()