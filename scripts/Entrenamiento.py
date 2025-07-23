import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

import tensorflow as tf
from keras import layers,callbacks
from keras.models import Sequential, load_model
from keras.optimizers import Adam

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

tf.config.list_physical_devices('GPU')
print(tf.test.is_built_with_cuda(), tf.test.is_gpu_available())

DATA      = Path("./data").resolve()
TRAIN_VAL = DATA / "train_val_csv"
TEST      = DATA / "test_csv"
Salida    = Path("dataset_completo.csv")
#
# Unir los csvs de entrenamiento para hacer uno solo
def une_csvs(carpeta):
    csvs = [p for p in DATA.rglob("*.csv") if p.parent==carpeta and p.stem!=Salida.stem]
    df =pd.concat((pd.read_csv(f) for f in csvs),ignore_index=True)
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(carpeta/Salida, index=False, float_format="%.6f")

    x= df.drop(columns=['label'])
    y= df['label']
    y_one_hot =poses_lb.fit_transform(y)
    return df,x,y,y_one_hot
#
poses_lb   =LabelBinarizer()
#
df_train_val,X_train_val,Y_train_val,Y_oh_train_val=une_csvs(TRAIN_VAL)
#
df_test, X_test, Y_test, Y_oh_test=une_csvs(TEST)

print(Y_oh_train_val,Y_train_val)

print(Y_oh_test,Y_test)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, Y_oh_train_val, test_size=0.20, stratify=Y_train_val, random_state=42)

n_features = X_train.shape[1]
n_classes  = y_train.shape[1]
#print(n_features,n_classes)
#
model = Sequential([
    layers.Input( shape=(n_features,)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.30),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.30),
    layers.Dense(n_classes, activation='softmax')
])
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model_path = Path ("./models").resolve()
models = [p for p in model_path.rglob("best_model*.keras")]
model_name = str(model_path / f'best_model_{len(models)}.keras')


checkpoint = callbacks.ModelCheckpoint(
    model_name, save_best_only=True, monitor='val_loss')

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=256,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint],
    verbose=2)


#Evaluar resultados
best_model = load_model(model_name)
test_loss, test_acc = best_model.evaluate(X_test, Y_oh_test, verbose=0)
print(f"Accuracy en test: {test_acc:.3f}")

y_pred = best_model.predict(X_test)
cm = confusion_matrix(Y_oh_test.argmax(1), y_pred.argmax(1))
print(cm)
print(classification_report(Y_oh_test.argmax(1), y_pred.argmax(1), target_names=poses_lb.classes_))

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
axes[1].legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.show()