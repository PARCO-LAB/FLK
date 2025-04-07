from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import sys
import os

input_path = "dataset/training_FLK/aist"
timestep = 64
name = "gru_aist_64"
_size = "10"

X_train = np.load(os.path.join(input_path,"X_train_"+str(timestep)+"_" +  _size +  ".npy"))
y_train = np.load(os.path.join(input_path,"y_train_"+str(timestep)+"_" +  _size +  ".npy"))
print(X_train.shape)
X_val = np.load(os.path.join(input_path,"X_val_"+str(timestep)+"_" +  _size +  ".npy"))
y_val = np.load(os.path.join(input_path,"y_val_"+str(timestep)+"_" +  _size +  ".npy"))

filepath_out="models/" + name
checkpoint = ModelCheckpoint(filepath_out, monitor='val_mae', verbose=1, save_best_only=True, mode = 'min')
callbacks_list = [checkpoint]

model = keras.Sequential(
    [
    layers.GRU(1024, return_sequences=True ,input_shape=(timestep,12*3)),
    layers.Dropout(0.1),
	layers.GRU(1024, return_sequences=False),
    layers.Dropout(0.1),
    layers.Dense(12*3)
    ]
)

model.summary()
opt = keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=opt, loss="mse", metrics=["mae"])

history = model.fit(X_train, y_train, 
                    epochs=200, 
                    batch_size=128, validation_data=(X_val,y_val) , callbacks=callbacks_list
                    )

model.save(filepath_out)