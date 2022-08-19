import csv
import numpy as np
import pandas as pd
from config import *
import keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed

# run on GPU
tf.keras.backend.clear_session()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("GPUs: ", tf.config.experimental.list_physical_devices('GPU'))
with tf.device("cpu:0"):
    print('-------------------------------------------')
    print("tf.keras code in this scope will run on GPU")
    print('-------------------------------------------')

# import X_train, test and y_train, test
y_train = pd.read_csv('../data/y_train_df.csv', usecols=[1, N_STEPS_FORWARD]) 
y_test = pd.read_csv('../data/y_test_df.csv', usecols=[1, N_STEPS_FORWARD]) 
X_train = pd.read_csv('../data/X_train_df.csv', usecols=[x for x in range(1, N_STEPS_BACK*N_FEATURES+1)]) 
X_test = pd.read_csv('../data/X_test_df.csv', usecols=[x for x in range(1, N_STEPS_BACK*N_FEATURES+1)]) 
# reshape back to original dimension
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] // N_FEATURES, N_FEATURES)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] // N_FEATURES, N_FEATURES)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Model
model = Sequential()
model.add(LSTM(units=96, input_shape=(X_train.shape[1], N_FEATURES)))
model.add(RepeatVector(N_STEPS_FORWARD))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(loss='mse', optimizer='adam')


# training checkpoint
checkpoint_filepath = '../model/train_ckpt/cp.ckpt'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_weights_only=True, monitor="loss", mode='max', save_best_only=True)
# model.load_weights(checkpoint_filepath)
history = model.fit(X_train, y_train, epochs=1, batch_size=128, callbacks=[model_checkpoint_callback], validation_split=0.1)
model.save('../model/B_CRE.h5')

#region
# plot training loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('../result/loss.jpg')
def my_normalize(arr, def_min, def_max):
  normed_array = []
  diff_def = def_max - def_min
  diff = max(arr) - min(arr)
  for i in arr:
    normed_i = ( ((i-min(arr))*diff_def) / diff ) + def_min
    normed_array.append(normed_i)
  return normed_array
normalized_loss = my_normalize(loss, 0, 1)
plt.plot(normalized_loss)
plt.title('normalized model loss')
plt.ylabel('normalized loss')
plt.xlabel('epoch')
plt.savefig('../result/loss-norm.jpg')
#endregion

model = load_model('../model/B_CRE.h5')

prediction = model.predict(X_test)
print(prediction.shape)

