# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:05:25 2019

@author: bmclPublicPC
"""
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import scipy.io as spio
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop 
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import h5py
from keras.models import *

btype = 0 # btype = 0 (SBP), 1 (DBP)
nsig = 1 # nsig = 0 (PPG), 1 (ECG+PPG)


base1 = spio.loadmat(r'mat', squeeze_me=True)

base1_ecg = base1['f_ECG']
base1_ppg = base1['f_PPG']
base_sbp = base1['SBP']
base_dbp = base1['DBP']

if(btype==0):
    base1_sbp = base1['SBP']

if(btype==1):
    base1_sbp = base1['DBP']

if(nsig==0):
    base1_biosig = np.dstack((base1_ppg, base_sbp))[0]

if(nsig==1):
    base1_biosig = np.dstack((base1_ppg, base_sbp, base1_ecg))[0]

def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=32, step=125):

    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
            
        else:
            if i+batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
            
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]-1))
        targets = np.zeros((len(rows),))
        
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            
            if(nsig==0):
                sample = data[indices][:,0].reshape(len(data[indices]), 1)
                samples[j] = sample
                
            if(nsig==1):
                samples[j] = np.dstack((data[indices][:,0], data[indices][:,2]))
                
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
            
train_gen = generator(base1_biosig,
                      lookback=lookback,
                      delay=delay,
                      min_index=train_start,
                      max_index=train_end,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)

val_gen = generator(base1_biosig,
                      lookback=lookback,
                      delay=delay,
                      min_index=val_start,
                      max_index=val_end,
                      step=step,
                      batch_size=batch_size)

test_gen = generator(base1_biosig,
                      lookback=lookback,
                      delay=delay,
                      min_index=test_start,
                      max_index=test_end,
                      step=step,
                      batch_size=batch_size)

val_steps = (val_end - val_start - lookback) // batch_size
test_steps = (test_end - test_start - lookback) // batch_size

model = Sequential()
model.add(layers.Conv1D(20, 200, padding='same', activation='relu', input_shape=(None, base1_biosig.shape[-1]-1)))
model.add(BatchNormalization())
model.add(layers.MaxPool1D(3))
model.add(layers.Conv1D(20, 200, padding='same', activation='relu', input_shape=(None, base1_biosig.shape[-1]-1)))
model.add(BatchNormalization())
model.add(layers.MaxPool1D(3))
model.add(layers.Conv1D(20, 200, padding='same', activation='relu', input_shape=(None, base1_biosig.shape[-1]-1)))
model.add(BatchNormalization())
model.add(layers.MaxPool1D(3))
model.add(layers.Bidirectional(layers.GRU(20, dropout=0.1, recurrent_dropout=0.5)))
model.add(BatchNormalization())
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer=Adam(lr=1e-4), loss='mse')
early_stopping = EarlyStopping(verbose=1, restore_best_weights=True, patience=5)
history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=50,validation_data=val_gen,validation_steps=val_steps,callbacks=[early_stopping])

loss = history.history['loss']
val_loss = history.history['val_loss']

model.save('.h5')

epochs = range(1, len(loss) + 1)

#model.save('keras_testing.h5')

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

print("-- Output -- ")
pred = model.predict_generator(test_gen, steps=test_steps)
print(pred)
ans = base1_biosig[test_start+lookback:test_end,[1]]

pred = pred.astype('uint8')
ans = ans.astype('uint8')

## correlation
pred_mean = np.mean(d_pred)
ans_mean = np.mean(d_ans)

averg = np.ones(len(d_pred)+1)
avergA = averg * pred_mean
avergB = averg * ans_mean
sss = (d_pred-avergA)*(d_ans-avergB)
sssS = sum(sss)
ppp = np.std(d_pred) * np.std(d_ans)
cor = sssS /((len(d_pred)+1) * ppp) 
corr = cor[0]
print('\ncorr : ', corr)

if(btype == 0):
    start = 70
    end = 190
    
if(btype == 1):
    start = 30
    end = 100

x = [a for a in range(end)]
y = [b for b in range(end)]

cor_print = corr
plt.figure(figsize=(6,6), dpi = 100)
plt.scatter(d_pred, d_ans)

plt.xlim(start, end)
plt.ylim(start, end)
plt.title('CC : ' + str(round(corr,3)))

if(btype==0):
    ylabels='reference_SBP'
    xlabels='predicted_SBP'
    
if(btype==1):
    ylabels='reference_DBP'
    xlabels='predicted_DBP'

plt.xlabel(xlabels)
plt.ylabel(ylabels)

plt.plot(x,y, 'k')
plt.show()

rmse = sqrt(mean_squared_error(ans[0:len(pred)], pred))
print('RMSE : {:.2f}\n'.format(rmse))
