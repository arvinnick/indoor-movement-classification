#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 00:44:05 2019

@author: mohammad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import tensorflow as tf

from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

#making data a one sequence.
path = 'MovementAAL/dataset/MovementAAL_RSS_'
sequences = list()
for i in range(1,315):
    file_path = path + str(i) + '.csv'
    print(file_path)
    df = pd.read_csv(file_path, header=0)
    values = df.values
    sequences.append(values)

targets = pd.read_csv('MovementAAL/dataset/MovementAAL_target.csv')
targets = targets.values[:,1]
#using the rooms' data as separate groups (validation, test and train)
groups = pd.read_csv('MovementAAL/groups/MovementAAL_DatasetGroup.csv', header=0)
groups = groups.values[:,1]

#Padding the sequence with the values in last row to max length
to_pad = 129
new_seq = []
for one_seq in sequences:
    len_one_seq = len(one_seq)
    last_val = one_seq[-1]
    n = to_pad - len_one_seq
   
    to_concat = np.repeat(one_seq[-1], n).reshape(4, n).transpose()
    new_one_seq = np.concatenate([one_seq, to_concat])
    new_seq.append(new_one_seq)
final_seq = np.stack(new_seq)

#truncate the sequence to length 60
seq_len = 60
final_seq=sequence.pad_sequences(final_seq, maxlen=seq_len, padding='post', dtype='float', truncating='post')


train = [final_seq[i] for i in range(len(groups)) if (groups[i]==2)]
validation = [final_seq[i] for i in range(len(groups)) if groups[i]==1]
test = [final_seq[i] for i in range(len(groups)) if groups[i]==3]

train_target = [targets[i] for i in range(len(groups)) if (groups[i]==2)]
validation_target = [targets[i] for i in range(len(groups)) if groups[i]==1]
test_target = [targets[i] for i in range(len(groups)) if groups[i]==3]


train = np.array(train)
validation = np.array(validation)
test = np.array(test)

train_target = np.array(train_target)
train_target = (train_target+1)/2

validation_target = np.array(validation_target)
validation_target = (validation_target+1)/2

test_target = np.array(test_target)
test_target = (test_target+1)/2

#building the model
model = Sequential()
model.add(LSTM(256,
               input_shape=(seq_len,
                            4)))
model.add(Dense(1,
                activation='sigmoid'))

#Lets fit it!
adam = Adam(lr=0.001)
chk = ModelCheckpoint('best_model.pkl',
                      monitor='val_acc',
                      save_best_only=True,
                      mode='max',
                      verbose=1)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
model.fit(train,
          train_target,
          epochs=200,
          batch_size=128,
          callbacks=[chk],
          validation_data=(validation,
          validation_target))

#evaluation
from sklearn.metrics import accuracy_score
test_preds = model.predict_classes(test)
sklearn.metrics.accuracy_score(test_target, test_preds)