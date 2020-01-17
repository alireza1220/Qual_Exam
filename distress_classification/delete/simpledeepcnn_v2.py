from __future__ import absolute_import, division, print_function, unicode_literals

#import tensorflow as tf

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import readdata
import readmodel
#import dataaug

SAVE_DIR = os.path.join(os.getcwd(), 'saved_models')

#IMG_HEIGHT = 64
#IMG_WIDTH = 15
batch_size = [256]
epochs = [125]
learning_rate = [0.01]
loss = ['categorical_crossentropy']
metrics = [['accuracy']]
data_augmentation = False

x_train, y_train_binary, x_valid, y_valid_binary, x_test, y_test_binary, data_name = readdata.gapv164()
model, model_name = readmodel.modelsimple()

model.fit(x_train, y_train_binary,
          batch_size=batch_size[0],
          epochs=epochs[0],
          validation_data=(x_valid, y_valid_binary),
          shuffle=True)

# save 
current_time = time.strftime("_%Y-%m-%d_%H-%M", time.localtime())
cnn_name = data_name+ '_'+ model_name + current_time + '.h5'
