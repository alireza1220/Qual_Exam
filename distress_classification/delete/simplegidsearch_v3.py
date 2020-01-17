from __future__ import absolute_import, division, print_function, unicode_literals

#import tensorflow as tf

import keras
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import readdata
#import readmodel
#import dataaug

from sklearn.model_selection import GridSearchCV


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator


SAVE_DIR = os.path.join(os.getcwd(), 'saved_models')


batch_size = [256]
epochs = [125]
learning_rate = [0.01]
loss = ['categorical_crossentropy']
metrics = [['accuracy']]
data_augmentation = False

x_train, y_train_binary, x_valid, y_valid_binary, x_test, y_test_binary, data_name = readdata.gapv164()


def modelsimple(n):
    output_classes = 2
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(1,64,64), data_format='channels_first'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(n, n)))
#    model.add(Conv2D(64,(3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    
    # Fully connected layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_classes))
    
    model.add(Activation('softmax'))
        
    return model


#param_grid = dict(neurons0=neurons0, neurons1=neurons1, neurons2=neurons2, neurons3=neurons3, lay_num = lay_num , dropout1 = dropout1, dropout2= dropout2)
#grid = GridSearchCV( estimator=modelR, param_grid=param_grid, n_jobs=-1)
#grid_result = grid.fit(arr_x_train, arr_y_train)

#model, model_name = readmodel.modelsimple()

# initiate RMSprop optimizer


#opt = keras.optimizers.RMSprop(learning_rate=learning_rate[0] , decay=1e-6)

# Let's train the model using RMSprop
#model.compile(loss=loss[0],
#              optimizer=opt,
#              metrics=metrics[0])
n = [2, 3]

param_grid = dict(epochs=[10,20,30])
grid = GridSearchCV(estimator=modelC, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x_train, y_train_binary)

param_grid = dict(epochs)#(batch_size = batch_size, epochs = epochs) #, learning_rate = learning_rate, loss = loss, metrics = metrics)
grid = GridSearchCV(estimator = modelsimple(), param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(x_train, y_train_binary)



print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



# save 
current_time = time.strftime("_%Y-%m-%d_%H-%M", time.localtime())
#cnn_name = data_name+ '_'+ model_name + current_time + '.h5'
