from __future__ import absolute_import, division, print_function, unicode_literals

#import tensorflow as tf

import os

import time
import numpy as np
import matplotlib.pyplot as plt

import readdata
import readmodel
#import dataaug

from sklearn.model_selection import GridSearchCV

SAVE_DIR = os.path.join(os.getcwd(), 'saved_models')


batch_size = [100] # [256]
epochs = [125, 140]
model_name = 'modelsimple'

x_train, y_train_binary, x_valid, y_valid_binary, x_test, y_test_binary, data_name = readdata.gapv164()
model = readmodel.modelchoose(model_name)


param_grid = dict(epochs = epochs)
grid = GridSearchCV(estimator = model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(x_train, y_train_binary)



print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
    
    
"""
model.fit(x_train, y_train_binary,
          batch_size=256,
          epochs=125,
          validation_data=(x_valid, y_valid_binary),
          shuffle=True)
"""

# save 
current_time = time.strftime("_%Y-%m-%d_%H-%M", time.localtime())
#cnn_name = data_name+ '_'+ model_name + current_time + '.h5'
