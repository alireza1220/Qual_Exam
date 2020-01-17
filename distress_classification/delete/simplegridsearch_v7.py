"""
https://keras.io/examples/cifar10_cnn/
We used standard SGD
batch size = 256,
learning rate = 0.01,
momentum = 0.7.
momentum of 0.9 gives similar results. 
Other learning rates and batch sizes decrease the performance.

The peak performance (reported in the paper) on the validation set was reached after 82 epochs.
Training was stopped at ca 200 epochs when the performance did not improve for a long time. 
"""
from __future__ import print_function
#import tensorflow as tf
#tf.__version__sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from sklearn.model_selection import GridSearchCV

import readdata
import readmodel

# importing data
x_train, y_train_binary, x_valid, y_valid_binary, x_test, y_test_binary, data_name = readdata.gapv164()

# defining constants
data_augmentation = True
model_name = 'modelsimple'

# using grid search
grid = 0 
if grid:
    print ('running grid')
    modelC = readmodel.modelchooseC(model_name)
    epochs = [125, 140]
    param_grid = dict(epochs = epochs)
    grid = GridSearchCV(estimator = modelC, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(x_train, y_train_binary)

    # showing the results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

# using single model
single_model = 1 
if single_model:
    print('running single')
    epochs = 100
    batch_size = 5
    model = readmodel.modelchoose(model_name)
    model.fit(x_train, y_train_binary,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_valid, y_valid_binary),
              shuffle=True)

# Save model and weights
import time
SAVE_DIR = os.path.join(os.getcwd(), 'saved_models')
current_time = time.strftime("_%Y-%m-%d_%H-%M", time.localtime())
cnn_name = data_name+ '_'+ model_name + current_time + '.h5'
    
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)
model_path = os.path.join(SAVE_DIR, cnn_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test_binary, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])



# to work on the grid search more in detais 
# implementing other models
# adding cross validation


