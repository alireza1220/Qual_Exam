"""
https://keras.io/examples/cifar10_cnn/
We used standard SGD
with batch size = 256,
learning rate = 0.01,
and momentum = 0.7.
 A momentum of 0.9 gives similar results. 
Other learning rates and batch sizes decrease the performance.

The peak performance (reported in the paper) on the validation set was reached after 82 epochs.
 Training was stopped at ca 200 epochs when the performance did not improve for a long time. 
"""
from __future__ import print_function
import os
from importlib.machinery import SourceFileLoader
MODULENAME = "laodgaps"
MODULEPATH = "/home/ali/p/ml/data_img/gaps/loadgaps.py"
lgaps = SourceFileLoader(MODULENAME, MODULEPATH).load_module()


import readdata
x_train, y_train_binary, x_valid, y_valid_binary, x_test, y_test_binary, data_name = readdata.gapv164()

batch_size = 32
num_classes = 2
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'gaps_v1.h5'


import readmodel
model = readmodel.modelchoose('modelsimple')

model.fit(x_train, y_train_binary,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_valid, y_valid_binary),
          shuffle=True)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test_binary, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
