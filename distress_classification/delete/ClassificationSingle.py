#!/usr/bin/env python
# coding: utf-8

# # CRACK Clssification using GAPS DATASET

# ## IMPORT LIBRARIES

# In[1]:


from __future__ import print_function
#import readdata
from tensorflow import keras
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from sklearn.model_selection import GridSearchCV
#from keras.preprocessing.image import ImageDataGenerator
#import matplotlib.pyplot as plt
import os


# ## LOAD DATA
# # v2 6-classes 256
# In[2]:


from importlib.machinery import SourceFileLoader
MODULENAME = "loadgaps"
MODULEPATH = "/home/ali/my_project/large_files/gaps/loadgaps.py"
lgaps = SourceFileLoader(MODULENAME, MODULEPATH).load_module()

#x_train, y_train, x_valid, y_valid, x_test, y_test = lgaps.loadv2_ZEB256(load = 'low')
x_train, y_train, x_valid, y_valid, x_test, y_test = lgaps.loadv2_NVD256(load = 'low')


# # v1 binary 64
# ## BUILD MODEL
# In[3]:


model_archive = {
                '1' : '2CCP_1FDD',
                '2' : 'DUPLICATE',
                '3' : 'Transfer_learning_vgg',
                '4' : 'Basic CNN'
                    }
print(model_archive)


# In[4]:
import readmodel
model_name = 'model_4' # 'model_1', 'model_2', or model_3
input_shape = x_train[0,:,:,:].shape
num_classes = y_train.shape[1]
model = readmodel.modelchoose(model_name,input_shape, num_classes)
model.summary()


# ## COMPILE MODEL
# In[5]:
# initiate RMSprop optimizer

opt = keras.optimizers.RMSprop(learning_rate=0.01, decay=1e-6)
print(num_classes)

# Let's train the model using RMSprop
if num_classes == 2:
    loss = 'binary_crossentropy'
if num_classes == 6:
    loss = 'categorical_crossentropy'
    
print(loss)

model.compile( loss = loss, # loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


#  ## GENERATE IMAGES
# In[10]:


epochs = 40 # 200
batch_size = 20
#num_classes = y_train_binary.shape[1]



'# ## FIT DATA to MODEL w/o IMG GENRATOR

# In[11]:


history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_valid, y_valid),
              shuffle=True)


# ## MODEL SCORE

# In[ ]:


scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# ## ACCURACY & VAL vs EPOCHS PLOTS

# In[ ]:


# plt.plot(history.history['accuracy'], label='accuracy')
#plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy')
#plt.ylim([0.5, 1])
#plt.legend(loc='lower right')


# ## SAVE MODEL and WEIGHTS

# In[ ]:

import pickle
pickle_out = open("dict.pickle","wb")
pickle.dump(history.history, pickle_out)
pickle_out.close()
print('history is saved')


model_weight_name = model_name + f'_epochs_{epochs}_batchsize_{batch_size}.h5'
print(model_weight_name)

# Save model and weights
#model_name = 'gaps_model_1.h5'
model_weight_name = model_name + f'_epochs_{epochs}_batchsize_{batch_size}.h5'

#os.getcwd()
# save_dir = os.path.join(os.getcwd(), 'saved_models')
save_dir = '/home/ali/my_project/gaps/saved_models'
model_path = os.path.join(save_dir, model_weight_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
# In[ ]:


os.getcwd()

