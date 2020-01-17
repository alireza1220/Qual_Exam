# model_1: https://keras.io/examples/cifar10_cnn/
from tensorflow.keras import layers, models


def modelchoose(model_name,input_shape, num_classes ):
    if model_name == 'model_1':
        model = model_1(input_shape, num_classes)
    elif model_name == 'model_2':
        model = model_2(input_shape, num_classes)
    elif model_name == 'model_3':
        model = model_3(input_shape, num_classes)
    elif model_name == 'model_4':
        model = model_4(input_shape, num_classes)
    elif model_name == 'model_5':
        model = model_5(input_shape, num_classes)
    elif model_name == 'model_6':
        model = VGG_16(input_shape, num_classes)
        
        
    else:
        print('please call a valid model name')
    return model 

def model_1(input_shape, num_classes):
    model_name = 'model_1_2CCP_1FDD'
    model = models.Sequential()
    print('last change')
    # CONV CONV POOL
#    model.add(layers.Conv2D(32, (3, 3), padding='same',
#                            input_shape=input_shape , data_format='channels_first'))
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    
    # CONV CONV POOL
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    
    # FLAT DENS DENS
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation('softmax'))
    
    print('X current model is: ' + model_name)
    return model

#def model_2(input_shape, num_classes):
def model_2():
    input_shape = (64, 64, 3)
    num_classes = 2
    model_name1 = 'DUPLICATE OF MODEL_1 to test'
    model = models.Sequential()
    
    # CONV CONV POOL
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                            input_shape=input_shape , data_format='channels_first'))
    #model.add(layers.Conv2D(32, (3, 3), padding='same',
    #                 input_shape=x_train.shape[1:]))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    
    # CONV CONV POOL
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    
    # FLAT DENS DENS
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation('softmax'))
    
    print('X current model is: ' + model_name1)
    return model


def model_3(input_shape, num_classes):
    model_name1 = 'Transfer_learning_vgg'
    # add preprocessing layer to the front of VGG
    from tensorflow.keras.applications.vgg16 import VGG16
    # re-size all the images to this
    # input_shape = (64, 64, 3) # feel free to change depending on dataset
    vgg = VGG16(input_shape= input_shape , weights='imagenet', include_top=False)
    for layer in vgg.layers:
        layer.trainable = False
    x = layers.Flatten()(vgg.output)
    # x = Dense(1000, activation='relu')(x)
    prediction = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=vgg.input, outputs=prediction)
    print(model_name1)
    return model

def model_4(input_shape, num_classes):
    #create model
    model = models.Sequential()
    #add model layers
    model.add(layers.Conv2D(64, kernel_size=3, input_shape=input_shape, data_format='channels_first'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, kernel_size=3))
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation('softmax'))
    return model

def model_5(input_shape, num_classes):
    print('add first channel')
    #create model
    model = models.Sequential()
    
    # CONV CONV POOL
    model.add(layers.Conv2D(32, kernel_size=3, input_shape=input_shape, data_format='channels_first'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))    
    
    # CONV CONV POOL
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    
    
    # FLAT DENS DENS
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation('softmax'))
    
    return model


def VGG_16(input_shape, num_classes):
    print('vgg16 is on process')
    model = models.Sequential()
#    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
#    model.add(layers.ZeroPadding2D((1,1), , input_shape=input_shape, data_format='channels_first'))
    model.add(layers.Convolution2D(64, 3, 3, padding='same', activation='relu',input_shape=input_shape, data_format='channels_first' ))
    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Convolution2D(64, 3, 3, activation='relu'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Convolution2D(128, 3, 3, activation='relu'))
    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Convolution2D(128, 3, 3, activation='relu'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

    #model.add(layers.ZeroPadding2D((1,1)))
    #model.add(layers.Convolution2D(256, 3, 3, activation='relu'))
    #model.add(layers.ZeroPadding2D((1,1)))
    #model.add(layers.Convolution2D(256, 3, 3, activation='relu'))
    #model.add(layers.ZeroPadding2D((1,1)))
    #model.add(layers.Convolution2D(256, 3, 3, activation='relu'))
    #model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

    #model.add(layers.ZeroPadding2D((1,1)))
    #model.add(layers.Convolution2D(512, 3, 3, activation='relu'))
    #model.add(layers.ZeroPadding2D((1,1)))
    #model.add(layers.Convolution2D(512, 3, 3, activation='relu'))
    #model.add(layers.ZeroPadding2D((1,1)))
    #model.add(layers.Convolution2D(512, 3, 3, activation='relu'))
    #model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Convolution2D(512, 3, 3, activation='relu'))
    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Convolution2D(512, 3, 3, activation='relu'))
    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Convolution2D(512, 3, 3, activation='relu'))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1000, activation='softmax'))

    model.add(layers.Dense(num_classes))
    model.add(layers.Activation('softmax'))
#    if weights_path:
#        model.load_weights(weights_path)

    return model


if __name__ == '__main__':
    model_n = 'model_1'
    model = modelchoose(model_n)

    
    
    
"""
from tensorflow.keras import layers, models
num_classes = 2
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding='same',
                        input_shape=(1, 64, 64), data_format='channels_first'))
#model.add(layers.Conv2D(32, (3, 3), padding='same',
#                 input_shape=x_train.shape[1:]))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes))
model.add(layers.Activation('softmax'))
# def modelchooseC(model_name):
#     def create_model():
#        model = modelchoose(model_name)
#        return model
#    
#    from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#    modelC = KerasClassifier(build_fn=create_model, verbose=1)
#    #modelC = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
#    return modelC
"""