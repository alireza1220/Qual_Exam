


# model_1: https://keras.io/examples/cifar10_cnn/
from tensorflow.keras import layers, models



# model_1: https://keras.io/examples/cifar10_cnn/
from tensorflow.keras import layers, models
model_name1 = 'Transfer_learning_vgg'
# add preprocessing layer to the front of VGG
from tensorflow.keras.applications.vgg16 import VGG16
# re-size all the images to this
input_shape = (64, 64, 3) # feel free to change depending on dataset
vgg = VGG16(input_shape= input_shape , weights='imagenet', include_top=False)
# don't train existing weights
for layer in vgg.layers:
        layer.trainable = False

x = layers.Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = layers.Dense(num_classes, activation='softmax')(x)

# create a model object
model = models.Model(inputs=vgg.input, outputs=prediction)
