from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

def make_baseline_model():
    # Create model
    model = Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=(96,96,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model


def make_finetuned_resnet50():
    #from keras.applications.resnet50 import preprocess_input
    from keras.applications.resnet50 import ResNet50
    from keras import layers, optimizers

    rn50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    model = Sequential()
    model.add(rn50)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation = "softmax"))

    # Unfreeze starting from the 5th convolution layer, block16

    rn50.Trainable=True

    set_trainable=False
    for layer in rn50.layers:
        if layer.name == 'res5c_branch2c':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    return model 


def make_finetuned_inceptv3():
    
    from keras.applications.inception_v3 import InceptionV3
    from keras import layers, optimizers

    inceptv3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    model = Sequential()
    model.add(inceptv3)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation = "softmax"))

    # Unfreeze starting from the last convolution layer

    inceptv3.Trainable=True

    set_trainable=False
    for layer in inceptv3.layers:
        if layer.name == 'conv2d_94':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    return model 



def make_finetuned_xcept():

    from keras.applications.xception import Xception
    from keras import layers, optimizers

    xcept = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    model = Sequential()
    model.add(xcept)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation = "softmax"))

    # Unfreeze starting from the last convolution layer

    xcept.Trainable=True
    set_trainable=False
    
    for layer in xcept.layers:
        if layer.name == 'block14_sepconv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    
    return model
