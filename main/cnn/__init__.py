from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU
from keras.regularizers import l2

class Convolucao(object):
    
   
    @staticmethod
    def constructor(width, height, channels, classes):
        model = Sequential()
        model.add(Conv2D(filters = 64, kernel_size = (3,3), activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(MaxPooling2D((2,2)))

        model.add(Conv2D(filters = 64, kernel_size = (3,3), activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(MaxPooling2D((2,2)))

        model.add(Conv2D(filters = 32, kernel_size = (3,3), activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(MaxPooling2D((2,2)))
        
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(64, activation = 'relu'))
        model.add(Dense(classes, activation = 'softmax')) 

#         model = keras.Sequential(
#     [
#         keras.Input(shape=(HEIGHT, WIDTH, 3)),
#         keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
#         keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
#         keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
#         keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         keras.layers.Flatten(),
#         keras.layers.Dropout(0.5),
#         keras.layers.Dense(64, activation="relu"),
#         keras.layers.Dense(len(labels), activation="softmax"),
#     ]
# )
        
        return model # INPUT => CONV => POOL => CONV => POOL => CONV => POOL => FC => FC => OUTPUT