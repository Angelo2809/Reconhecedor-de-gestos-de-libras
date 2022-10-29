from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU

class Convolucao(object):
    
   
    @staticmethod
    def constructor(width, height, channels, classes):
        # kernel_size = (3,3), padding = 'same', input_shape = (height, width, channels)
        model = Sequential()
        model.add(Conv2D(filters = 32))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2,2)))
        
        model.add(Conv2D(filters = 32, kernel_size = (3,3)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2,2)))
        
        model.add(Conv2D(filters = 64, kernel_size = (3,3)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2,2)))
        
        model.add(Flatten())
        model.add(Dense(64, activation = 'relu'))
        model.add(Dense(32, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation = 'softmax')) 
        
        return model # INPUT => CONV => POOL => CONV => POOL => CONV => POOL => FC => FC => OUTPUT