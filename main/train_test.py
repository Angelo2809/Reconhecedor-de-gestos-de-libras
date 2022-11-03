import os
import datetime as dt
import time
import tensorflow as tf
from keras.regularizers import l2


#---------------------------------------------------------------
#DEFS
def getDateStr():
        return str('{date:%Y%m%d_%H%M}').format(date=dt.datetime.now())

def getTimeMin(start, end):
        return (end - start)/60

#---------------------------------------------------------------
# Variables

EPOCHS = 30
CLASS = 21
FILE_NAME = 'model_angelo_'
train_dir = '../dataset/training'
test_dir = '../dataset/test'
img_x, img_y = 64,64
img_size = (img_x, img_y)
img_shape = img_size + (3,)
img_channel = 3
batch_size = 32
epochs = 30
learning_rate = 0.01
classes = 21
#---------------------------------------------------------------

# Main

print('[INFO] [INICIO]: ' + getDateStr() + '\n')

print('[INFO] Download dataset')

start = time.time()
training_set = tf.keras.preprocessing.image_dataset_from_directory( directory= train_dir,batch_size=32,image_size=(64, 64), shuffle= True)
test_set = tf.keras.preprocessing.image_dataset_from_directory( directory= test_dir,batch_size=32,image_size=(64, 64), shuffle= True)

print("[INFO] Treinando a CNN...")
model = tf.keras.models.Sequential([
    
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape = img_shape),
    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation="relu", kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(classes, activation = 'softmax'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')

])

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate), metrics = ['accuracy'], loss = tf.keras.losses.BinaryCrossentropy())

model.summary()

history = model.fit(training_set, validation_data = test_set, epochs = epochs)

EPOCHS = len(history.history["loss"])

print("[INFO] Salvando modelo treinado ...")

file_date = getDateStr()
model.save('../models/'+FILE_NAME+file_date+'.h5')
print('[INFO] modelo: ../models/'+FILE_NAME+file_date+'.h5 salvo!')

end = time.time()

print("[INFO] Tempo de execução da CNN: %.1f min" %(getTimeMin(start,end)))


print('[INFO] Summary: ')
model.summary()

print("\n[INFO] Avaliando a CNN...")
score = model.evaluate_generator(generator=test_set, steps=(test_set.n // test_set.batch_size), verbose=1)
print('[INFO] Accuracy: %.2f%%' % (score[1]*100), '| Loss: %.5f' % (score[0]))

print("[INFO] Sumarizando loss e accuracy para os datasets 'train' e 'test'")

print('\n[INFO] [FIM]: ' + getDateStr())
print('\n\n')
