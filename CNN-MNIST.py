"""Implementing lenet 5 architecture on mnist dataset"""


# Importing various keras
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

from keras.datasets import mnist

img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# data preprocessing
x_train = x_train.reshape(x_train.shape[0],img_rows, img_cols,1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols,1)
input_shape = (1, img_rows, img_cols)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
# creating the neural net implement lenet 5 architecture
classifier = Sequential()
classifier.add(Convolution2D(6,(5,5),padding='same',activation='relu',input_shape=(28,28,1)))
classifier.add(MaxPooling2D((2,2),strides=2))
classifier.add(Convolution2D(16,(5,5), activation='relu'))
classifier.add(MaxPooling2D((2,2),strides=2))
classifier.add(Flatten())
classifier.add(Dense(units=84,activation='relu'))
classifier.add(Dense(units=10,activation='softmax'))
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# fitting the training set into fhe neural net
classifier.fit(x_train,y_train,batch_size=128,epochs = 25)
# checking the accuracy of the model
score = classifier.evaluate(x_test, y_test)
print(score)