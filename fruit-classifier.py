from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

classifier = Sequential()
classifier.add(Convolution2D(filters=64,kernel_size=(3,3),strides = (1,1),activation='relu',input_shape=(100,100,3)))
classifier.add(MaxPooling2D(pool_size=(3,3),strides=2))
classifier.add(Convolution2D(filters=128,kernel_size=(3,3),strides = (1,1),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(3,3),strides=2))
classifier.add(Convolution2D(filters=256,kernel_size=(3,3),strides = (1,1),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(3,3),strides=2))
classifier.add(Convolution2D(filters=512,kernel_size=(3,3),strides = (1,1),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
classifier.add(Convolution2D(filters=1024,kernel_size=(3,3),strides = (1,1),activation='relu'))
classifier.add(Flatten())
classifier.add(Dense(units=512,activation='relu'))
classifier.add(Dense(units=114,activation='softmax'))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'fruits-360/Training',
        target_size=(100,100),
        batch_size=128,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'fruits-360/Test',
        target_size=(100,100),
        batch_size=128,
        class_mode='categorical')

classifier.fit_generator(
        train_generator,
        steps_per_epoch=57276,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=1000)