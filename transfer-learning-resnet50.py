# -*- coding: utf-8 -*-

from keras.models import load_model
import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications import ResNet50V2  # adicionei isso
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# adicionei isso
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
# re-size all the images to this
IMAGE_SIZE = [400, 400]

train_path = 'ebhi-split-2categorias/train'
valid_path = 'ebhi-split-2categorias/val'
test_path = 'ebhi-split-2categorias/test'

# add preprocessing layer to the front of resnet
resnet = ResNet50V2(input_shape=(400, 400, 3),
                    weights='imagenet', include_top=False)

# don't train existing weights
for layer in resnet.layers:
    layer.trainable = False

    # useful for getting number of classes
folders = glob(train_path + '/*')


# our layers - you can add more if you want
x = Flatten()(resnet.output)
# x = Dense(64, activation='relu')(x) # descomentei e troquei de 1000 para 64
#prediction = Dense(len(folders), activation='softmax')(x)
prediction = Dense(1, activation='sigmoid')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)

# adicionei
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(MaxPooling2D(2,2))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(2, activation='softmax'))
# ate aqui


# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range = 0.2, #comentei
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=(400, 400),
                                                 batch_size=32,
                                                 color_mode='rgb',  # adicionei isso
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(valid_path,
                                            target_size=(400, 400),
                                            batch_size=32,
                                            color_mode='rgb',  # adicionei isso
                                            class_mode='binary')
valid_set = valid_datagen.flow_from_directory(valid_path,
                                              target_size=(400, 400),
                                              batch_size=32,
                                              class_mode='binary')


# fit the model
r = model.fit(
    training_set,
    validation_data=valid_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(valid_set)
)
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()

plt.savefig('LossVal_loss_resnet')
plt.show()
# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()

plt.savefig('AccVal_acc_resnet')
plt.show()

model.save('hist_model_resnet.h5')
