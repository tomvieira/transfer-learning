# -*- coding: utf-8 -*-

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications import ResNet50V2 #adicionei isso
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D #adicionei isso
# re-size all the images to this
IMAGE_SIZE = [400, 400]

train_path = 'dataset/train'
valid_path = 'dataset/test'

# add preprocessing layer to the front of resnet
resnet = ResNet50V2(input_shape=(400,400,3), weights='imagenet', include_top=False)

# don't train existing weights
for layer in resnet.layers:
  layer.trainable = False



  # useful for getting number of classes
folders = glob(train_path + '/*')


# our layers - you can add more if you want
x = Flatten()(resnet.output)
#x = Dense(64, activation='relu')(x) # descomentei e troquei de 1000 para 64
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)

#adicionei
#model.add(Conv2D(64, (3,3), activation='relu'))
#model.add(MaxPooling2D(2,2))
#model.add(Flatten())
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.4))
#model.add(Dense(3, activation='softmax'))
#ate aqui



# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam', #mudei maiuscula a
  metrics=['accuracy']
)


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   #shear_range = 0.2, #comentei
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (400, 400),
                                                 batch_size = 32,
                                                 color_mode='rgb', #adicionei isso
                                                 #class_mode="binary",#adicionei isso
                                                 shuffle=True,#adicionei isso
                                                 seed=42,#adicionei isso
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(valid_path,
                                            target_size = (400, 400) ,
                                            batch_size = 32,
                                            color_mode='rgb', #adicionei isso
                                                 #class_mode="binary",#adicionei isso
                                            shuffle=True,#adicionei isso
                                            seed=42,#adicionei isso
                                            class_mode = 'categorical')

'''r=model.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 2000)'''

# fit the model
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

import tensorflow as tf

from keras.models import load_model

model.save('hist_model_resnet.keras')
