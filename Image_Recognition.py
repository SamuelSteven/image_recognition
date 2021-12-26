import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import random
import os
from shutil import copytree, rmtree
from shutil import copy
from collections import defaultdict
import collections
import matplotlib.image as img
import numpy as np
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras import models
from tensorflow import keras

image_list = ['banana', 'book', 'turtle']
images = []


def train_model():
    K.clear_session()
    n_classes = 3
    img_width, img_height = 299, 299
    train_data_dir = 'train_image'
    validation_data_dir = 'validation_image'
    nb_train_samples = 30  # 30
    nb_validation_samples = 30  # 3
    batch_size = 8

    train_datagen = ImageDataGenerator(
        rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(
        img_height, img_width), batch_size=batch_size, class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(
        img_height, img_width), batch_size=batch_size, class_mode='categorical')

    inception = InceptionV3(weights='imagenet', include_top=False)
    x = inception.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)

    predictions = Dense(3, kernel_regularizer=regularizers.l2(
        0.005), activation='softmax')(x)

    model = Model(inputs=inception.input, outputs=predictions)
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(
        filepath='best_model_5class.hdf5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('history_5class.log')

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=nb_train_samples // batch_size,
                                  validation_data=validation_generator,
                                  validation_steps=nb_validation_samples // batch_size,
                                  epochs=30, verbose=1,
                                  callbacks=[csv_logger, checkpointer])
    model.save('model_trained_5class.hdf5')


def predict_class(model, images, show=True):
    for img in images:
        img = image.load_img(img, target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.

        pred = model.predict(img)
        index = np.argmax(pred)
        image_list.sort()
        pred_value = image_list[index]
        if show:
            plt.imshow(img[0])
            plt.axis('off')
            plt.title(pred_value)
            plt.show()


def predict_image():
    images.append('test_image/banana/banana_test.jpg')
    images.append('test_image/book/book_test.jpg')
    images.append('test_image/turtle/turtle_test.jpg')
    predict_class(model_best, images, True)


# train_model()
K.clear_session()
model_best = load_model(
    'model/best_model_5class.hdf5', compile=False)
predict_image()
