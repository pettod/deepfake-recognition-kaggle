import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
import math
import numpy as np
import os
import tensorflow as tf


BATCH_SIZE = 16
DATA_DIRECTORY = "../cropped_faces/resnet_data"
EPOCHS = 1000
IMAGE_SIZE = (224, 224)
LEARNING_RATE = 0.0001
TRAIN_DIRECTORY = DATA_DIRECTORY + "/train"
VALIDATION_DIRECTORY = DATA_DIRECTORY + "/train"


def getNumberOfSteps(data_directory, batch_size):
    return math.floor(sum(
        [len(files) for r, d, files in os.walk(data_directory)]) / batch_size)


def getBatchGenerator(
    data_directory, image_size, batch_size, horizontal_flip=False,
        vertical_flip=False):
    gen = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=horizontal_flip, vertical_flip=vertical_flip)
    batches = gen.flow_from_directory(
        data_directory, target_size=image_size, class_mode="categorical",
        shuffle=True, batch_size=batch_size)
    return batches


def main():
    with tf.device("/device:GPU:0"):

        # Load batch generators
        train_batches = getBatchGenerator(
            TRAIN_DIRECTORY, IMAGE_SIZE, BATCH_SIZE)
        validation_batches = getBatchGenerator(
            VALIDATION_DIRECTORY, IMAGE_SIZE, BATCH_SIZE, True, True)

        # Create model
        model = keras.applications.resnet50.ResNet50(weights=None, classes=2)
        model.compile(
            optimizer=Adam(lr=LEARNING_RATE), loss="binary_crossentropy",
            metrics=["accuracy"])

        # Fit data
        early_stopping = EarlyStopping(patience=10)
        checkpointer = ModelCheckpoint(
            "resnet50_best.h5", verbose=1, save_best_only=True)
        model.fit_generator(
            train_batches,
            steps_per_epoch=getNumberOfSteps(TRAIN_DIRECTORY, BATCH_SIZE),
            epochs=EPOCHS,
            callbacks=[early_stopping, checkpointer],
            validation_data=validation_batches,
            validation_steps=getNumberOfSteps(
                VALIDATION_DIRECTORY, BATCH_SIZE))
        model.save("resnet50_final.h5")


main()
