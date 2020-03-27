import math
import os
import keras
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model


BATCH_SIZE = 10
SAMPLE_SHAPE = (128, 128, 3)
LOAD_MODEL = False
MODEL_PATH = "deepfake-detection-model.h5"
DATA_DIR = "../input/cropped-faces-1"
TRAIN_DATA_DIR = DATA_DIR + "/train"
VALID_DATA_DIR = DATA_DIR + "/valid"


def getStepsPerEpoch(directory, batch_size):
    return math.floor(sum(
        [len(files) for r, d, files in os.walk(directory)]) / batch_size)


def getModel(input_shape, load_model, model_path):
    if load_model:
        model = load_model(model_path)
    else:
        googleNet_model = InceptionResNetV2(
            include_top=False, weights="imagenet", input_shape=input_shape)
        googleNet_model.trainable = True
        model = Sequential()
        model.add(googleNet_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(units=2, activation="softmax"))
        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Adam(
                lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                amsgrad=False),
            metrics=["accuracy"])
    return model


def main():
    # Load or create model
    model = getModel(SAMPLE_SHAPE, LOAD_MODEL, MODEL_PATH)

    # Load data
    gen = keras.preprocessing.image.ImageDataGenerator()
    train_batches = gen.flow_from_directory(
        TRAIN_DATA_DIR, class_mode="categorical", shuffle=True,
        batch_size=BATCH_SIZE, target_size=SAMPLE_SHAPE[:2])
    valid_batches = gen.flow_from_directory(
        VALID_DATA_DIR, class_mode="categorical", shuffle=True,
        batch_size=BATCH_SIZE, target_size=SAMPLE_SHAPE[:2])

    # Train
    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=2, verbose=0, mode="auto")
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="loss", verbose=1,
        save_best_only=True, mode="min")
    history = model.fit_generator(
        train_batches,
        steps_per_epoch=getStepsPerEpoch(TRAIN_DATA_DIR, BATCH_SIZE),
        epochs=1000,
        validation_data=valid_batches,
        validation_steps=getStepsPerEpoch(VALID_DATA_DIR, BATCH_SIZE),
        verbose=1,
        callbacks=[checkpoint_callback])
    model.save(MODEL_PATH)


main()
