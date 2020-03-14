import cv2
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
from keras.optimizers import Adam
import math
from mtcnn import MTCNN
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time


BATCH_SIZE = 16
EPOCHS = 1000
IMAGE_SIZE = (224, 224)
LEARNING_RATE = 0.0001
MODEL_PATH = "resnet50_best.h5"
SUBMISSION_CSV = "sample_submission.csv"
TEST_DATA_DIRECTORY = "../input/deepfake-detection-challenge/test_videos"
TRAIN_DATA_DIRECTORY = "../cropped_faces/resnet_data"
TRAIN_DIRECTORY = TRAIN_DATA_DIRECTORY + "/train"
VALIDATION_DIRECTORY = TRAIN_DATA_DIRECTORY + "/train"


def getFaces(path, image_size):
    height = image_size[0]
    width = image_size[1]
    m = MTCNN()
    cap = cv2.VideoCapture(path)
    faces = []
    if (not cap.isOpened()):
        print(path)
        print("VIDEO НЕ ОТКРЫВАЕТСЯ СУКА БЛЯТЬ")
        return []
    while(cap.isOpened()):
        ret, img = cap.read()
        if not ret:
            break
        box_faces = m.detect_faces(img)
        for j in range(len(box_faces)):
            box = box_faces[j]["box"]
            w = max(box[2], box[3])
            if(box[2] > box[3]):
                img_face = img[box[1]-(box[2] - box[3])//2:box[1] -
                               (box[2] - box[3])//2+w, box[0]:box[0]+w]
            else:
                img_face = img[box[1]:box[1]+w, box[0] -
                               (-box[2] + box[3])//2:box[0]-(-box[2] + box[3])//2+w]
            if(np.min(np.shape(img_face)) != 0):
                faces.append(cv2.resize(img_face, (height, width)))
    return faces


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


def train():
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
            MODEL_PATH, verbose=1, save_best_only=True)
        model.fit_generator(
            train_batches,
            steps_per_epoch=getNumberOfSteps(TRAIN_DIRECTORY, BATCH_SIZE),
            epochs=EPOCHS,
            callbacks=[early_stopping, checkpointer],
            validation_data=validation_batches,
            validation_steps=getNumberOfSteps(
                VALIDATION_DIRECTORY, BATCH_SIZE))
        model.save("resnet50_final.h5")


def test():
    # Load model
    print("Loading model")
    t_start_program = time.time()
    model = load_model(MODEL_PATH)
    model_loading_time = round(time.time() - t_start_program, 2)
    print("Model loading time: {}s".format(model_loading_time))

    # Load CSV file
    submission_file = pd.read_csv(SUBMISSION_CSV)
    submission_file.label = submission_file.label.astype(float)

    # Loop test videos
    print("Cropping faces from videos")
    number_of_videos = len(os.listdir(TEST_DATA_DIRECTORY))
    for i, file_name in enumerate(sorted(os.listdir(TEST_DATA_DIRECTORY))):

        # Crop faces from test video
        t_start_video = time.time()
        faces_in_video = getFaces(
            TEST_DATA_DIRECTORY + '/' + file_name, IMAGE_SIZE)
        faces_loading_time = round(time.time() - t_start_video, 2)

        # Predict score for each face
        predictions = []
        t0 = time.time()
        for face in faces_in_video:
            face = np.expand_dims(face, axis=0)
            predictions.append(model.predict(face)[0][1])
        t_video_processed = time.time()
        prediction_time = round(t_video_processed - t_start_video, 2)
        total_spent_time = int((t_video_processed - t_start_program) / 60)
        print((
            "Video: {:5}/{}, {:20}. Number of faces: {:5}. " +
            "Cropping faces time: {:7}s. Prediction time: {:6}s. " +
            "Total time: {:4}min").format(
                i+1, number_of_videos, file_name, len(faces_in_video),
                faces_loading_time, prediction_time, total_spent_time))

        # Compute final score for video and write to CSV
        video_score = np.mean(np.array(predictions))
        submission_file.at[i, "filename"] = file_name
        submission_file.at[i, "label"] = video_score

    # Write CSV to file
    submission_file.to_csv("submission.csv", index=False)
    print("Submission file written")


def main():
    #train()
    test()


main()
