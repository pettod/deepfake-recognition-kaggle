import cv2
import face_recognition
import imutils
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
from keras.optimizers import Adam
import math
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time


BATCH_SIZE = 16
EPOCHS = 1000
EVERY_ITH_FRAME = 10
IMAGE_SIZE = (224, 224)
LEARNING_RATE = 0.0001
MODEL_PATH = "../input/resnet-model/resnet50_best.h5"
SUBMISSION_CSV = "../input/deepfake-detection-challenge/sample_submission.csv"
TEST_DATA_DIRECTORY = "../input/deepfake-detection-challenge/test_videos"
TRAIN_DATA_DIRECTORY = "../input/cropped-faces"
TRAIN_DIRECTORY = TRAIN_DATA_DIRECTORY + "/train"
VALIDATION_DIRECTORY = TRAIN_DATA_DIRECTORY + "/train"


def rotatePoint(p, c, rad_angle):
    p[0] -= c[0]
    p[1] -= c[1]
    p = [int(p[0] * math.cos(rad_angle) - p[1] * math.sin(rad_angle) + c[0]),
         int(p[0] * math.sin(rad_angle) + p[1] * math.cos(rad_angle) + c[1])]
    return p


def cropAndAlign(
        img, location, landmarks, left_eye_loc_x=0.3, left_eye_loc_y=0.3):
    # Find the gravity center of the eye points
    left_eye = [
        sum([point[0] for point in landmarks["left_eye"]]) //
        len(landmarks["left_eye"]),
        sum([point[1] for point in landmarks["left_eye"]]) //
        len(landmarks["left_eye"])]
    right_eye = [
        sum([point[0] for point in landmarks["right_eye"]]) //
        len(landmarks["left_eye"]),
        sum([point[1] for point in landmarks["right_eye"]]) //
        len(landmarks["left_eye"])]
    y = right_eye[1] - left_eye[1]
    x = right_eye[0] - left_eye[0]
    angle = math.atan2(y, x)
    deg_angle = 180*math.atan2(y, x)/math.pi

    img = imutils.rotate(img, deg_angle, tuple(left_eye))
    location = list(location)
    location[3], location[0] = rotatePoint(
        [location[3], location[0]], left_eye, angle)
    location[1], location[2] = rotatePoint(
        [location[1], location[2]], left_eye, angle)

    w = abs(location[1] - location[3])
    h = abs(location[0] - location[2])
    size = max(w, h)

    left_eye[0] -= location[3]
    left_eye[1] -= location[0]

    if(w > h):
        img = imutils.translate(
            img, size * left_eye_loc_x - left_eye[0],
            size * left_eye_loc_y - left_eye[1] - (w-h)//2)
        img = img[
            location[0] - (w-h)//2:location[0] -
            (w-h)//2+w, location[3]:location[3]+w]
    else:
        img = imutils.translate(
            img, size * left_eye_loc_x - left_eye[0] - (h-w)//2,
            size * left_eye_loc_y - left_eye[1])
        img = img[
            location[0]:location[0]+h, location[3] -
            (h-w)//2: location[3] - (h-w)//2+h]
    return img


def getFaces(path, image_size, every_ith_frame=1):
    height = image_size[0]
    width = image_size[1]
    cap = cv2.VideoCapture(path)
    faces = []
    if (not cap.isOpened()):
        print(path)
        print("VIDEO НЕ ОТКРЫВАЕТСЯ СУКА БЛЯТЬ")
        return []
    i_frame = 1
    while(cap.isOpened()):
        ret, img = cap.read()
        if i_frame == every_ith_frame:
            i_frame = 1
        else:
            i_frame += 1
            continue
        if not ret:
            break
        box_faces = face_recognition.face_locations(img, model="hog")
        landmarks = face_recognition.face_landmarks(img, box_faces)
        for j in range(len(box_faces)):
            box = box_faces[j]
            img_face = cropAndAlign(img, box, landmarks[j])
            if(np.min(np.shape(img_face)) != 0):
                faces.append(cv2.resize(img_face, (height, width)))
    return faces


def getBatchGenerator(
    data_directory, image_size, batch_size, horizontal_flip=False,
        vertical_flip=False):
    gen = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=horizontal_flip, vertical_flip=vertical_flip)
    batches = gen.flow_from_directory(
        data_directory, target_size=image_size, class_mode="categorical",
        shuffle=True, batch_size=batch_size)
    return batches


def getNumberOfSteps(data_directory, batch_size):
    return math.floor(sum(
        [len(files) for r, d, files in os.walk(data_directory)]) / batch_size)


def test(print_time=True):
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
            TEST_DATA_DIRECTORY + '/' + file_name, IMAGE_SIZE, EVERY_ITH_FRAME)
        faces_loading_time = round(time.time() - t_start_video, 2)

        # Predict score for each face
        predictions = []
        t_start_predicting = time.time()
        for face in faces_in_video:
            face = np.expand_dims(face, axis=0)
            predictions.append(model.predict(face)[0][1])
        t_video_processed = time.time()
        prediction_time = round(t_video_processed - t_start_predicting, 2)
        total_spent_time = int((t_video_processed - t_start_program) / 60)
        if print_time:
            print((
                "Video: {:5}/{}, {:20}. Number of faces: {:5}. " +
                "Cropping faces time: {:7}s. Prediction time: {:6}s. " +
                "Total time: {:4}min").format(
                    i+1, number_of_videos, file_name, len(faces_in_video),
                    faces_loading_time, prediction_time, total_spent_time))
        else:
            print("Video: {}/{}".format(i+1, number_of_videos))

        # Compute final score for video and write to CSV
        video_score = np.mean(np.array(predictions))
        submission_file.at[i, "filename"] = file_name
        submission_file.at[i, "label"] = video_score

    # Write CSV to file
    submission_file.to_csv("submission.csv", index=False)
    print("Submission file written")


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


def main():
    #train()
    test()


main()
