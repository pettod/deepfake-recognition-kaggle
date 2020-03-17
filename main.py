import cv2
import json
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
from keras.optimizers import Adam
import math
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
import time


# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 1000
EVERY_ITH_FRAME = 10
IMAGE_SIZE = (224, 224)
LEARNING_RATE = 0.0001

# Training and testing paths
CREATE_MODEL_PATH = "resnet50_best.h5"
LOAD_MODEL_PATH = "../input/resnet-model/resnet50_best.h5"
RAW_TRAIN_DATA_DIRECTORY = "../input/deepfake-detection-challenge/train_sample_videos"
LABELS_PATH = "../input/deepfake-detection-challenge/metadata.json"
SUBMISSION_CSV = "../input/deepfake-detection-challenge/sample_submission.csv"
TEST_DATA_DIRECTORY = "../input/deepfake-detection-challenge/test_videos"
TRAIN_DATA_DIRECTORY = "../input/cropped-faces"
TRAIN_DIRECTORY = TRAIN_DATA_DIRECTORY + "/train"
VALIDATION_DIRECTORY = TRAIN_DATA_DIRECTORY + "/train"

# Face detection model paths
FACE_DETECTION_MODEL_FILE = "../input/face-detection-config-files/res10_300x300_ssd_iter_140000_fp16.caffemodel"
FACE_DETECTION_CONFIG_FILE = "../input/face-detection-config-files/deploy.prototxt"


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


def getFaces(
        path, image_size, net, every_ith_frame=1, confidence_threshold=0.5,
        only_one_face_per_frame=False):
    cap = cv2.VideoCapture(path)
    faces = []
    i_frame = 1
    first_face_coordinates = []
    if (not cap.isOpened()):
        print("Cannot open video:", path)
        return faces
    while(cap.isOpened()):

        # Read video frame
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames and read only every i'th frame
        if i_frame != every_ith_frame:
            i_frame += 1
            continue
        else:
            i_frame = 1
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Detect faces from frame
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()

        # Iterate all detections
        multiple_faces_per_frame = []
        face_coordinates = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Detected face
            if confidence > confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                face_width = x2 - x1
                face_height = y2 - y1

                # Make detected area square
                if face_height < face_width:
                    coordinate_change = (face_width - face_height) // 2
                    x1 += coordinate_change
                    x2 -= coordinate_change
                else:
                    coordinate_change = (face_height - face_width) // 2
                    y1 += coordinate_change
                    y2 -= coordinate_change
                face = cv2.resize(frame[y1:y2, x1:x2], image_size)
                if not only_one_face_per_frame:
                    faces.append(face)
                else:
                    multiple_faces_per_frame.append(face)
                    face_coordinates.append([x1, y1, x2, y2])

        # Add only one face from frame, and should be the same face
        if only_one_face_per_frame and len(multiple_faces_per_frame) > 0:

            # Add first face from first frame (from which the face was
            # detected, for example first time detected from frame 5)
            if len(faces) == 0:
                x1 = face_coordinates[0][0]
                y1 = face_coordinates[0][1]
                x2 = face_coordinates[0][2]
                y2 = face_coordinates[0][3]
                faces.append(multiple_faces_per_frame[0])
                first_face_coordinates = [x1, y1, x2, y2]

            # If frame has multiple detected faces, add the closest
            # corresponding to the face detected from first frame
            elif len(multiple_faces_per_frame) > 1:
                closest_face_index = 0
                closest_distance = 4000*4000*4
                x1_a = first_face_coordinates[0]
                y1_a = first_face_coordinates[1]
                x2_a = first_face_coordinates[2]
                y2_a = first_face_coordinates[3]
                for i in range(len(multiple_faces_per_frame)):
                    x1_b = face_coordinates[i][0]
                    y1_b = face_coordinates[i][1]
                    x2_b = face_coordinates[i][2]
                    y2_b = face_coordinates[i][3]
                    face_distance = (
                        (x1_a - x1_b) ** 2 +
                        (y1_a - y1_b) ** 2 +
                        (x2_a - x2_b) ** 2 +
                        (y2_a - y2_b) ** 2)
                    if face_distance < closest_distance:
                        closest_distance = face_distance
                        closest_face_index = i
                faces.append(multiple_faces_per_frame[closest_face_index])

            # Add only one detected face, could make errors if video has for
            # example 2 faces, but sometimes only 1 face is detected
            else:
                faces.append(multiple_faces_per_frame[0])

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


def loadLabels():
    # Load labels from metadata
    labels = []
    with open(LABELS_PATH) as labels_json:
        labels_dict = json.load(labels_json)
        for key, value in labels_dict.items():
            if value["label"] == "FAKE":
                labels.append(0)
            else:
                labels.append(1)
    return labels


def rotatePoint(p, c, rad_angle):
    p[0] -= c[0]
    p[1] -= c[1]
    p = [int(p[0] * math.cos(rad_angle) - p[1] * math.sin(rad_angle) + c[0]),
         int(p[0] * math.sin(rad_angle) + p[1] * math.cos(rad_angle) + c[1])]
    return p


def test(print_time=True):
    # Load model
    print("Loading model")
    t_start_program = time.time()
    net = cv2.dnn.readNetFromCaffe(
        FACE_DETECTION_CONFIG_FILE, FACE_DETECTION_MODEL_FILE)
    model = load_model(LOAD_MODEL_PATH)
    model_loading_time = round(time.time() - t_start_program, 2)
    print("Model loading time: {}s".format(model_loading_time))

    # Load CSV file
    submission_file = pd.read_csv(SUBMISSION_CSV)
    submission_file.label = submission_file.label.astype(float)

    # Loop test videos
    print("Cropping faces from videos")
    number_of_videos = len(os.listdir(TEST_DATA_DIRECTORY))
    for i, file_name in enumerate(sorted(os.listdir(TEST_DATA_DIRECTORY))):
        try:

            # Crop faces from test video
            t_start_video = time.time()
            faces_in_video = getFaces(
                TEST_DATA_DIRECTORY + '/' + file_name, IMAGE_SIZE, net,
                EVERY_ITH_FRAME)
            faces_loading_time = round(time.time() - t_start_video, 2)

            # Predict score for each face
            predictions = []
            t_start_predicting = time.time()
            for face in faces_in_video:
                face = np.expand_dims(face, axis=0)
                predictions.append(model.predict(face)[0])
            t_video_processed = time.time()
            prediction_time = round(t_video_processed - t_start_predicting, 2)
            total_spent_time = int((t_video_processed - t_start_program) / 60)

            # Print processing times
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
            video_score = 0.5
            if len(predictions) > 0:
                predictions = np.array(predictions)
                fake_scores = predictions[:, 0]
                real_scores = predictions[:, 1]
                video_score = np.mean((real_scores - fake_scores + 1) / 2)
        except Exception as e:
            video_score = 0.5

        # Write video file name and score
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
            CREATE_MODEL_PATH, verbose=1, save_best_only=True)
        model.fit_generator(
            train_batches,
            steps_per_epoch=getNumberOfSteps(TRAIN_DIRECTORY, BATCH_SIZE),
            epochs=EPOCHS,
            callbacks=[early_stopping, checkpointer],
            validation_data=validation_batches,
            validation_steps=getNumberOfSteps(
                VALIDATION_DIRECTORY, BATCH_SIZE))
        model.save("resnet50_final.h5")


if __name__ == "__main__":
    #train()
    test()
