import cv2
import json
import keras
from keras.applications import InceptionResNetV2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, \
    AveragePooling2D, Input, Flatten, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.regularizers import l2
import math
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
import time

# Import 'dlib', if not installed, install offline from wheel file
try:
    import dlib
    print("dlib already installed")
except Exception as e:
    import subprocess
    import sys
    t_start_install = time.time()
    print("Installing dlib...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "../input/dlib-package/dlib-19.19.0"])
    install_time = round(time.time() - t_start_install, 2)
    print("Install ready, time: {}s".format(install_time))
    import dlib


# Data parameters
BATCH_SIZE = 16
EVERY_ITH_FRAME = 30
IMAGE_SIZE = (128, 128)
NUMBER_OF_FACES_PER_VIDEO = 1
NUMBER_OF_CNN_LAYERS = 56

# Training and testing paths
CNN_MODEL_FILE_NAME = "resnet{}_best.h5".format(NUMBER_OF_CNN_LAYERS)
LABELS_PATH = "../input/deepfake-detection-challenge/metadata.json"
LOAD_MODEL_PATH = "../input/resnet-model/deepfake-detection-model.h5"
RAW_TRAIN_DATA_DIRECTORY = "../input/deepfake-detection-challenge/train_sample_videos"
SUBMISSION_CSV = "../input/deepfake-detection-challenge/sample_submission.csv"
TEST_DATA_DIRECTORY = "../input/deepfake-detection-challenge/test_videos"
TRAIN_DATA_DIRECTORY = "../input/cropped-faces-{}".format(NUMBER_OF_FACES_PER_VIDEO)
TRAIN_DIRECTORY = TRAIN_DATA_DIRECTORY + "/train"
VALIDATION_DIRECTORY = TRAIN_DATA_DIRECTORY + "/validation"

# Face detection model paths
FACE_DETECTION_MODEL_FILE = "../input/face-detection-config-files/res10_300x300_ssd_iter_140000_fp16.caffemodel"
FACE_DETECTION_CONFIG_FILE = "../input/face-detection-config-files/deploy.prototxt"


def cropFixedSizedFaceFromCoordinates(detection, i, frame, image_size):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    x1 = detection.left()
    y1 = detection.top()
    x2 = detection.right()
    y2 = detection.bottom()

    # Limit coordinates if they go over frame boundaries
    x1 = min(np.shape(frame)[1], max(0, x1))
    x2 = min(np.shape(frame)[1], max(0, x2))
    y1 = min(np.shape(frame)[0], max(0, y1))
    y2 = min(np.shape(frame)[0], max(0, y2))
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

    return face


def getFaces(
        path, image_size, detector, every_ith_frame=1,
        confidence_threshold=0.5, remove_outliers=True):
    cap = cv2.VideoCapture(path)
    faces = []
    i_frame = 1
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

        # Detect faces from frame
        detections, scores, idx = detector.run(frame, 0)

        # Iterate all detections
        detected_faces_in_frame = 0
        for i, detection in enumerate(detections):

            # Detected face
            if scores[i] > 0.4:
                detected_faces_in_frame += 1
                face = cropFixedSizedFaceFromCoordinates(
                    detection, i, frame, image_size)

                # Add new detected human in a video/frame or add to existing
                # lists of detected humans
                if len(faces) < detected_faces_in_frame:
                    faces.append([face])

                # Only one human in video
                elif len(faces) == 1:
                    faces[0].append(face)

                # Multiple humans in video, check which human face is this
                else:
                    errors = []
                    for human_faces in faces:
                        face_mean = np.mean(np.array(human_faces), axis=0)
                        l1_color_error = np.mean(np.abs(face_mean - face))
                        errors.append(l1_color_error)
                    faces[np.argmin(errors)].append(face)

    # Remove outliers from each human faces
    if remove_outliers:
        faces, outliers = removeOutliers(faces)

    return faces


def removeOutliers(human_faces_in_video, outlier_detection_factor=1.5):
    def removeOutliersFromHumanFaces(human_faces):
        # No faces in list
        if len(human_faces) == 0:
            return [], []

        # Compute similarity scores
        faces_array = np.array(human_faces)
        face_mean = np.mean(faces_array, axis=0).astype(np.uint8)
        similarity_scores = []
        for i, face in enumerate(human_faces):
            l1_color_error = np.mean(np.abs(face_mean - face))
            similarity_scores.append(int(l1_color_error))

        # Define outlier threshold
        similarity_median = np.median(similarity_scores)
        outlier_threshold = outlier_detection_factor*similarity_median
        true_faces = []
        outliers = []

        # Find outliers and true faces
        for i, face in enumerate(human_faces):
            face_similarity_score = similarity_scores[i]
            if face_similarity_score > outlier_threshold:
                outliers.append(face)
            else:
                true_faces.append(face)
        return true_faces, outliers

    true_faces = []
    outliers = []
    for human_faces in human_faces_in_video:
        max_faces_per_human = max(
            [len(human_faces) for human_faces in human_faces_in_video])

        # Remove outliers from humans that were detected often
        # (rarely detected humans can be outliers)
        if len(human_faces) > max_faces_per_human / 2:
            human_faces, outliers_in_human_faces = \
                removeOutliersFromHumanFaces(human_faces)
            true_faces.append(human_faces)
            outliers.append(outliers_in_human_faces)
        else:
            outliers.append(human_faces)

    return true_faces, outliers


def test(print_time=True):
    # Load model
    print("Loading model")
    t_start_program = time.time()
    detector = dlib.get_frontal_face_detector()
    model = tf.keras.models.load_model(LOAD_MODEL_PATH)
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
            humans_in_video = getFaces(
                TEST_DATA_DIRECTORY + '/' + file_name, IMAGE_SIZE, detector,
                EVERY_ITH_FRAME)
            faces_loading_time = round(time.time() - t_start_video, 2)

            # Loop each human's faces
            predictions = []
            for human_faces in humans_in_video:

                # Predict score for each stack of face
                t_start_predicting = time.time()

                # Create samples by taking random faces
                number_of_batches = math.ceil(len(human_faces) / BATCH_SIZE)
                human_faces = np.array(human_faces)
                for b in range(number_of_batches):
                    start_sample_index = b * BATCH_SIZE
                    end_sample_index = min(
                        (b+1) * BATCH_SIZE, human_faces.shape[0])
                    batch = human_faces[start_sample_index:end_sample_index]
                    predictions += list(model.predict(batch))
                t_video_processed = time.time()
                prediction_time = round(
                    t_video_processed - t_start_predicting, 2)
                total_spent_time = int(
                    (t_video_processed - t_start_program) / 60)

            # Print processing times
            if print_time:
                number_of_faces = sum(
                    [len(human_faces) for human_faces in humans_in_video])
                print((
                    "Video: {:5}/{}, {:20}. Number of faces: {:5}. " +
                    "Cropping faces time: {:7}s. Prediction time: {:6}s. " +
                    "Total time: {:4}min").format(
                        i+1, number_of_videos, file_name, number_of_faces,
                        faces_loading_time, prediction_time, total_spent_time))
            else:
                print("Video: {}/{}".format(i+1, number_of_videos))

            # Compute final score for video and write to CSV
            video_score = 0.5
            if len(predictions) > 0:
                final_scores = []
                for prediction in predictions:
                    fake_score = prediction[0]
                    real_score = prediction[1]
                    if fake_score > real_score:
                        final_scores.append(1 - fake_score)
                    else:
                        final_scores.append(real_score)
                video_score = np.mean(np.array(final_scores))

        # Handle unexpected exceptions, give score 0.5
        except Exception as e:
            print("Exception thrown, video: {}/{}, {}".format(
                i+1, number_of_videos, file_name))
            video_score = 0.5

        # Write video file name and score
        submission_file.at[i, "filename"] = file_name
        submission_file.at[i, "label"] = video_score

    # Write CSV to file
    submission_file.to_csv("submission.csv", index=False)
    print("Submission file written")


if __name__ == "__main__":
    test()
