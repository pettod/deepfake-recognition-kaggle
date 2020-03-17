import cv2
import json
import os
import random
import time

# Import from project file
from main import \
    getFaces, \
    loadLabels, \
    FACE_DETECTION_CONFIG_FILE, \
    FACE_DETECTION_MODEL_FILE, \
    IMAGE_SIZE, \
    LABELS_PATH, \
    RAW_TRAIN_DATA_DIRECTORY, \
    TRAIN_DIRECTORY


EVERY_ITH_FRAME = 10
NUMBER_OF_FACES_PER_VIDEO = 5
TARGET_PATH_FAKE = TRAIN_DIRECTORY + "/fake"
TARGET_PATH_REAL = TRAIN_DIRECTORY + "/real"


def createTrainData(print_time=True):
    t_start_program = time.time()
    labels = loadLabels()
    net = cv2.dnn.readNetFromCaffe(
        FACE_DETECTION_CONFIG_FILE, FACE_DETECTION_MODEL_FILE)

    # Iterate training videos
    number_of_videos = len(os.listdir(RAW_TRAIN_DATA_DIRECTORY))
    for i, file_name in enumerate(sorted(os.listdir(
            RAW_TRAIN_DATA_DIRECTORY))):

        # Crop faces from train video
        t_start_video = time.time()
        faces_in_video = getFaces(
            RAW_TRAIN_DATA_DIRECTORY + '/' + file_name, IMAGE_SIZE, net,
            EVERY_ITH_FRAME)

        # Don't take this video if not enough detected faces
        if NUMBER_OF_FACES_PER_VIDEO > len(faces_in_video):
            print((
                "No saved faces from video: {}. " +
                "Number of detected faces: {}").format(
                    file_name, len(faces_in_video)))
            continue

        # Pick random faces
        random_face_indices = sorted(random.sample(
            range(len(faces_in_video)), NUMBER_OF_FACES_PER_VIDEO))
        picked_faces = [
            face for j, face in enumerate(faces_in_video)
            if j in random_face_indices]
        t_faces_loaded = time.time()
        faces_loading_time = round(t_faces_loaded - t_start_video, 2)
        total_spent_time = int((t_faces_loaded - t_start_program) / 60)

        # Save faces
        path = TARGET_PATH_FAKE
        if labels[i]:
            path = TARGET_PATH_REAL
        for j, face in enumerate(picked_faces):
            sample_file_name = "{}_{}.png".format(str(i+1), str(j+1))
            cv2.imwrite(path + '/' + sample_file_name, face)

        # Print processing times
        if print_time:
            print((
                "Video: {:5}/{}, {:20}. Number of faces: {:5}. " +
                "Cropping faces time: {:7}s. Total time: {:4}min").format(
                    i+1, number_of_videos, file_name, len(faces_in_video),
                    faces_loading_time, total_spent_time))
        else:
            print("Video: {}/{}".format(i+1, number_of_videos))


if __name__ == "__main__":
    createTrainData()
