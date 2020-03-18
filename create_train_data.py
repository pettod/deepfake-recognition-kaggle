import cv2
import json
import os
import random
import time

# Import from project file
from main import \
    getFaces, \
    loadLabels, \
    EVERY_ITH_FRAME, \
    FACE_DETECTION_CONFIG_FILE, \
    FACE_DETECTION_MODEL_FILE, \
    IMAGE_SIZE, \
    LABELS_PATH, \
    NUMBER_OF_FACES_PER_VIDEO, \
    ONLY_ONE_FACE_PER_FRAME, \
    RAW_TRAIN_DATA_DIRECTORY, \
    TRAIN_DIRECTORY


TARGET_PATH_FAKE = TRAIN_DIRECTORY + "/fake"
TARGET_PATH_REAL = TRAIN_DIRECTORY + "/real"


def createTrainData(print_time=True):
    t_start_program = time.time()
    labels = loadLabels()
    net = cv2.dnn.readNetFromCaffe(
        FACE_DETECTION_CONFIG_FILE, FACE_DETECTION_MODEL_FILE)

    # Create paths if not existing
    if not os.path.isdir(TARGET_PATH_FAKE):
        os.makedirs(TARGET_PATH_FAKE)
    if not os.path.isdir(TARGET_PATH_REAL):
        os.makedirs(TARGET_PATH_REAL)

    # Iterate training videos
    number_of_videos = len(os.listdir(RAW_TRAIN_DATA_DIRECTORY))
    for i, file_name in enumerate(sorted(os.listdir(
            RAW_TRAIN_DATA_DIRECTORY))):

        # Crop faces from train video
        t_start_video = time.time()
        faces_in_video = getFaces(
            RAW_TRAIN_DATA_DIRECTORY + '/' + file_name, IMAGE_SIZE, net,
            EVERY_ITH_FRAME, only_one_face_per_frame=ONLY_ONE_FACE_PER_FRAME)

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

        # Horizontally concatenate faces
        sample_image = cv2.hconcat(picked_faces)
        sample_file_name = "{}.png".format(str(i+1))
        cv2.imwrite(path + '/' + sample_file_name, cv2.hconcat(picked_faces))

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
