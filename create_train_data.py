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
    RAW_TRAIN_DATA_DIRECTORY, \
    TRAIN_DIRECTORY


TARGET_PATH_FAKE = TRAIN_DIRECTORY + "/fake"
TARGET_PATH_REAL = TRAIN_DIRECTORY + "/real"


def createDataPaths(fake_path, real_path):
    # Create paths if not existing
    if not os.path.isdir(fake_path):
        os.makedirs(fake_path)
    if not os.path.isdir(real_path):
        os.makedirs(real_path)


def isVideoProcessedPreviously(sample_file_name):
    name_and_ending = sample_file_name.split('.')
    video_0_name = "{}_0.{}".format(name_and_ending[0], name_and_ending[1])
    return os.path.exists(TARGET_PATH_FAKE + '/' + video_0_name) or \
        os.path.exists(TARGET_PATH_REAL + '/' + video_0_name)


def createTrainData(print_time=True):
    t_start_program = time.time()
    labels = loadLabels()
    net = cv2.dnn.readNetFromCaffe(
        FACE_DETECTION_CONFIG_FILE, FACE_DETECTION_MODEL_FILE)
    createDataPaths(TARGET_PATH_FAKE, TARGET_PATH_REAL)

    # Iterate training videos
    number_of_videos = len(os.listdir(RAW_TRAIN_DATA_DIRECTORY))
    for i, file_name in enumerate(sorted(os.listdir(
            RAW_TRAIN_DATA_DIRECTORY))):
        sample_file_name = "{}.png".format(file_name.split('.')[0])

        # Skip video face cropping if it is done in previous session
        if isVideoProcessedPreviously(sample_file_name):
            print((
                "Skipping video '{}'. It is processed in the previous " +
                "session.").format(file_name))
            continue

        # Crop faces from train video
        t_start_video = time.time()
        humans_in_video = getFaces(
            RAW_TRAIN_DATA_DIRECTORY + '/' + file_name, IMAGE_SIZE, net,
            EVERY_ITH_FRAME)

        for j, human_faces in enumerate(humans_in_video):

            # Don't take this video/human_faces if not enough detected faces
            if NUMBER_OF_FACES_PER_VIDEO > len(human_faces):
                print((
                    "No saved faces from video: {}. " +
                    "Number of detected faces: {}").format(
                        file_name, len(human_faces)))
                continue

            # Pick random faces
            random_face_indices = sorted(random.sample(
                range(len(human_faces)), NUMBER_OF_FACES_PER_VIDEO))
            picked_faces = [
                face for k, face in enumerate(human_faces)
                if k in random_face_indices]

            # Save faces
            path = TARGET_PATH_FAKE
            if labels[i]:
                path = TARGET_PATH_REAL

            # Horizontally concatenate faces
            sample_image = cv2.hconcat(picked_faces)
            sample_file_name_with_index = "{}_{}.{}".format(
                sample_file_name.split('.')[0], j,
                sample_file_name.split('.')[1])
            cv2.imwrite(
                "{}/{}".format(path, sample_file_name_with_index),
                cv2.hconcat(picked_faces))

        # Print processing times
        if print_time:
            t_faces_loaded = time.time()
            faces_loading_time = round(t_faces_loaded - t_start_video, 2)
            total_spent_time = int((t_faces_loaded - t_start_program) / 60)
            number_of_faces = sum(
                [len(human_faces) for human_faces in humans_in_video])
            print((
                "Video: {:5}/{}, {:20}. Number of faces: {:5}. " +
                "Cropping faces time: {:7}s. Total time: {:4}min").format(
                    i+1, number_of_videos, file_name, number_of_faces,
                    faces_loading_time, total_spent_time))
        else:
            print("Video: {}/{}".format(i+1, number_of_videos))


if __name__ == "__main__":
    createTrainData()
