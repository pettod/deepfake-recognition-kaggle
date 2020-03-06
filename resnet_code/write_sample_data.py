import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random


NUMBER_OF_RANDOM_FACES = 3
LABEL_FILE_NAME = "metadata.json"
SAMPLE_SIZE = (224, 224)
TARGET_PATH_FAKE = "../cropped_faces/resnet_data/train/fake/"
TARGET_PATH_REAL = "../cropped_faces/resnet_data/train/real/"
TRAIN_DATA_PATH = "../cropped_faces/deepfake/faces_train_videos/"
SAVE_TYPE = 0  # 0: ".png", 1: ".npy"
START_VIDEO_INDEX = 0


def cropSquareImage(image):
    # Create square image
    height, width, _ = image.shape
    side_length = np.minimum(height, width)
    center_x = width // 2
    center_y = height // 2
    start_y = center_y - side_length // 2
    end_y = center_y + side_length // 2
    start_x = center_x - side_length // 2
    end_x = center_x + side_length // 2
    return image[start_y:end_y, start_x:end_x]


def loadLabels():
    # Load labels from metadata
    labels = []
    with open(LABEL_FILE_NAME) as labels_json:
        labels_dict = json.load(labels_json)
        for key, value in labels_dict.items():
            if value["label"] == "FAKE":
                labels.append(0)
            else:
                labels.append(1)
    return labels


def pickRandomFrames(video_frames):
    random_frame_indices = sorted(random.sample(
        range(len(video_frames)), NUMBER_OF_RANDOM_FACES))
    random_frames = []
    for i in random_frame_indices:
        random_frames.append(video_frames[i])
    random_frames = np.swapaxes(np.swapaxes(
        np.array(random_frames), 0, 2), 0, 1)
    return random_frames


def readFramesFromVideo(video_frames_path, transform_to_gray=True):
    video_frames = []
    frame_names = sorted(os.listdir(video_frames_path))

    # Loop frame paths
    for frame_name in frame_names:
        frame_path = video_frames_path + frame_name
        image = plt.imread(frame_path)

        # Crop image to square and resize to fixed size
        image = cv2.resize(
            cropSquareImage(image),
            dsize=SAMPLE_SIZE,
            interpolation=cv2.INTER_CUBIC)

        # Transform image to gray scale
        if transform_to_gray:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image[image < 0] = 0
            image[image > 1] = 1
        video_frames.append(image)
    return video_frames


def saveArray(
        image_array, label, sample_number, save_functions, save_file_types):
    # Save image array
    if label:
        save_functions[SAVE_TYPE](
            TARGET_PATH_REAL + str(sample_number) + save_file_types[SAVE_TYPE],
            image_array)
    else:
        save_functions[SAVE_TYPE](
            TARGET_PATH_FAKE + str(sample_number) + save_file_types[SAVE_TYPE],
            image_array)


def main():
    labels = loadLabels()
    save_functions = [plt.imsave, np.save]
    save_file_types = [".png", ".npy"]

    # Read video folder
    video_paths = sorted(os.listdir(TRAIN_DATA_PATH))
    number_of_videos = len(video_paths)
    for i in range(START_VIDEO_INDEX, number_of_videos):
        print("Video {}/{}".format(i+1, number_of_videos), end="\r")
        video_path = video_paths[i]
        video_frames_path = TRAIN_DATA_PATH + video_path + "/PNG/"

        # Read frames inside video folder
        video_frames = readFramesFromVideo(video_frames_path)

        # Save frames
        if len(video_frames) >= NUMBER_OF_RANDOM_FACES:
            random_frames = pickRandomFrames(video_frames)
            saveArray(
                random_frames, labels[i], i+1, save_functions, save_file_types)

    print()


main()
