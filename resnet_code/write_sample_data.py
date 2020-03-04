import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random


NUMBER_OF_RANDOM_FACES = 5
LABEL_FILE_NAME = "metadata.json"
SAMPLE_SIZE = 224
TARGET_PATH_FAKE = "../cropped_faces/resnet_data/train/fake/"
TARGET_PATH_REAL = "../cropped_faces/resnet_data/train/real/"
TRAIN_DATA_PATH = "../cropped_faces/deepfake/faces_train_videos/"
START_VIDEO_INDEX = 0


def cropSquareImage(image):
    # Create square image
    height, width, _ = image.shape
    side_lenth = np.minimum(height, width)
    center_x = width // 2
    center_y = height // 2
    start_y = center_y - side_lenth // 2
    end_y = center_y + side_lenth // 2
    start_x = center_x - side_lenth // 2
    end_x = center_x + side_lenth // 2
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


def rgb2gray(rgb):
    # Convert RGB image to gray scale
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def main():
    labels = loadLabels()

    # Read video folder
    video_paths = sorted(os.listdir(TRAIN_DATA_PATH))
    number_of_videos = len(video_paths)
    for i in range(START_VIDEO_INDEX, number_of_videos):
        print("Video {}/{}".format(i+1, number_of_videos), end="\r")
        video_path = video_paths[i]
        video_frames_path = TRAIN_DATA_PATH + video_path + "/PNG/"
        video_frames = []

        # Read frames inside video folder
        frame_names = sorted(os.listdir(video_frames_path))
        for frame_name in frame_names:
            frame_path = video_frames_path + frame_name
            image = plt.imread(frame_path)
            square_image = np.resize(
                cropSquareImage(image), (SAMPLE_SIZE, SAMPLE_SIZE, 3))
            gray_image = rgb2gray(square_image)
            video_frames.append(gray_image)

        # Save frames
        if len(video_frames) >= NUMBER_OF_RANDOM_FACES:

            # Take random frames
            random_frame_indices = sorted(random.sample(
                range(len(video_frames)), NUMBER_OF_RANDOM_FACES))
            random_frames = []
            for j in random_frame_indices:
                random_frames.append(video_frames[j])
            random_frames = np.array(random_frames)

            # Write numpy arrays to file
            if labels[i]:
                np.save(TARGET_PATH_REAL + str(i+1) + ".npy", random_frames)
            else:
                np.save(TARGET_PATH_FAKE + str(i+1) + ".npy", random_frames)


main()
