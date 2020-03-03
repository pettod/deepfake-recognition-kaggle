import matplotlib.pyplot as plt
import os
import random


TRAIN_DATA_PATH = "../cropped_faces/deepfake/faces_train_videos/"
TARGET_PATH_REAL = "../cropped_faces/train/real/"
TARGET_PATH_FAKE = "../cropped_faces/train/fake/"
NUMBER_OF_RANDOM_FACES = 5


def main():

    # Read video folder
    for path in sorted(os.listdir(TRAIN_DATA_PATH)):
        video_frames_path = TRAIN_DATA_PATH + path + '/PNG/'
        video_frames = []

        # Read frames inside video folder
        frame_names = sorted(os.listdir(video_frames_path))
        for frame_name in frame_names:
            frame_path = video_frames_path + frame_name
            video_frames.append(plt.imread(frame_path))

        random_frame_indices = sorted(random.sample(
            range(len(frame_names)), NUMBER_OF_RANDOM_FACES))


main()
