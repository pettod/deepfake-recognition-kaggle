import csv
import face_recognition
import numpy as np
import os


TEST_DATA_PATH = "../input/deepfake-detection-challenge/test_videos/"
SUBMISSION_FILE_NAME = "submission.csv"


def writeSubmissionFile(video_file_names, class_probabilities):
    with open(SUBMISSION_FILE_NAME, "w+") as submission_file:
        class_probabilities = [str(element) for element in class_probabilities]
        writer = csv.writer(submission_file, delimiter=';', quoting=csv.QUOTE_ALL)
        writer.writerow(["filename", "label"])
        for i in range(len(video_file_names)):
            writer.writerow([video_file_names[i], class_probabilities[i]])


def readTestVideoNames():
    video_file_names = []
    for directory_name, _, file_names in os.walk(TEST_DATA_PATH):
        for video_name in sorted(file_names):
            video_file_names.append(video_name)
    return video_file_names


def main():
    test_video_file_names = readTestVideoNames()
    class_probabilities = np.random.rand(len(test_video_file_names))
    writeSubmissionFile(test_video_file_names, class_probabilities)
    

main()
