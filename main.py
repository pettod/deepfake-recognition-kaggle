import csv
import cv2
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


def readFramesFromVideo(video):
    frames = []
    while(video.isOpened()):
        is_readable, frame = video.read()
        if is_readable:
            frames.append(frame)
        else:
            break
    return np.array(frames)


def main():
    test_video_file_names = readTestVideoNames()
    video = cv2.VideoCapture(TEST_DATA_PATH + test_video_file_names[1])
    frames = readFramesFromVideo(video)
    print("Frames shape:", frames.shape)
    number_of_frames = frames.shape[0]
    for i in range(number_of_frames):
        print("Frame {}/{}".format(i+1, number_of_frames), end="\r")
        frame = frames[i]
        faces_locations = face_recognition.face_locations(frame)
        #hog_descriptor = cv2.HOGDescriptor()
        for y1, x1, y2, x2 in faces_locations:
            face = frame[y1:y2, x2:x1, :]
            #cv2.imwrite(str(i)+".png", face)
            #hog = hog_descriptor.compute(face)

    #class_probabilities = np.random.rand(len(test_video_file_names))
    #writeSubmissionFile(test_video_file_names, class_probabilities)
    

main()
