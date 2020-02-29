import csv
import cv2
import face_recognition
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


LABEL_FILE_NAME = "metadata.json"
SUBMISSION_FILE_NAME = "submission.csv"
TEST_DATA_PATH = "faces_development/"  #"../input/deepfake-detection-challenge/test_videos/"
TRAIN_DATA_PATH = "faces_development/"


def writeSubmissionFile(video_file_names, class_probabilities):
    with open(SUBMISSION_FILE_NAME, "w+") as submission_file:
        class_probabilities = [str(element) for element in class_probabilities]
        writer = csv.writer(submission_file, delimiter=';', quoting=csv.QUOTE_ALL)
        writer.writerow(["filename", "label"])
        for i in range(len(video_file_names)):
            writer.writerow([video_file_names[i], class_probabilities[i]])


def readVideoNames(data_path):
    video_file_names = []
    for directory_name, _, file_names in os.walk(data_path):
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


def cropFaces(test_video_file_names):
    for i in range(len(test_video_file_names)):
        video = cv2.VideoCapture(TEST_DATA_PATH + test_video_file_names[i])
        frames = readFramesFromVideo(video)
        number_of_frames = frames.shape[0]
        for j in range(number_of_frames):
            print("Frame {}/{}".format(j+1, number_of_frames), end="\r")
            frame = frames[j]
            faces_locations = face_recognition.face_locations(frame)
            for y1, x1, y2, x2 in faces_locations:
                face = frame[y1:y2, x2:x1, :]
                cv2.imwrite(str(j)+".png", face)


def loadFaces(path):
    videos = []
    for directory_name, _, file_names in sorted(os.walk(path)):
        if file_names != []:
            video_frames = []
            for frame_name in sorted(file_names):
                video_frames.append(
                    cv2.imread(directory_name + '/' + frame_name))
            videos.append(video_frames)
            yield video_frames


def hogFeatureVector(
        videos, orientations=8, pixels_per_cell=(16, 16),
        cells_per_block=(1, 1), plot_hog=False):
    video_histograms = []
    for i, video in enumerate(videos):
        print("HOG, video {}".format(i+1), end="\r")
        face_histograms = []
        for j, face in enumerate(video):
            feature_vector, hog_image = hog(
                face, orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block, visualize=True,
                multichannel=True)
            histogram, _, _ = plt.hist(feature_vector, orientations)
            face_histograms.append(histogram)

            if plot_hog:
                plt.subplot(131)
                plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                plt.subplot(132)
                plt.imshow(hog_image)
                plt.subplot(133)
                plt.hist(feature_vector, orientations)
                plt.show()
        if face_histograms != []:
            histogram_array = np.array(face_histograms)
            mean_histogram = np.mean(histogram_array, axis=0)
            std_histogram = np.std(histogram_array, axis=0)
            video_histograms.append([mean_histogram, std_histogram])
    return video_histograms


def loadLabels():
    labels = []
    with open(LABEL_FILE_NAME) as labels_json:
        labels_dict = json.load(labels_json)
        for key, value in labels_dict.items():
            if value["label"] == "FAKE":
                labels.append(0)
            else:
                labels.append(1)
    return labels


def visualizeHistograms(video_histograms):
    for i in range(len(video_histograms)):
        mean_histogram = video_histograms[i][0]
        std_histogram = video_histograms[i][1]
        plt.figure(i)
        plt.subplot(121)
        plt.hist(mean_histogram)
        plt.title("Mean")
        plt.subplot(122)
        plt.hist(std_histogram)
        plt.title("STD")
    plt.show()


def checkLabelsToRemove(labels, path):
    i = 0
    existing_labels = []
    for directory_name, _, file_names in sorted(os.walk(path)):
        if directory_name != path:
            if len(os.listdir(directory_name)) > 0:
                existing_labels.append(labels[i])
            i += 1
    return existing_labels


def train():
    labels = loadLabels()
    labels = checkLabelsToRemove(labels, TRAIN_DATA_PATH)
    videos = loadFaces(TRAIN_DATA_PATH)
    video_histograms = hogFeatureVector(videos, plot_hog=False)
    feature_vectors = np.array(video_histograms).reshape(
        len(video_histograms), 16)
    X_train, X_test, y_train, y_test = train_test_split(
        feature_vectors, labels, test_size=0.33, random_state=42)

    print("Training started")
    model = RandomForestClassifier(max_depth=8, random_state=13)
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    precision = accuracy_score(y_test, y_preds)
    print("Precision:", precision)
    pickle.dump(model, open("models/model.sav", 'wb')) 


def test():
    model = pickle.load(open("models/model.sav", 'rb'))
    test_video_file_names = readVideoNames(TEST_DATA_PATH)
    videos = loadFaces(TEST_DATA_PATH)
    video_histograms = hogFeatureVector(videos, plot_hog=False)
    feature_vectors = np.array(video_histograms).reshape(
        len(video_histograms), 16)
    classes = model.predict(feature_vectors)
    class_probabilities = np.random.rand(len(test_video_file_names))
    writeSubmissionFile(test_video_file_names, classes)


def main():
    train()
    #test()


main()
