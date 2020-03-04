import csv
import cv2
import json
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()


LABEL_FILE_NAME = "metadata.json"
SUBMISSION_FILE_NAME = "submission.csv"
TEST_DATA_PATH = "../cropped_faces/faces_test_videos/"  #"../input/deepfake-detection-challenge/test_videos/"
TRAIN_DATA_PATH = "../cropped_faces/faces_train_sample_videos/"
TRAIN_SIZE = 400
FRAME_CNT = 400


def writeSubmissionFile(video_file_names, class_probabilities):
    with open(SUBMISSION_FILE_NAME, "w+") as submission_file:
        class_probabilities = [str(element) for element in class_probabilities]
        writer = csv.writer(submission_file, delimiter=',')
        writer.writerow(["filename", "label"])
        real_cnt = 0
        for i in range(len(video_file_names)):
            real_cnt += 1
            writer.writerow([video_file_names[i][6:], class_probabilities[i]])
        print("REALS: %d" % real_cnt)

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
    for dir in sorted(os.listdir(path)): #range(1, TRAIN_SIZE+1):
        files_png = os.listdir(path + dir + '/' + 'PNG/')
        video_frames = []

        for j in range(len(files_png[0:min(len(files_png), FRAME_CNT)])):
            file = files_png[j]
            video_frames.append(cv2.imread(path + dir + "/" + file))

        yield video_frames
        #videos.append(video_frames)
    #return videos
    #for directory_name, _, file_names in sorted(os.walk(path)):
    #    if file_names != []:
    #        video_frames = []
    #        for frame_name in sorted(file_names):
    #            video_frames.append(
    #                cv2.imread(directory_name + '/' + frame_name))
    #        videos.append(video_frames)
    #        yield video_frames


def hogFeatureVector(
        videos, orientations=8, pixels_per_cell=(16, 16),
        cells_per_block=(1, 1), plot_hog=False):
    def parallel_hist(video):
        print("HOG, video")
        face_histograms = []
        for face in video:
            if(face is None):
                continue
            feature_vector, hog_image = hog(
                face, orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block, visualize=True,
                multichannel=True)
            histogram, _, _ = plt.hist(feature_vector, orientations)
            face_histograms.append(histogram)
        if face_histograms != []:
            histogram_array = np.array(face_histograms)
            mean_histogram = np.mean(histogram_array, axis=0)
            std_histogram = np.std(histogram_array, axis=0)
            return [mean_histogram, std_histogram]
        return [range(8), range(8)]
    video_histograms = Parallel(n_jobs=num_cores)(delayed(parallel_hist)(video) for video in videos)
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


#def visualizeHistograms(video_histograms):
#    for i in range(len(video_histograms)):
#        mean_histogram = video_histograms[i][0]
#        std_histogram = video_histograms[i][1]
#        plt.figure(i)
#        plt.subplot(121)
#        plt.hist(mean_histogram)
#        plt.title("Mean")
#        plt.subplot(122)
#        plt.hist(std_histogram)
#        plt.title("STD")
#    plt.show()


def checkLabelsToRemove(labels, path):
    existing_labels = []
    for i in range(1, len(labels)+1):
        pngs = os.listdir(TRAIN_DATA_PATH + 'video_%03d.mp4/PNG' % i )
        if(len(pngs) == 0):
            print("CYKA BLAT: %03d" % i)
            continue
        existing_labels.append(labels[i-1])
        

    # for directory_name, _, file_names in sorted(os.walk(path)):
    #     if directory_name != path:
    #         if len(os.listdir(directory_name)) > 0:
    #         else:
    #             print(i)
    #         i += 1
    print("%03d LABELS EXIST " % len(existing_labels))
    return existing_labels


def train(feature_path='models/feature_train.dat', save_feature_path='models/feature_train.dat'):
    labels = loadLabels()[0:TRAIN_SIZE]
    #labels = checkLabelsToRemove(labels, TRAIN_DATA_PATH)
    videos = loadFaces(TRAIN_DATA_PATH)
    if(feature_path is None):
        video_histograms = hogFeatureVector(videos, plot_hog=False)
        feature_vectors = np.array(video_histograms).reshape(
            len(video_histograms), 16)
        pickle.dump(feature_vectors, open(save_feature_path, "wb"))
    else:
        feature_vectors = pickle.load(open(feature_path, "rb"))
    X_train, X_test, y_train, y_test = train_test_split(
        feature_vectors, labels, test_size=0.33, random_state=42)

    print("Training started")
    model = RandomForestClassifier(max_depth=8, random_state=13)
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    print(labels)
    precision = accuracy_score(y_test, y_preds)
    evaluation_score = logLoss(y_test, y_preds)
    print("Precision:", precision)
    print("Evaluation score:", evaluation_score)
    pickle.dump(model, open("models/model.sav", 'wb')) 


def logLoss(y_pred, y_true):
    if len(y_pred) != len(y_true):
        Exception("No same length")
    sum_term = 0
    eps = 1e-15
    n = len(y_pred)
    for i in range(n):
        y_p = y_pred[i]
        y_t = y_true[i]
        sum_term += y_t * np.log(y_p + eps) + (1 - y_t) * np.log(1 - y_p + eps)
    return -1 * sum_term / n


def test(feature_path='models/feature_test.dat', save_feature_path='models/feature_test.dat'):
    model = pickle.load(open("models/model.sav", 'rb'))
    test_video_file_names = sorted(os.listdir(TEST_DATA_PATH)) #readVideoNames(TEST_DATA_PATH)
    videos = loadFaces(TEST_DATA_PATH)
    if(feature_path is None):
        video_histograms = hogFeatureVector(videos, plot_hog=False)
        feature_vectors = np.array(video_histograms).reshape(
            len(video_histograms), 16)
        pickle.dump(feature_vectors, open(save_feature_path, "wb"))
    else:
        feature_vectors = pickle.load(open(feature_path, "rb"))
    classes = model.predict(feature_vectors)
    class_probabilities = np.random.rand(len(test_video_file_names))
    writeSubmissionFile(test_video_file_names, classes)


def main():    
    train(None)
    test(None)


main()
