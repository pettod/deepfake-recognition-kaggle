import cv2
import json
import keras
from keras.applications import InceptionResNetV2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, \
    AveragePooling2D, Input, Flatten, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.regularizers import l2
import math
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
import time


# Data parameters
BATCH_SIZE = 16
EVERY_ITH_FRAME = 5
IMAGE_SIZE = (224, 224)
NUMBER_OF_FACES_PER_VIDEO = 10
NUMBER_OF_CNN_LAYERS = 56

# Training and testing paths
CNN_MODEL_FILE_NAME = "resnet{}_best.h5".format(NUMBER_OF_CNN_LAYERS)
LABELS_PATH = "../input/deepfake-detection-challenge/metadata.json"
LOAD_MODEL_PATH = "../input/resnet-model/{}".format(CNN_MODEL_FILE_NAME)
RAW_TRAIN_DATA_DIRECTORY = "../input/deepfake-detection-challenge/train_sample_videos"
SUBMISSION_CSV = "../input/deepfake-detection-challenge/sample_submission.csv"
TEST_DATA_DIRECTORY = "../input/deepfake-detection-challenge/test_videos"
TRAIN_DATA_DIRECTORY = "../input/cropped-faces-{}".format(NUMBER_OF_FACES_PER_VIDEO)
TRAIN_DIRECTORY = TRAIN_DATA_DIRECTORY + "/train"
VALIDATION_DIRECTORY = TRAIN_DATA_DIRECTORY + "/validation"

# Face detection model paths
FACE_DETECTION_MODEL_FILE = "../input/face-detection-config-files/res10_300x300_ssd_iter_140000_fp16.caffemodel"
FACE_DETECTION_CONFIG_FILE = "../input/face-detection-config-files/deploy.prototxt"


def cropAndAlign(
        img, location, landmarks, left_eye_loc_x=0.3, left_eye_loc_y=0.3):
    # Find the gravity center of the eye points
    left_eye = [
        sum([point[0] for point in landmarks["left_eye"]]) //
        len(landmarks["left_eye"]),
        sum([point[1] for point in landmarks["left_eye"]]) //
        len(landmarks["left_eye"])]
    right_eye = [
        sum([point[0] for point in landmarks["right_eye"]]) //
        len(landmarks["left_eye"]),
        sum([point[1] for point in landmarks["right_eye"]]) //
        len(landmarks["left_eye"])]
    y = right_eye[1] - left_eye[1]
    x = right_eye[0] - left_eye[0]
    angle = math.atan2(y, x)
    deg_angle = 180*math.atan2(y, x)/math.pi

    img = imutils.rotate(img, deg_angle, tuple(left_eye))
    location = list(location)
    location[3], location[0] = rotatePoint(
        [location[3], location[0]], left_eye, angle)
    location[1], location[2] = rotatePoint(
        [location[1], location[2]], left_eye, angle)

    w = abs(location[1] - location[3])
    h = abs(location[0] - location[2])
    size = max(w, h)

    left_eye[0] -= location[3]
    left_eye[1] -= location[0]

    if(w > h):
        img = imutils.translate(
            img, size * left_eye_loc_x - left_eye[0],
            size * left_eye_loc_y - left_eye[1] - (w-h)//2)
        img = img[
            location[0] - (w-h)//2:location[0] -
            (w-h)//2+w, location[3]:location[3]+w]
    else:
        img = imutils.translate(
            img, size * left_eye_loc_x - left_eye[0] - (h-w)//2,
            size * left_eye_loc_y - left_eye[1])
        img = img[
            location[0]:location[0]+h, location[3] -
            (h-w)//2: location[3] - (h-w)//2+h]
    return img


def cropFixedSizedFaceFromCoordinates(detections, i, frame, image_size):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    x1 = int(detections[0, 0, i, 3] * frame_width)
    x2 = int(detections[0, 0, i, 5] * frame_width)
    y1 = int(detections[0, 0, i, 4] * frame_height)
    y2 = int(detections[0, 0, i, 6] * frame_height)

    # Limit coordinates if they go over frame boundaries
    x1 = min(np.shape(frame)[1], max(0, x1))
    x2 = min(np.shape(frame)[1], max(0, x2))
    y1 = min(np.shape(frame)[0], max(0, y1))
    y2 = min(np.shape(frame)[0], max(0, y2))
    face_width = x2 - x1
    face_height = y2 - y1

    # Make detected area square
    if face_height < face_width:
        coordinate_change = (face_width - face_height) // 2
        x1 += coordinate_change
        x2 -= coordinate_change
    else:
        coordinate_change = (face_height - face_width) // 2
        y1 += coordinate_change
        y2 -= coordinate_change
    face = cv2.resize(frame[y1:y2, x1:x2], image_size)

    return face


def getFaces(
        path, image_size, net, every_ith_frame=1, confidence_threshold=0.5,
        remove_outliers=True):
    cap = cv2.VideoCapture(path)
    faces = []
    i_frame = 1
    if (not cap.isOpened()):
        print("Cannot open video:", path)
        return faces
    while(cap.isOpened()):

        # Read video frame
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames and read only every i'th frame
        if i_frame != every_ith_frame:
            i_frame += 1
            continue
        else:
            i_frame = 1

        # Detect faces from frame
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()

        # Iterate all detections
        detected_faces_in_frame = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Detected face
            if confidence > confidence_threshold:
                detected_faces_in_frame += 1
                face = cropFixedSizedFaceFromCoordinates(
                    detections, i, frame, image_size)

                # Add new detected human in a video/frame or add to existing
                # lists of detected humans
                if len(faces) < detected_faces_in_frame:
                    faces.append([face])

                # Only one human in video
                elif len(faces) == 1:
                    faces[0].append(face)

                # Multiple humans in video, check which human face is this
                else:
                    errors = []
                    for human_faces in faces:
                        face_mean = np.mean(np.array(human_faces), axis=0)
                        l1_color_error = np.mean(np.abs(face_mean - face))
                        errors.append(l1_color_error)
                    faces[np.argmin(errors)].append(face)

    # Remove outliers from each human faces
    if remove_outliers:
        faces, outliers = removeOutliers(faces)

    return faces


def getBatchGenerator(data_directory, image_size, batch_size, normalize=True):
    read_image_size = (
        image_size[0],
        image_size[1] * NUMBER_OF_FACES_PER_VIDEO)
    gen = keras.preprocessing.image.ImageDataGenerator()
    batches = gen.flow_from_directory(
        data_directory, target_size=read_image_size, class_mode="categorical",
        shuffle=True, batch_size=batch_size)

    # Remove horizontal concatenation and stack faces in channel dimesion
    while True:
        video_faces, labels = batches.next()
        batch_samples = np.array([
            stackFacesFromSample(sample, image_size)
            for sample in video_faces])
        if normalize:
            batch_samples = normalizeArray(batch_samples)
        yield batch_samples, labels


def getModel(input_shape):
    model = InceptionResNetV2(
        include_top=False, weights=None, input_shape=input_shape, classes=2)
    for layer in model.layers:
        layer.trainable = False
    last_layer = model.layers[-1].output
    pool_layer = GlobalAveragePooling2D()(last_layer)
    output_layer = Dense(2, activation="softmax")(pool_layer)
    return Model(model.input, output_layer)


def getNumberOfSteps(data_directory, batch_size):
    return math.floor(sum(
        [len(files) for r, d, files in os.walk(data_directory)]) / batch_size)


def loadLabels():
    # Load labels from metadata
    labels = []
    with open(LABELS_PATH) as labels_json:
        labels_dict = json.load(labels_json)
        for key, value in labels_dict.items():
            if value["label"] == "FAKE":
                labels.append(0)
            else:
                labels.append(1)
    return labels


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def normalizeArray(sample):
    return sample / 255 - 0.5


def removeOutliers(human_faces_in_video, outlier_detection_factor=1.5):
    def removeOutliersFromHumanFaces(human_faces):
        # No faces in list
        if len(human_faces) == 0:
            return [], []

        # Compute similarity scores
        faces_array = np.array(human_faces)
        face_mean = np.mean(faces_array, axis=0).astype(np.uint8)
        similarity_scores = []
        for i, face in enumerate(human_faces):
            l1_color_error = np.mean(np.abs(face_mean - face))
            similarity_scores.append(int(l1_color_error))

        # Define outlier threshold
        similarity_median = np.median(similarity_scores)
        outlier_threshold = outlier_detection_factor*similarity_median
        true_faces = []
        outliers = []

        # Find outliers and true faces
        for i, face in enumerate(human_faces):
            face_similarity_score = similarity_scores[i]
            if face_similarity_score > outlier_threshold:
                outliers.append(face)
            else:
                true_faces.append(face)
        return true_faces, outliers

    true_faces = []
    outliers = []
    for human_faces in human_faces_in_video:
        max_faces_per_human = max(
            [len(human_faces) for human_faces in human_faces_in_video])

        # Remove outliers from humans that were detected often
        # (rarely detected humans can be outliers)
        if len(human_faces) > max_faces_per_human / 2:
            human_faces, outliers_in_human_faces = \
                removeOutliersFromHumanFaces(human_faces)
            true_faces.append(human_faces)
            outliers.append(outliers_in_human_faces)
        else:
            outliers.append(human_faces)

    return true_faces, outliers


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def rotatePoint(p, c, rad_angle):
    p[0] -= c[0]
    p[1] -= c[1]
    p = [int(p[0] * math.cos(rad_angle) - p[1] * math.sin(rad_angle) + c[0]),
         int(p[0] * math.sin(rad_angle) + p[1] * math.cos(rad_angle) + c[1])]
    return p


def stackFacesFromSample(horizontal_concatenated_sample, image_size):
    gray_faces = np.mean(horizontal_concatenated_sample, axis=-1)
    sample = []
    for j in range(NUMBER_OF_FACES_PER_VIDEO):
        start_index = j * image_size[1]
        end_index = (j+1) * image_size[1]
        sample.append(gray_faces[:, start_index:end_index])
    return np.moveaxis(np.array(sample), 0, -1)


def test(print_time=True):
    # Load model
    print("Loading model")
    t_start_program = time.time()
    net = cv2.dnn.readNetFromCaffe(
        FACE_DETECTION_CONFIG_FILE, FACE_DETECTION_MODEL_FILE)
    model = load_model(LOAD_MODEL_PATH)
    model_loading_time = round(time.time() - t_start_program, 2)
    print("Model loading time: {}s".format(model_loading_time))

    # Load CSV file
    submission_file = pd.read_csv(SUBMISSION_CSV)
    submission_file.label = submission_file.label.astype(float)

    # Loop test videos
    print("Cropping faces from videos")
    number_of_videos = len(os.listdir(TEST_DATA_DIRECTORY))
    for i, file_name in enumerate(sorted(os.listdir(TEST_DATA_DIRECTORY))):
        try:

            # Crop faces from test video
            t_start_video = time.time()
            humans_in_video = getFaces(
                TEST_DATA_DIRECTORY + '/' + file_name, IMAGE_SIZE, net,
                EVERY_ITH_FRAME)
            faces_loading_time = round(time.time() - t_start_video, 2)

            # Loop each human's faces
            predictions = []
            for human_faces in humans_in_video:

                # Predict score for each stack of face
                t_start_predicting = time.time()
                gray_faces = list(np.mean(np.array(human_faces), axis=-1))
                number_of_samples = int(
                    len(gray_faces) / NUMBER_OF_FACES_PER_VIDEO)

                # Create samples by taking random faces
                for j in range(number_of_samples):
                    random_face_indices = sorted(random.sample(
                        range(len(gray_faces)), NUMBER_OF_FACES_PER_VIDEO))
                    picked_faces = []
                    for k in reversed(random_face_indices):
                        picked_faces.append(gray_faces[k])
                        gray_faces.pop(k)
                    picked_faces = np.moveaxis(np.array(picked_faces), 0, -1)
                    sample = normalizeArray(np.expand_dims(
                        picked_faces, axis=0))
                    predictions.append(model.predict(sample)[0])
                t_video_processed = time.time()
                prediction_time = round(
                    t_video_processed - t_start_predicting, 2)
                total_spent_time = int(
                    (t_video_processed - t_start_program) / 60)

            # Print processing times
            if print_time:
                number_of_faces = sum(
                    [len(human_faces) for human_faces in humans_in_video])
                print((
                    "Video: {:5}/{}, {:20}. Number of faces: {:5}. " +
                    "Cropping faces time: {:7}s. Prediction time: {:6}s. " +
                    "Total time: {:4}min").format(
                        i+1, number_of_videos, file_name, number_of_faces,
                        faces_loading_time, prediction_time, total_spent_time))
            else:
                print("Video: {}/{}".format(i+1, number_of_videos))

            # Compute final score for video and write to CSV
            video_score = 0.5
            if len(predictions) > 0:
                final_scores = []
                for prediction in predictions:
                    fake_score = prediction[0]
                    real_score = prediction[1]
                    if fake_score > real_score:
                        final_scores.append(1 - fake_score)
                    else:
                        final_scores.append(real_score)
                video_score = np.mean(np.array(final_scores))

        # Handle unexpected exceptions, give score 0.5
        except Exception as e:
            print("Exception thrown, video: {}/{}, {}".format(
                i+1, number_of_videos, file_name))
            video_score = 0.5

        # Write video file name and score
        submission_file.at[i, "filename"] = file_name
        submission_file.at[i, "label"] = video_score

    # Write CSV to file
    submission_file.to_csv("submission.csv", index=False)
    print("Submission file written")


def train():
    with tf.device("/device:GPU:0"):

        # Check train and validation directories exist
        all_paths = [TRAIN_DIRECTORY, VALIDATION_DIRECTORY]
        for path in all_paths:
            if not os.path.isdir(path):
                print("Directory does not exist: {}".format(path))
                return

        # Load batch generators
        train_batches = getBatchGenerator(
            TRAIN_DIRECTORY, IMAGE_SIZE, BATCH_SIZE)
        validation_batches = getBatchGenerator(
            VALIDATION_DIRECTORY, IMAGE_SIZE, BATCH_SIZE)

        # Create model
        input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], NUMBER_OF_FACES_PER_VIDEO)
        model = getModel(input_shape)
        model.compile(
            optimizer=Adam(lr=1e-4), loss="binary_crossentropy",
            metrics=["accuracy"])

        # Fit data
        early_stopping = EarlyStopping(patience=10)
        checkpointer = ModelCheckpoint(
            CNN_MODEL_FILE_NAME, verbose=1, save_best_only=True)
        model.fit_generator(
            train_batches,
            steps_per_epoch=getNumberOfSteps(TRAIN_DIRECTORY, BATCH_SIZE),
            epochs=1000,
            callbacks=[early_stopping, checkpointer],
            validation_data=validation_batches,
            validation_steps=getNumberOfSteps(
                VALIDATION_DIRECTORY, BATCH_SIZE))
        model.save("resnet{}_final.h5".format(NUMBER_OF_CNN_LAYERS))


if __name__ == "__main__":
    #train()
    test()
