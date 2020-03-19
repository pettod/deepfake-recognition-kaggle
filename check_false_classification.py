import cv2
import datetime
from keras.models import load_model
import numpy as np
import os
import time

# Import from project file
from main import \
    stackFacesFromSample, \
    IMAGE_SIZE, \
    LOAD_MODEL_PATH, \
    NUMBER_OF_FACES_PER_VIDEO, \
    VALIDATION_DIRECTORY


SAVE_DIRECTORY = "../input/wrong_classifications/{}".format(
    datetime.datetime.fromtimestamp(
        time.time()).strftime("%Y-%m-%d__%H-%M-%S"))


def saveWrongClassifications(model, sample_path, save_directory, class_number):
    number_of_samples = len(os.listdir(sample_path))
    for i, file_name in enumerate(sorted(os.listdir(sample_path))):
        print("Sample: {:5}/{}, {}".format(
                i+1, number_of_samples, file_name))

        # Read and predict sample
        sample_rgb_horizontal = cv2.imread("{}/{}".format(
            sample_path, file_name))
        sample = np.expand_dims(stackFacesFromSample(
            sample_rgb_horizontal, IMAGE_SIZE), axis=0)
        prediction = model.predict(sample)[0]

        # Wrong prediction
        wrong_prediction = prediction[0] < prediction[1]
        if class_number == 1:
            wrong_prediction = prediction[0] > prediction[1]
        if wrong_prediction:
            fake_score = int(round(prediction[0], 2) * 100)
            real_score = int(round(prediction[1], 2) * 100)
            new_file_name = "{}/{}_{}_{}.png".format(
                save_directory, file_name.split('.')[0], fake_score,
                real_score)
            cv2.imwrite(new_file_name, sample_rgb_horizontal)


def checkFalseClassifications():
    # Load model
    print("Loading model")
    t_start_program = time.time()
    model = load_model(LOAD_MODEL_PATH)
    model_loading_time = round(time.time() - t_start_program, 2)
    print("Model loading time: {}s".format(model_loading_time))

    # Create saving directories
    false_fake_directory = "{}/{}".format(SAVE_DIRECTORY, "false_fake")
    false_real_directory = "{}/{}".format(SAVE_DIRECTORY, "false_real")
    os.makedirs(false_fake_directory)
    os.makedirs(false_real_directory)

    # Save wrongly classified fake samples and wrongly classified real samples
    sample_paths = [
        VALIDATION_DIRECTORY + "/fake", VALIDATION_DIRECTORY + "/real"]
    saveWrongClassifications(model, sample_paths[0], false_fake_directory, 0)
    saveWrongClassifications(model, sample_paths[1], false_real_directory, 1)


if __name__ == "__main__":
    checkFalseClassifications()
