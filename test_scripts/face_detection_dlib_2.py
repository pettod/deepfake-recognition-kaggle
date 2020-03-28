import cv2
import dlib
import time
import numpy as np
from scipy import ndimage

# Import from project files
from main_v2 import getFaces, removeOutliers


MODEL_FILE = "../input/face-detection-config-files/res10_300x300_ssd_iter_140000_fp16.caffemodel"
CONFIG_FILE = "../input/face-detection-config-files/deploy.prototxt"
VIDEO_PATH = "../input/deepfake-detection-challenge/aajxdztmpb.mp4"
EVERY_ITH_FRAME = 10
IMAGE_SIZE = (224, 224)


if __name__ == "__main__":
    # Detect faces
    detector = dlib.get_frontal_face_detector()
    t_start_detect_faces = time.time()
    faces_in_video = getFaces(
        VIDEO_PATH, IMAGE_SIZE, detector, EVERY_ITH_FRAME, remove_outliers=False)
    face_detection_time = round(time.time() - t_start_detect_faces, 2)

    # Show outliers
    true_faces, outliers = removeOutliers(
        faces_in_video, outlier_detection_factor=1.5)
    for i in range(max(len(true_faces), len(outliers))):
        if i < len(true_faces):
            if len(true_faces[i]) > 0:
                h_stacked_true_faces = np.hstack(true_faces[i])
                cv2.imshow("true_faces", h_stacked_true_faces)
        if i < len(outliers):
            if len(outliers[i]) > 0:
                h_stacked_outliers = np.hstack(outliers[i])
                cv2.imshow("outliers", h_stacked_outliers)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
