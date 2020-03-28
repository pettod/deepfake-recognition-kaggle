import cv2
import numpy as np
import time

video_name = "../input/deepfake-detection-challenge/test_videos/aassnaulhq.mp4"

times = []
for i in range(10):
    t_start = time.time()
    cap = cv2.VideoCapture(video_name)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    t_read = time.time()
    times.append(t_read - t_start)
read_frame_by_frame = round(np.mean(np.array(times)), 2)
print("Time to read all the frames in video: {}".format(read_frame_by_frame))
