import cv2
import numpy as np
import json
import os
from mtcnn import MTCNN
import tensorflow.compat.v1 as tf
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
detector = MTCNN()
files = os.listdir('.\\Deepfake\\test_videos\\')
faces = []


for file in files:
    VIDEO_DIR = ".\\Deepfake\\test_videos\\video_%s" % (file)
    FACE_DIR = ".\\Deepfake\\faces\\video_%s" % (file)

    cnt = 1
    os.mkdir(FACE_DIR)
    cap = cv2.VideoCapture(VIDEO_DIR)
    while(cap.isOpened()): 
    #for file in files:
        print("Video #%s Frame #%03d" % (file, cnt))
        ret, img = cap.read()
        #if(ret == False):
        #    break
        #img = cv2.imread(file) 
        #plt.imshow(img)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        box = detector.detect_faces(img)
        if(len(box)==0):
            continue

        box = box[0]["box"]
        img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        cv2.imshow("frame", img)
        cv2.imwrite(FACE_DIR + "\\%03d.png" % (cnt), img)
        cnt+=1

