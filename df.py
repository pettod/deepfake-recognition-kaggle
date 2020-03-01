import cv2
import numpy as np
import json
import os
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
#tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
config = tf.ConfigProto(device_count = {'GPU':2})
config.gpu_options.per_process_gpu_memory_fraction = 1
tf.keras.backend.set_session(tf.Session(config=config))
    
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

with tf.device('/device:GPU:0'):
    detector = MTCNN()
    for categ in ['train_sample_videos']:
        files = os.listdir('.\\Deepfake\\%s\\' % categ)
        faces = []
        for i in range(112, len(files)):
            file = files[i]
            VIDEO_DIR = ".\\Deepfake\\%s\\%s" % (categ, file)
            FACE_DIR = ".\\Deepfake\\faces_%s\\video_%s" % (categ, file)
            cnt = 0
            os.mkdir(FACE_DIR)
            cap = cv2.VideoCapture(VIDEO_DIR)
            while(cap.isOpened()): 
            #for file in files:
                cnt+=1
                print("Video #%03d Frame #%03d" % (i, cnt))
                ret, img = cap.read()
                #img = cv2.resize(img, (,20))
                if not ret:
                    break
                #img = cv2.imread(file) 
                #plt.imshow(img)
                #if cv2.waitKey(0) & 0xFF == ord('q'): 
                #    break
                box_faces = detector.detect_faces(img)
                #cv2.imwrite(FACE_DIR + "\\%03d.png" % cnt, img)
                with open(FACE_DIR + '\\%03d.json' % cnt, 'w') as outfile:
                    json.dump(box_faces, outfile)
                for j in range(len(box_faces)):
                    box = box_faces[j]["box"]
                    img_face = img[max(0,box[1]-20):min(np.shape(img)[0],box[1]+box[3]+20),
                    max(0, box[0]-20):min(np.shape(img)[1],box[0]+box[2]+20)]
                    #cv2.imshow("frame", img)
                    if(len(box)!=0):
                        cv2.imwrite(FACE_DIR + "\\%03d_%d.png" % (cnt, j), img_face)

   