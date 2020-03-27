import dlib
import cv2
import os
import re
import json
from pylab import *
from PIL import Image, ImageChops, ImageEnhance

train_frame_folder = '../df/Dataset/train'
with open('../df/Dataset/metadata.json', 'r') as file:
    data = json.load(file)
list_of_train_data = [f for f in os.listdir(train_frame_folder) if f.endswith('.mp4')]
detector = dlib.get_frontal_face_detector()
vid_cnt = 0

FAKE_CNT = 3000
REAL_CNT = 3000

for vid in list_of_train_data:
    count = 0
    cap = cv2.VideoCapture(os.path.join(train_frame_folder, vid))
    frameRate = cap.get(5)
    if data[vid]['label'] == 'REAL':
        if REAL_CNT == 0:
            continue
        REAL_CNT-=1
    else:
        if FAKE_CNT == 0:
            continue
        FAKE_CNT-=1
    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if ret != True:
            break
        if frameId % ((int(frameRate)+1)*1) == 0:
            face_rects, scores, idx = detector.run(frame, 0)
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                crop_img = frame[y1:y2, x1:x2]
                try:
                    if data[vid]['label'] == 'REAL':
                        cv2.imwrite('dataset/real/'+vid.split('.')[0]+'_'+str(count)+'.png', cv2.resize(crop_img, (128, 128)))
                    elif data[vid]['label'] == 'FAKE':
                        cv2.imwrite('dataset/fake/'+vid.split('.')[0]+'_'+str(count)+'.png', cv2.resize(crop_img, (128, 128)))
                except:
                    pass
                count+=1
    vid_cnt+=1
    print("%d video \t %s" % (vid_cnt, vid))