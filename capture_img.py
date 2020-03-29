import dlib
import cv2
import os
import re
import json
from pylab import *
from PIL import Image, ImageChops, ImageEnhance
import time
import imutils
train_frame_folder = '../df/Dataset/train'
with open('../df/Dataset/metadata.json', 'r') as file:
    data = json.load(file)
list_of_train_data = [f for f in os.listdir(train_frame_folder) if f.endswith('.mp4')]
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
vid_cnt = 0

# FAKE_CNT = 3000
# REAL_CNT = 3000

def rotatePoint(p, c, rad_angle):
    p[0] -= c[0]
    p[1] -= c[1]
    p = [int(p[0] * math.cos(rad_angle) - p[1] * math.sin(rad_angle) + c[0]),
         int(p[0] * math.sin(rad_angle) + p[1] * math.cos(rad_angle) + c[1])]
    return p

def cropAndAlign(
        img, location, left_eye_loc_x=0.3, left_eye_loc_y=0.3):
    # Find the gravity center of the eye points
    landmarks = dict()
    shape = predictor(img, location)
    dots = np.zeros((68,2), dtype='int')
    for i in range(68):
        dots[i] = (shape.part(i).x, shape.part(i).y)
    
    landmarks["left_eye"] = dots[36:40]
    landmarks["right_eye"] = dots[43:48]

    left_eye = [
        sum([point[0] for point in landmarks["left_eye"]]) //
        len(landmarks["left_eye"]),
        sum([point[1] for point in landmarks["left_eye"]]) //
        len(landmarks["left_eye"])]
    right_eye = [
        sum([point[0] for point in landmarks["right_eye"]]) //
        len(landmarks["right_eye"]),
        sum([point[1] for point in landmarks["right_eye"]]) //
        len(landmarks["right_eye"])]
    y = right_eye[1] - left_eye[1]
    x = right_eye[0] - left_eye[0]
    angle = math.atan2(y, x)
    deg_angle = 180*angle/math.pi

    img = imutils.rotate(img, deg_angle, tuple(left_eye))
    location = [location.top(), location.right(), location.bottom(), location.left()]
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

for vid in list_of_train_data:
    count = 0
    cap = cv2.VideoCapture(os.path.join(train_frame_folder, vid))
    # frameRate = cap.get(5)
    frameRate = 30
    # if(os.path.exists('dataset/cropped-faces-1/train/real/' + vid.split('.')[0] + '_0.png') or \
    #    os.path.exists('dataset/cropped-faces-1/train/fake/' + vid.split('.')[0] + '_0.png') or\
    #    os.path.exists('dataset/cropped-faces-1/train/fake/' + vid.split('.')[0][4::-1] + '_0.png')):
    #    vid_cnt+=1
    #    print("Exists, skip")
    #    continue
    # if data[vid]['label'] == 'REAL':
    #     if REAL_CNT == 0:
    #         continue
    #     REAL_CNT-=1
    # else:
    #     if FAKE_CNT == 0:
    #         continue
    #     FAKE_CNT-=1
    t = time.time()
    frameId = 0#cap.get(1)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret != True:
            break
        if frameId % ((int(frameRate)+1)*1) == 0:
            face_rects, scores, idx = detector.run(frame, 0)
            for i, d in enumerate(face_rects):
                crop_img = cropAndAlign(frame, d)
                try:
                    if data[vid]['label'] == 'REAL':
                        #cv2.imshow('a', cv2.resize(crop_img, (128, 128)))
                        cv2.imwrite('dataset/cropped-faces-1/train/real/'+\
                            vid.split('.')[0]+'_'+str(count)+'.png', cv2.resize(crop_img, (128, 128)))
                    elif data[vid]['label'] == 'FAKE':
                        #cv2.imshow('a', cv2.resize(crop_img, (128, 128)))
                        cv2.imwrite('dataset/cropped-faces-1/train/fake/'+\
                            vid.split('.')[0]+'_'+str(count)+'.png', cv2.resize(crop_img, (128, 128)))
                except:
                    pass
                count+=1
        frameId+=1
    t = time.time() - t
    
    vid_cnt+=1
    print("%d video \t %s \t Time: %f" % (vid_cnt, vid, t))