import dlib
import cv2


cap = cv2.VideoCapture("../input/deepfake-detection-challenge/test_videos/kmqkiihrmj.mp4")
ret, img = cap.read()

cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
dets = cnn_face_detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))
for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
        i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

cv2.rectangle(img,(d.rect.left(),d.rect.top()),(d.rect.right(),d.rect.bottom()),(0,255,0),2)
cv2.imshow("y", img)
cv2.waitKey(0)