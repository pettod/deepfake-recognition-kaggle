import cv2


DNN = "CAFFE"
if DNN == "CAFFE":
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "opencv_face_detector_uint8.pb"
    configFile = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

cap = cv2.VideoCapture("../input/deepfake-detection-challenge/test_videos/jhczqfefgw.mp4")
while(cap.isOpened()):
    ret, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)

    frame_height = img.shape[0]
    frame_width = img.shape[1]

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    conf_threshold = 0.5
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_width = x2 - x1
            face_height = y2 - y1
            if face_height < face_width:
                coordinate_change = (face_width - face_height) // 2
                x1 += coordinate_change
                x2 -= coordinate_change
            else:
                coordinate_change = (face_height - face_width) // 2
                y1 += coordinate_change
                y2 -= coordinate_change
            print(x1, y1, x2, y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.imshow("y", img)
            cv2.waitKey(0)

