import cv2


MODEL_FILE = "../input/face-detection-config-files/res10_300x300_ssd_iter_140000_fp16.caffemodel"
CONFIG_FILE = "../input/face-detection-config-files/deploy.prototxt"
VIDEO_PATH = "../input/deepfake-detection-challenge/test_videos/acazlolrpz.mp4"


def getFaces(path, image_size, net):
    cap = cv2.VideoCapture(path)
    faces = []
    if (not cap.isOpened()):
        print(path)
        print("VIDEO НЕ ОТКРЫВАЕТСЯ СУКА БЛЯТЬ")
        return []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
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
                face = cv2.resize(frame[y1:y2, x1:x2], image_size)
                faces.append(face)
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.imshow("y", frame)
        #cv2.waitKey(0)
    return faces


if __name__ == "__main__":
    net = cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)
    faces_in_video = getFaces(VIDEO_PATH, (224, 224), net)
    print("Number of detected faces in video:", len(faces_in_video))
    for face in faces_in_video:
        cv2.imshow("face", face)
        cv2.waitKey(0)
