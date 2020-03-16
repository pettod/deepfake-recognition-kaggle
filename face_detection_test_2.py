import cv2
import time


MODEL_FILE = "../input/face-detection-config-files/res10_300x300_ssd_iter_140000_fp16.caffemodel"
CONFIG_FILE = "../input/face-detection-config-files/deploy.prototxt"
VIDEO_PATH = "../input/deepfake-detection-challenge/test_videos/gahgyuwzbu.mp4"
EVERY_ITH_FRAME = 10
IMAGE_SIZE = (224, 224)


def getFaces(
        path, image_size, net, every_ith_frame=1, confidence_threshold=0.5):
    cap = cv2.VideoCapture(path)
    faces = []
    if (not cap.isOpened()):
        print("Cannot open video:", path)
        return faces
    i_frame = 1
    while(cap.isOpened()):

        # Read video frame
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames and read only every i'th frame
        if i_frame != every_ith_frame:
            i_frame += 1
            continue
        else:
            i_frame = 1
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Detect faces from frame
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()

        # Iterate all detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Detected face
            if confidence > confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                face_width = x2 - x1
                face_height = y2 - y1

                # Make detected area square
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("y", frame)
        cv2.waitKey(0)
    return faces


if __name__ == "__main__":
    net = cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)
    t_start_detect_faces = time.time()
    faces_in_video = getFaces(VIDEO_PATH, IMAGE_SIZE, net, EVERY_ITH_FRAME)
    face_detection_time = round(time.time() - t_start_detect_faces, 2)
    print("Number of detected faces: {}. Face detection time: {}s".format(
            len(faces_in_video), face_detection_time))
    exit()
    for face in faces_in_video:
        cv2.imshow("face", face)
        cv2.waitKey(0)
