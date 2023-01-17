import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def crop(img):
<<<<<<< HEAD
    h, w, c= img.shape
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        result = face_detection.process(img)
=======
    h, w, c = img.shape
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        result = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
>>>>>>> 69d9147fdd865a61d1f8875087463713e486b0e5
        box = result.detections[0].location_data.relative_bounding_box
        x1 = round(box.xmin * w)
        x2 = round((box.xmin + box.width) * w)
        y1 = round(box.ymin * h)
        y2 = round((box.ymin + box.height) * h)
<<<<<<< HEAD
        print(img.shape)
        roi_img = img[y1:y2, x1:x2]
        print(roi_img.shape)
    return roi_img
=======
        roi_img = img[y1:y2, x1:x2]
    return roi_img
>>>>>>> 69d9147fdd865a61d1f8875087463713e486b0e5
