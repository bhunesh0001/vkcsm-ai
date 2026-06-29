import os
import cv2

MODEL_PATH = os.path.join(
    "models",
    "face_detection_yunet_2023mar.onnx"
)

detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320, 320)
)


def detect_face(frame):

    h, w = frame.shape[:2]

    detector.setInputSize((w, h))

    _, faces = detector.detect(frame)

    if faces is None:

        return {
            "faceCount": 0,
            "violation": "NO_FACE"
        }

    count = len(faces)

    if count > 1:

        return {
            "faceCount": count,
            "violation": "MULTIPLE_FACE"
        }

    face = faces[0]

    x = int(face[0])
    y = int(face[1])
    fw = int(face[2])
    fh = int(face[3])

    centerX = x + fw / 2
    centerY = y + fh / 2

    return {
        "faceCount": 1,
        "violation": "OK",
        "box": {
            "x": x,
            "y": y,
            "w": fw,
            "h": fh,
            "centerX": centerX,
            "centerY": centerY
        }
    }