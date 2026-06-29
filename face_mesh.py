import cv2
from mediapipe.python.solutions import face_mesh

mp_face_mesh = face_mesh

face_mesh_detector = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


def detect_face_direction(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh_detector.process(rgb)

    if not results.multi_face_landmarks:
        return {
            "faceCount": 0,
            "violation": "NO_FACE"
        }

    if len(results.multi_face_landmarks) > 1:
        return {
            "faceCount": len(results.multi_face_landmarks),
            "violation": "MULTIPLE_FACE"
        }

    landmarks = results.multi_face_landmarks[0].landmark

    nose = landmarks[1]

    x = nose.x
    y = nose.y

    if x < 0.40:
        return {
            "faceCount": 1,
            "violation": "LOOKING_LEFT"
        }

    if x > 0.60:
        return {
            "faceCount": 1,
            "violation": "LOOKING_RIGHT"
        }

    if y < 0.35:
        return {
            "faceCount": 1,
            "violation": "LOOKING_UP"
        }

    if y > 0.65:
        return {
            "faceCount": 1,
            "violation": "LOOKING_DOWN"
        }

    return {
        "faceCount": 1,
        "violation": "OK"
    }