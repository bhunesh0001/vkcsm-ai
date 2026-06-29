from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import binascii

from detector import detect_face
from landmark import detect_landmarks
from headpose import get_head_pose, get_direction

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FaceDetectDTO(BaseModel):
    studentId: int
    image: str


@app.get("/")
def home():
    return {
        "message": "AI Proctor Running"
    }


@app.post("/detect")
def detect(data: FaceDetectDTO):

    try:

        img = data.image

        if "," in img:
            img = img.split(",", 1)[1]

        missing_padding = len(img) % 4

        if missing_padding:
            img += "=" * (4 - missing_padding)

        image = base64.b64decode(img)

        nparr = np.frombuffer(image, np.uint8)

        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {
                "faceCount": 0,
                "violation": "INVALID_IMAGE"
            }

        # -----------------------------
        # Face Detection
        # -----------------------------

        face = detect_face(frame)

        if face["faceCount"] == 0:
            return face

        if face["faceCount"] > 1:
            return face

        # -----------------------------
        # Crop Face
        # -----------------------------

        box = face["box"]

        padding = 30

        x = max(0, box["x"] - padding)
        y = max(0, box["y"] - padding)

        w = min(
            frame.shape[1] - x,
            box["w"] + padding * 2
        )

        h = min(
            frame.shape[0] - y,
            box["h"] + padding * 2
        )

        crop = frame[y:y + h, x:x + w]

        # -----------------------------
        # Face Landmarks
        # -----------------------------

        image_points = detect_landmarks(crop)

        if image_points is None:
            return {
                "faceCount": 1,
                "violation": "FACE_NOT_TRACKED"
            }

        # -----------------------------
        # Head Pose
        # -----------------------------

        pose = get_head_pose(
            image_points,
            crop.shape[1],
            crop.shape[0]
        )

        if pose is None:
            return {
                "faceCount": 1,
                "violation": "FACE_NOT_TRACKED"
            }

        pitch, yaw, roll = pose

        direction = get_direction(
            pitch,
            yaw
        )

        return {
            "faceCount": 1,
            "violation": direction,
            "pitch": round(pitch, 2),
            "yaw": round(yaw, 2),
            "roll": round(roll, 2)
        }

    except binascii.Error:

        return {
            "faceCount": 0,
            "violation": "INVALID_BASE64"
        }

    except Exception as ex:

        return {
            "faceCount": 0,
            "violation": "ERROR",
            "message": str(ex)
        }