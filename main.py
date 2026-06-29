from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np

from landmark import detect_landmarks
from headpose import get_head_direction

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

    img = data.image

    if "," in img:
        img = img.split(",")[1]

    image = base64.b64decode(img)

    nparr = np.frombuffer(image, np.uint8)

    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = detect_landmarks(frame)

    # No face
    if len(result.face_landmarks) == 0:
        return {
            "faceCount": 0,
            "violation": "NO_FACE"
        }

    # Multiple faces
    if len(result.face_landmarks) > 1:
        return {
            "faceCount": len(result.face_landmarks),
            "violation": "MULTIPLE_FACE"
        }

      # Debug

    return {
        "faceCount": len(result.face_landmarks),
        "matrixCount": len(result.facial_transformation_matrixes),
        "matrix": str(result.facial_transformation_matrixes)
    }
    
    return {
        "faceCount": 1,
        "violation": direction
    }