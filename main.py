from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np

from face_mesh import detect_face_direction

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Development ke liye
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

    frame = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    result = detect_face_direction(frame)

    return result