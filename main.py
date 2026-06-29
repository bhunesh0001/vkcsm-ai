from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np

from detector import detect_face

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
        "status": "Running"
    }


@app.post("/detect")
def detect(dto: FaceDetectDTO):

    img = dto.image

    if "," in img:
        img = img.split(",")[1]

    img_bytes = base64.b64decode(img)

    npimg = np.frombuffer(
        img_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        npimg,
        cv2.IMREAD_COLOR
    )

    result = detect_face(frame)

    return result