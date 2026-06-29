import cv2
import numpy as np
import math

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose
    (0.0, -63.6, -12.5),      # Chin
    (-43.3, 32.7, -26.0),     # Left Eye
    (43.3, 32.7, -26.0),      # Right Eye
    (-28.9, -28.9, -24.1),    # Left Mouth
    (28.9, -28.9, -24.1)      # Right Mouth
], dtype=np.float64)

def get_head_pose(image_points, width, height):

    focal_length = width

    center = (width / 2, height / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    pose_matrix = cv2.hconcat(
        (rotation_matrix, translation_vector)
    )

    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(
        pose_matrix
    )

    pitch = float(eulerAngles[0])
    yaw = float(eulerAngles[1])
    roll = float(eulerAngles[2])

    return pitch, yaw, roll

    def get_direction(pitch, yaw):

    if yaw < -20:
        return "LOOKING_LEFT"

    if yaw > 20:
        return "LOOKING_RIGHT"

    if pitch < -18:
        return "LOOKING_UP"

    if pitch > 18:
        return "LOOKING_DOWN"

    return "OK"