import math

def get_head_direction(matrix):

    r11 = matrix[0]
    r13 = matrix[2]
    r21 = matrix[4]
    r22 = matrix[5]
    r23 = matrix[6]
    r31 = matrix[8]
    r33 = matrix[10]

    yaw = math.degrees(math.atan2(r13, r33))
    pitch = math.degrees(math.atan2(-r23, r22))
    roll = math.degrees(math.atan2(r21, r11))

    if yaw < -18:
        return "LOOKING_LEFT"

    if yaw > 18:
        return "LOOKING_RIGHT"

    if pitch > 15:
        return "LOOKING_DOWN"

    if pitch < -15:
        return "LOOKING_UP"

    return "OK"