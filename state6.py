# turn and detect the doll to decide next state
import func
import numpy as np
import cv2
import datetime
doll_detected = 0  # 0: none, 1: carna, 2: melody
doll_detect_thresh = 0.5  # detection threshold (seconds)
last_doll_detect_time = 0
last_doll_detected = 0
last_detect_time = 0
location_time_thresh = 0.5  # seconds
def state6(frame):
    global doll_detected, last_detect_time, last_doll_detected, last_doll_detect_time
    next_state = "state6"
    current_time = datetime.datetime.now().timestamp()
    if current_time - last_detect_time >= 0.25 and doll_detected == 0:
        result = func.detect_doll(frame)
        if result != last_doll_detected:
            last_doll_detect_time = current_time
            last_doll_detected = result
        elif result == last_doll_detected and result != 0:
            if current_time - last_doll_detect_time >= doll_detect_thresh:
                doll_detected = result
        else:
            doll_detected = 0
        last_detect_time = current_time
    lr = fb = ud = yw = 0
    if doll_detected != 0:
        next_state = "state71" if doll_detected == 1 else "state72"
    debug_frame = frame.copy()
    cv2.putText(
        debug_frame,
        f"DOLL DETECTED: {doll_detected}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        debug_frame,
        f"LR: {lr} FB: {fb} UD: {ud} YW: {yw}",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    return [lr, fb, ud, yw], debug_frame, next_state