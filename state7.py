# state7.py
import cv2
import numpy as np
import math
from simple_pid import PID

from func import CAMERA_MTX, DIST_COEFFS, MARKER_LENGTH

# Copy PID structure from func.py (but do NOT import track_marker)
PID_X = PID(50, 0.0, 20, setpoint=0)
PID_Y = PID(100, 0.0, 40, setpoint=0)
PID_Z = PID(50, 0.0, 20, setpoint=0)
PID_YAW = PID(30, 0.0, 12, setpoint=0)

PID_X.output_limits   = (-50, 50)
PID_Y.output_limits   = (-70, 70)
PID_Z.output_limits   = (-50, 50)
PID_YAW.output_limits = (-50, 50)

ERROR_POS  = 0.12
ERROR_YAW  = math.radians(10)


def pick_marker(corners, ids, side):
    """Pick left or right marker among multiple markers with same ID."""
    arr = []
    for i, mid in enumerate(ids.flatten()):
        if mid == 3:   # YOUR target marker id
            cx = corners[i][0][:, 0].mean()
            arr.append((cx, i))

    if not arr:
        return None

    # sort by x center → left = smallest x
    arr.sort(key=lambda x: x[0])

    if side == "left":
        return arr[0][1]
    else:
        return arr[-1][1]


def estimate_pose(corners):
    """SolvePnP for a specific marker corners."""
    half = MARKER_LENGTH / 2.0
    objp = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float32)

    imgp = corners[0].astype(np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        objp, imgp, CAMERA_MTX, DIST_COEFFS,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        return None

    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)

    R, _ = cv2.Rodrigues(rvec)
    R_cam_to_marker = R.T
    t_cam = -R_cam_to_marker @ tvec
    x, y, z = t_cam.flatten().tolist()

    yaw = math.atan2(R[1, 0], R[0, 0])

    return x, y, z, yaw, rvec, tvec


def track_marker_state7(frame, side, target_pos):
    """Full control logic like func.track_marker, but supporting L/R choice."""
    if frame is None:
        return [0, 0, 0, 0], frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect markers
    ar_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(ar_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return [0,0,0,0], frame

    idx = pick_marker(corners, ids, side)
    if idx is None:
        return [0,0,0,0], frame

    # solvePnP
    pose = estimate_pose(corners[idx])
    if pose is None:
        return [0,0,0,0], frame

    x, y, z, yaw, rvec, tvec = pose

    # target
    tx, ty, tz = target_pos[:3]
    tyaw = target_pos[3] if len(target_pos) >= 4 else 0

    PID_X.setpoint = tx
    PID_Y.setpoint = ty
    PID_Z.setpoint = tz
    PID_YAW.setpoint = tyaw

    u_x = PID_X(x)
    u_y = PID_Y(y)
    u_z = PID_Z(z)
    u_yaw = PID_YAW(yaw)

    if abs(tx - x) < ERROR_POS:  u_x = 0
    if abs(ty - y) < ERROR_POS:  u_y = 0
    if abs(tz - z) < ERROR_POS:  u_z = 0
    if abs(tyaw - yaw) < ERROR_YAW: u_yaw = 0

    # rotation marker → camera
    Rm, _ = cv2.Rodrigues(rvec)
    u_mark = np.array([u_x, u_y, u_z], dtype=float).reshape(3,1)
    u_cam  = Rm @ u_mark
    ux, uy, uz = u_cam.flatten()

    lr = int(ux)
    fb = int(uz)
    ud = int(-uy)
    yw = int(u_yaw)

    debug = frame.copy()
    cv2.putText(debug, f"state7 side={side}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
    cv2.putText(debug, f"RC: LR={lr} FB={fb} UD={ud} YW={yw}", (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

    return [lr, fb, ud, yw], debug


# ---------- STATE ENTRY POINTS ----------
TARGET_LAND = np.array([0,0.07,0.5,0], dtype=float)

import datetime

location_time_thresh = 0.5  # seconds
last_arrival_time_71 = 0
last_arrival_time_72 = 0


# ------------------ STATE 71 : use LEFT marker ------------------
def state71(frame):

    global last_arrival_time_71
    current_time = datetime.datetime.now().timestamp()

    rc, debug_frame = track_marker_state7(
        frame,
        side="left",
        target_pos=TARGET_LAND
    )
    lr, fb, ud, yw = rc

    # --- arrival tracking logic (copied from state4) ---
    if any(v != 0 for v in rc):
        last_arrival_time_71 = current_time

    next_state = "state71"
    if current_time - last_arrival_time_71 >= location_time_thresh:
        # reached
        lr = fb = ud = yw = 0
        rc = [0, 0, 0, 0]
        next_state = "state8"     # ← 左 marker 抵達後改進到右 marker 流程（你可以改成別的）

    # draw debug
    cv2.putText(
        debug_frame,
        f"state71 RC: {lr} {fb} {ud} {yw}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )

    return rc, debug_frame, next_state



# ------------------ STATE 72 : use RIGHT marker ------------------
def state72(frame):

    global last_arrival_time_72
    current_time = datetime.datetime.now().timestamp()

    rc, debug_frame = track_marker_state7(
        frame,
        side="right",
        target_pos=TARGET_LAND
    )
    lr, fb, ud, yw = rc

    # --- arrival tracking logic (same as state4) ---
    if any(v != 0 for v in rc):
        last_arrival_time_72 = current_time

    next_state = "state72"
    if current_time - last_arrival_time_72 >= location_time_thresh:
        lr = fb = ud = yw = 0
        rc = [0, 0, 0, 0]
        next_state = "state8"   # ← 右 marker 抵達後要去的下一個 state

    cv2.putText(
        debug_frame,
        f"state72 RC: {lr} {fb} {ud} {yw}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 200, 0),
        2,
    )

    return rc, debug_frame, next_state
