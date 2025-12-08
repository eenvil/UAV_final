# state7.py
import cv2
import numpy as np
import math
from simple_pid import PID
import datetime

from func import CAMERA_MTX, DIST_COEFFS, MARKER_LENGTH

# ---------------- PID CONFIG ----------------
# (tuned separately from func.py if needed)
PID_X = PID(50, 0.0, 20, setpoint=0)
PID_Y = PID(100, 0.0, 40, setpoint=0)
PID_Z = PID(50, 0.0, 20, setpoint=0)
PID_YAW = PID(30, 0.0, 12, setpoint=0)

PID_X.output_limits   = (-50, 50)
PID_Y.output_limits   = (-70, 70)
PID_Z.output_limits   = (-50, 50)
PID_YAW.output_limits = (-50, 50)

ERROR_POS  = 0.12                     # meters
ERROR_YAW  = math.radians(10)         # radians

# target pose in marker frame (same for left/right)
TARGET_LAND = np.array([0.0, 0.07, 0.5, 0.0], dtype=float)

# arrival logic
location_time_thresh = 0.5  # seconds
last_arrival_time_71 = 0.0
last_arrival_time_72 = 0.0

# phase-2 locking (have we already seen two markers once?)
locked_71 = False
locked_72 = False

# yaw-search parameters (when marker not found)
SEARCH_YAW_SPEED = 25          # RC yaw speed while searching
SEARCH_FLIP_PERIOD = 1.5       # how often to flip yaw direction (seconds)
search_dir_71 = 1
search_dir_72 = -1             # opposite initial direction for variety
last_search_flip_71 = 0.0
last_search_flip_72 = 0.0


# ---------------- HELPER: PICK MARKER INDICES ----------------
def pick_marker_indices(corners, ids):
    """
    Return a sorted list of (cx, idx) for all markers with id == 3.
    cx is the image x-center of the marker, idx is index in corners/ids.
    """
    arr = []
    if ids is None:
        return arr

    for i, mid in enumerate(ids.flatten()):
        if mid == 3:   # your target ArUco ID
            cx = corners[i][0][:, 0].mean()
            arr.append((cx, i))

    # sort left→right by image x-center
    arr.sort(key=lambda x: x[0])
    return arr  # list of (cx, idx)


# ---------------- HELPER: ESTIMATE POSE ----------------
def estimate_pose(corners):
    """SolvePnP for a specific marker corners -> pose of camera in marker frame."""
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

    # R: marker -> camera
    R, _ = cv2.Rodrigues(rvec)
    R_cam_to_marker = R.T

    # camera origin in marker frame: X_m = -R^T * t
    t_cam = -R_cam_to_marker @ tvec
    x, y, z = t_cam.flatten().tolist()

    # yaw of camera in marker frame
    yaw = math.atan2(R[1, 0], R[0, 0])

    return x, y, z, yaw, rvec, tvec


# ---------------- TRACKING CORE (FOR STATE 7) ----------------
def track_marker_state7(frame, side, target_pos):
    """
    Like func.track_marker, but:
      - chooses left/right marker when two are visible,
      - returns whether we saw ANY marker, and whether we saw TWO markers.

    Returns:
      rc   : [lr, fb, ud, yw]
      debug_frame : visualization frame
      found      : bool, True if we have at least one marker pose
      seen_two   : bool, True if we currently see >=2 markers with id=3
    """
    if frame is None:
        return [0, 0, 0, 0], frame, False, False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect markers
    ar_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(ar_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)

    debug = frame.copy()

    # gather all candidate markers with id=3
    arr = pick_marker_indices(corners, ids)
    seen_two = len(arr) >= 2

    if len(arr) == 0:
        # no markers with id=3
        cv2.putText(debug, "FOUND=False TWO=False", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        return [0, 0, 0, 0], debug, False, seen_two

    # choose which marker to track, depending on side
    # left  -> smallest cx
    # right -> largest cx
    if side == "left":
        _, idx = arr[0]
    else:
        _, idx = arr[-1]

    pose = estimate_pose(corners[idx])
    if pose is None:
        cv2.putText(debug, "FOUND=False (solvePnP fail)", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return [0, 0, 0, 0], debug, False, seen_two

    x, y, z, yaw, rvec, tvec = pose

    # target in marker frame
    tx, ty, tz = target_pos[:3]
    tyaw = target_pos[3] if len(target_pos) >= 4 else 0.0

    PID_X.setpoint   = tx
    PID_Y.setpoint   = ty
    PID_Z.setpoint   = tz
    PID_YAW.setpoint = tyaw

    u_x   = PID_X(x)
    u_y   = PID_Y(y)
    u_z   = PID_Z(z)
    u_yaw = PID_YAW(yaw)

    # deadzones in marker frame
    if abs(tx - x) < ERROR_POS:
        u_x = 0.0
    if abs(ty - y) < ERROR_POS:
        u_y = 0.0
    if abs(tz - z) < ERROR_POS:
        u_z = 0.0
    if abs(tyaw - yaw) < ERROR_YAW:
        u_yaw = 0.0

    # rotate control vector marker -> camera
    Rm, _ = cv2.Rodrigues(rvec)
    u_mark = np.array([[u_x], [u_y], [u_z]], dtype=float)
    u_cam  = Rm @ u_mark
    ux, uy, uz = u_cam.flatten()

    lr = int(ux)
    fb = int(uz)
    ud = int(-uy)     # +y_cam is down → negative to go up
    yw = int(u_yaw)

    cv2.putText(debug, f"state7 side={side}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(debug, f"RC: LR={lr} FB={fb} UD={ud} YW={yw}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(debug, f"FOUND=True TWO={seen_two}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    return [lr, fb, ud, yw], debug, True, seen_two


# ------------------ STATE 71 : use LEFT marker ------------------
def state71(frame):
    global last_arrival_time_71, locked_71
    global search_dir_71, last_search_flip_71

    current_time = datetime.datetime.now().timestamp()

    rc, debug_frame, found, seen_two = track_marker_state7(
        frame,
        side="left",
        target_pos=TARGET_LAND,
    )
    lr, fb, ud, yw = rc

    # --------- PHASE 1: not locked yet -> require seeing two markers ----------
    if not locked_71:
        if found and seen_two:
            # we have seen both markers once -> lock into phase 2
            locked_71 = True
        else:
            # not yet: yaw-search to try to bring both markers into view
            if current_time - last_search_flip_71 > SEARCH_FLIP_PERIOD:
                search_dir_71 *= -1
                last_search_flip_71 = current_time

            yw = SEARCH_YAW_SPEED * search_dir_71
            lr = fb = ud = 0
            rc = [lr, fb, ud, yw]
            found = False  # do not allow arrival while still acquiring

    # --------- PHASE 2: locked -> track even if only one marker visible -------
    if locked_71:
        if not found:
            # lost marker -> yaw-search to reacquire (but stay in phase2)
            if current_time - last_search_flip_71 > SEARCH_FLIP_PERIOD:
                search_dir_71 *= -1
                last_search_flip_71 = current_time

            yw = SEARCH_YAW_SPEED * search_dir_71
            lr = fb = ud = 0
            rc = [lr, fb, ud, yw]

    # --------- ARRIVAL LOGIC (PHASE 3) ----------
    # Reset timer if we are moving OR marker not found OR not locked yet
    if any(v != 0 for v in rc) or not found or not locked_71:
        last_arrival_time_71 = current_time

    next_state = "state71"
    # Only declare arrival if:
    #  - we have passed Phase 1 (locked_71),
    #  - marker is found,
    #  - we've been still long enough.
    if locked_71 and found and current_time - last_arrival_time_71 >= location_time_thresh:
        lr = fb = ud = yw = 0
        rc = [0, 0, 0, 0]
        next_state = "state8"

    # ---------- DEBUG OVERLAYS FOR PHASE VISUALIZATION ----------
    if not locked_71:
        phase_text = "PHASE 1: ACQUIRE (need two markers)"
        color = (0, 165, 255)   # orange
    else:
        if not found:
            phase_text = "PHASE 2: TRACK (lost marker -> searching)"
            color = (0, 255, 255)   # yellow
        else:
            # Check if we're effectively in Phase 3 (close to arrival)
            if current_time - last_arrival_time_71 >= location_time_thresh:
                phase_text = "PHASE 3: ARRIVAL -> STATE 8"
                color = (255, 0, 0)   # red
            else:
                phase_text = "PHASE 2: TRACK"
                color = (0, 255, 0)   # green

    cv2.putText(debug_frame, phase_text, (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    marker_status = f"FOUND={found}  SEEN_TWO={seen_two}  LOCKED={locked_71}"
    cv2.putText(debug_frame, marker_status, (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(debug_frame, "Side = LEFT", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

    cv2.putText(
        debug_frame,
        f"state71 RC: {lr} {fb} {ud} {yw}",
        (10, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    return rc, debug_frame, next_state


# ------------------ STATE 72 : use RIGHT marker ------------------
def state72(frame):
    global last_arrival_time_72, locked_72
    global search_dir_72, last_search_flip_72

    current_time = datetime.datetime.now().timestamp()

    rc, debug_frame, found, seen_two = track_marker_state7(
        frame,
        side="right",
        target_pos=TARGET_LAND,
    )
    lr, fb, ud, yw = rc

    # --------- PHASE 1: not locked yet -> require seeing two markers ----------
    if not locked_72:
        if found and seen_two:
            locked_72 = True
        else:
            if current_time - last_search_flip_72 > SEARCH_FLIP_PERIOD:
                search_dir_72 *= -1
                last_search_flip_72 = current_time

            yw = SEARCH_YAW_SPEED * search_dir_72
            lr = fb = ud = 0
            rc = [lr, fb, ud, yw]
            found = False

    # --------- PHASE 2: locked -> track even if only one marker visible -------
    if locked_72:
        if not found:
            if current_time - last_search_flip_72 > SEARCH_FLIP_PERIOD:
                search_dir_72 *= -1
                last_search_flip_72 = current_time

            yw = SEARCH_YAW_SPEED * search_dir_72
            lr = fb = ud = 0
            rc = [lr, fb, ud, yw]

    # --------- ARRIVAL LOGIC (PHASE 3) ----------
    if any(v != 0 for v in rc) or not found or not locked_72:
        last_arrival_time_72 = current_time

    next_state = "state72"
    if locked_72 and found and current_time - last_arrival_time_72 >= location_time_thresh:
        lr = fb = ud = yw = 0
        rc = [0, 0, 0, 0]
        next_state = "state8"

    # ---------- DEBUG OVERLAYS FOR PHASE VISUALIZATION ----------
    if not locked_72:
        phase_text = "PHASE 1: ACQUIRE (need two markers)"
        color = (0, 165, 255)   # orange
    else:
        if not found:
            phase_text = "PHASE 2: TRACK (lost marker -> searching)"
            color = (0, 255, 255)   # yellow
        else:
            if current_time - last_arrival_time_72 >= location_time_thresh:
                phase_text = "PHASE 3: ARRIVAL -> STATE 8"
                color = (255, 0, 0)   # red
            else:
                phase_text = "PHASE 2: TRACK"
                color = (0, 255, 0)   # green

    cv2.putText(debug_frame, phase_text, (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    marker_status = f"FOUND={found}  SEEN_TWO={seen_two}  LOCKED={locked_72}"
    cv2.putText(debug_frame, marker_status, (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(debug_frame, "Side = RIGHT", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

    cv2.putText(
        debug_frame,
        f"state72 RC: {lr} {fb} {ud} {yw}",
        (10, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 200, 0),
        2,
    )

    return rc, debug_frame, next_state
