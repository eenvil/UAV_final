import numpy as np
import torch
import cv2
from torchvision import transforms
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords
from utils.plots import plot_one_box
from models.experimental import attempt_load
from typing import Tuple
import math
from simple_pid import PID

device = torch.device('cpu')
model = attempt_load('./best.pt', map_location=device) 
def detect_doll(frame) -> int:
    '''
    Docstring for detect_doll
    
    :param frame: the current video frame
    :return: 0 no doll detected, 1 carna, 2 melody
    :rtype: int
    '''
    # Load model before calling this function, device default to cpu
    global model, device
    if frame is None:
        return 0
    # Pre-process image
    img0 = letterbox(frame, (640, 640), stride=64, auto=True)[0]
    img = img0.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)        
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    # Inference
    with torch.no_grad():
        output = model(img)[0]

    pred = non_max_suppression_kpt(output, 0.25, 0.65)[0]
    if pred is None or len(pred) == 0:
        return 0
    names = getattr(model, 'names', model.module.names if hasattr(model, 'module') else [])
    for *xyxy, conf, cls in pred:
        try:
            class_name = names[int(cls)].lower()
            if "carna" in class_name:
                return 1
            elif "melody" in class_name:
                return 2
        except (IndexError, AttributeError):
            continue

    return 0


PID_X = PID(0.5, 0.0, 0.1, setpoint=0)
PID_Y = PID(0.5, 0.0, 0.1, setpoint=0)
PID_Z = PID(0.5, 0.0, 0.1, setpoint=0)
PID_YAW = PID(0.5, 0.0, 0.1, setpoint=0)
PID_X.output_limits = (-50, 50)
PID_Y.output_limits = (-50, 50)
PID_Z.output_limits = (-50, 50)
PID_YAW.output_limits = (-50, 50)
error_threshold = 0.1  # meters
yaw_error_threshold = math.radians(10.0)  # radians
def reset_pid_controllers():
    PID_X.reset()
    PID_Y.reset()
    PID_Z.reset()
    PID_YAW.reset()

def track_marker(frame: np.ndarray,
                 target_pos: np.ndarray,
                 marker_id: int) -> Tuple[int, int, int, int]:
    """
    Compute RC commands (lr, fb, ud, yw) for Tello using:
      - position PIDs in the MARKER frame
      - 3D rotation (rvec) only to map control → drone frame.

    :param frame: calibrated BGR frame
    :param target_pos: np.array([x, y, z]) or [x, y, z, yaw_target] in marker frame (meters, radians)
    :param marker_id: ArUco marker ID
    :return: (lr, fb, ud, yaw) RC velocities
    """

    pose = get_drone_position(frame, marker_id)
    if pose is None:
        # Marker not found → hover
        return 0, 0, 0, 0

    x_m, y_m, z_m, yaw, (rvec, tvec) = pose

    # Handle NaN
    if any(isinstance(v, float) and math.isnan(v) for v in (x_m, y_m, z_m, yaw)):
        return 0, 0, 0, 0

    # ------------------------------------------------------
    # 1) Parse target in marker frame
    # ------------------------------------------------------
    tx, ty, tz = target_pos[:3]
    if target_pos.shape[0] >= 4:
        tyaw = float(target_pos[3])
    else:
        # default: face marker with yaw = 0 in marker frame
        tyaw = 0.0

    # ------------------------------------------------------
    # 2) Update PID setpoints in MARKER frame
    # ------------------------------------------------------
    PID_X.setpoint   = tx
    PID_Y.setpoint   = ty
    PID_Z.setpoint   = tz
    PID_YAW.setpoint = tyaw

    # ------------------------------------------------------
    # 3) Compute raw PID outputs in MARKER frame
    #     simple-pid: error = setpoint - measurement
    # ------------------------------------------------------
    u_x   = PID_X(x_m)      # control in marker X
    u_y   = PID_Y(y_m)      # control in marker Y
    u_z   = PID_Z(z_m)      # control in marker Z
    u_yaw = PID_YAW(yaw)    # yaw control (marker yaw)

    # ------------------------------------------------------
    # 4) Deadzone in MARKER frame to avoid twitching
    # ------------------------------------------------------
    # Per-axis deadzone
    if abs(tx - x_m) < error_threshold:
        u_x = 0.0
    if abs(ty - y_m) < error_threshold:
        u_y = 0.0
    if abs(tz - z_m) < error_threshold:
        u_z = 0.0

    # Overall distance deadzone (more aggressive smoothing)
    # dist = math.sqrt((tx - x_m)**2 + (ty - y_m)**2 + (tz - z_m)**2)
    # if dist < error_threshold:
    #     u_x = u_y = u_z = 0.0

    # Yaw deadzone (e.g. 3 degrees)
    yaw_error = tyaw - yaw
    # (optional) wrap to [-pi, pi] if your yaw can go beyond that
    if abs(yaw_error) < yaw_error_threshold:
        u_yaw = 0.0

    # ------------------------------------------------------
    # 5) Rotate control vector from MARKER → CAMERA/DRONE frame
    #
    # rvec, tvec are marker→camera pose from ArUco:
    #   X_cam = R_mc * X_marker + t_mc
    #
    # Here, u_m is a desired velocity in marker coords.
    # To express it in camera coords:
    #   u_c = R_mc * u_m
    # ------------------------------------------------------
    R_mc, _ = cv2.Rodrigues(rvec)  # 3x3 rotation matrix
    u_m = np.array([[u_x],
                    [u_y],
                    [u_z]], dtype=float)
    u_c = R_mc @ u_m

    v_xc = float(u_c[0, 0])  # right
    v_yc = float(u_c[1, 0])  # down
    v_zc = float(u_c[2, 0])  # forward

    # ------------------------------------------------------
    # 6) Map to Tello RC axes
    #
    # send_rc_control(lr, fb, ud, yaw):
    #   lr  > 0 : move right  (≈ +x_c)
    #   fb  > 0 : move forward(≈ +z_c)
    #   ud  > 0 : move up     (opposite of +y_c)
    # ------------------------------------------------------
    def clip_rc(v: float, limit: int = 50) -> int:
        return int(max(-limit, min(limit, round(v))))

    lr = clip_rc(v_xc*100) # scale to cm/s
    fb = clip_rc(v_zc*100)
    ud = clip_rc(-v_yc*100)   # +y_c is down → negative to go up
    yw = clip_rc(u_yaw*100 if u_yaw is not None else 0.0, limit=50)  # already limited by PID_YAW.output_limits

    return lr, fb, ud, yw



# ---- Global calibration state (lazy loaded from calibration.xml) ----
CAMERA_MTX: np.ndarray
DIST_COEFFS: np.ndarray

# Physical side length of the ArUco marker (change this!)
MARKER_LENGTH = 0.15  # e.g. 0.10 = 10 cm

def get_drone_position(frame: np.ndarray,
                       marker_id: int
                       ) -> Tuple[float, float, float, float, Tuple[np.ndarray, np.ndarray]]:
    '''
    Compute the drone (camera) position in the marker coordinate system.

    :param frame: current BGR video frame from the drone
    :param marker_id: ArUco marker ID to track
    :return: (x, y, z, yaw) of the drone in marker coordinates
             units are the same as MARKER_LENGTH (e.g. meters)
             yaw is in radians
             rvec and tvec are the rotation and translation vectors from
             the marker to the camera (drone) frame.
             returns (nan, nan, nan, nan, (nan, nan)) if marker not found
    '''

    # Safety: make sure calibration is loaded
    # (remove this if you initialize CAMERA_MTX/DIST_COEFFS right here)
    if CAMERA_MTX is None or DIST_COEFFS is None:
        raise RuntimeError("CAMERA_MTX and DIST_COEFFS must be initialized before calling get_drone_position")

    # --- Detect ArUco marker ---
    # If your frame is already undistorted, this is fine.
    # If not, you can optionally undistort here using CAMERA_MTX & DIST_COEFFS.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4x4 dictionary (change if you used a different one)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # OpenCV 4.7+:
    try:
        parameters = cv2.aruco.DetectorParameters()
    except AttributeError:
        # Older OpenCV:
        parameters = cv2.aruco.DetectorParameters_create() # type: ignore

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        # No markers at all
        nan3 = np.array([float('nan'), float('nan'), float('nan')])
        return float('nan'), float('nan'), float('nan'), float('nan'), (nan3, nan3)

    ids = ids.flatten()
    if marker_id not in ids:
        # Our marker not found
        nan3 = np.array([float('nan'), float('nan'), float('nan')])
        return float('nan'), float('nan'), float('nan'), float('nan'), (nan3, nan3)

    # Index of the desired marker
    idx = int(np.where(ids == marker_id)[0][0])
    marker_corners = [corners[idx]]  # shape (1, 4, 2)

    # --- Estimate pose of the marker (marker -> camera) ---
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        marker_corners,
        MARKER_LENGTH,
        CAMERA_MTX,
        DIST_COEFFS
    )

    # rvecs: (1, 1, 3) or (1, 3); tvecs: (1, 1, 3) or (1, 3)
    rvec = rvecs[0].reshape(3, 1)
    tvec = tvecs[0].reshape(3, 1)

    # Rotation from marker frame (world) to camera frame (body)
    R_marker_to_cam, _ = cv2.Rodrigues(rvec)

    # ---- Camera position in marker frame ----
    # If X_cam = R_mc * X_marker + t_mc,
    # then camera origin in marker frame is:
    #   X_marker(cam) = -R_mc^T * t_mc
    R_cam_to_marker = R_marker_to_cam.T
    t_cam_in_marker = -R_cam_to_marker @ tvec

    x, y, z = t_cam_in_marker.flatten().tolist()

    # ---- Yaw: rotation around marker Z, from marker X toward marker Y ----
    # Standard yaw from rotation matrix (Z-Y-X convention)
    yaw = math.atan2(R_marker_to_cam[1, 0], R_marker_to_cam[0, 0])

    return float(x), float(y), float(z), float(yaw), (rvec, tvec)