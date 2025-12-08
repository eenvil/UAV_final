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


PID_X = PID(50, 0.0, 20, setpoint=0)
PID_Y = PID(100, 0.0, 40, setpoint=0)
PID_Z = PID(50, 0.0, 20, setpoint=0)
PID_YAW = PID(30, 0.0, 12, setpoint=0)
PID_X.output_limits = (-50, 50)
PID_Y.output_limits = (-70, 70)
PID_Z.output_limits = (-50, 50)
PID_YAW.output_limits = (-50, 50)
error_threshold = 0.16  # meters
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

    lr = clip_rc(v_xc) # scale to cm/s
    fb = clip_rc(v_zc)
    ud = clip_rc(-v_yc)   # +y_c is down → negative to go up
    yw = clip_rc(u_yaw if u_yaw is not None else 0.0, limit=50)  # already limited by PID_YAW.output_limits

    return lr, fb, ud, yw



# ---- Global calibration state (lazy loaded from calibration.xml) ----
CAMERA_MTX: np.ndarray
DIST_COEFFS: np.ndarray
CALIBRATE_FILE = "calibration.xml"
fs = cv2.FileStorage(CALIBRATE_FILE, cv2.FILE_STORAGE_READ)
if not fs.isOpened():
    raise IOError(f"Cannot open calibration file: {CALIBRATE_FILE}")

CAMERA_MTX = fs.getNode("K").mat()
DIST_COEFFS = fs.getNode("dist").mat()   # <--- HERE: use "dist", not "distCoeffs"
fs.release()

# Physical side length of the ArUco marker (change this!)
MARKER_LENGTH = 0.15  # e.g. 0.10 = 10 cm

def get_drone_position(frame: np.ndarray,
                       marker_id: int
                       ) -> Tuple[float, float, float, float, Tuple[np.ndarray, np.ndarray]]:
    '''
    Compute the drone (camera) position in the marker coordinate system.
    Compatible with OpenCV 4.9+
    '''

    # Safety: make sure calibration is loaded
    # --- Detect ArUco marker ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4x4 dictionary (change if you used a different one)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # OpenCV 4.7+ / 4.9 Detector setup
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    corners, ids, _ = detector.detectMarkers(gray)

    # Helper for returning NaNs
    nan3 = np.array([float('nan'), float('nan'), float('nan')])
    
    if ids is None:
        return float('nan'), float('nan'), float('nan'), float('nan'), (nan3, nan3)

    ids = ids.flatten()
    if marker_id not in ids:
        return float('nan'), float('nan'), float('nan'), float('nan'), (nan3, nan3)

    # Index of the desired marker
    idx = int(np.where(ids == marker_id)[0][0])
    
    # Get the 2D corners of the specific marker we found
    # corners[idx] is shape (1, 4, 2) -> we need (4, 2) for solvePnP
    marker_corners_img = corners[idx][0]

    # --- Estimate pose (cv2.solvePnP instead of estimatePoseSingleMarkers) ---
    
    # Define the 3D coordinates of the marker corners in the marker's own frame.
    # The center is (0,0,0). Z points out of the marker.
    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left (matches detectMarkers output)
    half_size = MARKER_LENGTH / 2.0
    marker_obj_points = np.array([
        [-half_size,  half_size, 0],  # Top-Left
        [ half_size,  half_size, 0],  # Top-Right
        [ half_size, -half_size, 0],  # Bottom-Right
        [-half_size, -half_size, 0]   # Bottom-Left
    ], dtype=np.float32)

    # Solve PnP to get rotation (rvec) and translation (tvec) of marker relative to camera
    success, rvec, tvec = cv2.solvePnP(
        marker_obj_points, 
        marker_corners_img, 
        CAMERA_MTX, 
        DIST_COEFFS, 
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if not success:
        return float('nan'), float('nan'), float('nan'), float('nan'), (nan3, nan3)

    # Ensure vectors are (3, 1) shape for matrix math
    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)

    # --- Coordinate Transformation ---
    
    # Rotation from marker frame (world) to camera frame (body)
    R_marker_to_cam, _ = cv2.Rodrigues(rvec)

    # Camera position in marker frame:
    # X_marker(cam) = -R_mc^T * t_mc
    R_cam_to_marker = R_marker_to_cam.T
    t_cam_in_marker = -R_cam_to_marker @ tvec

    x, y, z = t_cam_in_marker.flatten().tolist()

    # ---- Yaw Calculation ----
    # Rotation of the camera relative to the marker
    # We extract yaw from the rotation matrix
    yaw = math.atan2(R_marker_to_cam[1, 0], R_marker_to_cam[0, 0])

    return float(x), float(y), float(z), float(yaw), (rvec, tvec)