import numpy as np
import torch
import cv2
from torchvision import transforms
from yolo.models.experimental import attempt_load
from yolo.utils.datasets import letterbox
from yolo.utils.general import non_max_suppression_kpt, scale_coords
from yolo.utils.plots import plot_one_box
from yolo.models.experimental import attempt_load
from typing import Tuple
import math
from simple_pid import PID

device = torch.device('cpu')
model = attempt_load('./yolo/best.pt', map_location=device) 
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
error_threshold = 0.05  # meters
time_threshold = 1.0  # seconds
def reset_pid_controllers():
    PID_X.reset()
    PID_Y.reset()
    PID_Z.reset()
    PID_YAW.reset()

def track_marker(frame: np.ndarray, target_pos: np.ndarray, marker_id: int) -> Tuple[int, int, int, int]:
    '''
    :param frame: current video frame
    :param target_pos: np.array([x_target, y_target, z_target]) or [x, y, z, yaw_target]
                       in marker coordinates, same units as get_drone_position
    :param marker_id: ArUco marker ID
    :return: (lr, fb, ud, yw) RC velocities for Tello
    '''

    pose = get_drone_position(frame, marker_id)
    if pose is None:
        # Marker not found → hover
        return 0, 0, 0, 0

    x, y, z, yaw = pose

    # Handle NaN (if your get_drone_position returns NaNs on failure)
    if any(map(lambda v: isinstance(v, float) and math.isnan(v), (x, y, z, yaw))):
        return 0, 0, 0, 0

    # Target position (marker frame)
    # target_pos may be 3D ([x,y,z]) or 4D ([x,y,z,yaw_target])
    tx, ty, tz = target_pos[:3]
    if target_pos.shape[0] >= 4:
        tyaw = target_pos[3]
    else:
        tyaw = 0.0  # default: face the marker
    # ------------------------------------------------------
    # 1) Update PID setpoints
    # ------------------------------------------------------
    PID_X.setpoint   = tx
    PID_Y.setpoint   = ty
    PID_Z.setpoint   = tz
    PID_YAW.setpoint = tyaw

    # ------------------------------------------------------
    # 2) Compute raw PID outputs (in "marker" coordinates)
    #    simple-pid uses: error = setpoint - measurement
    # ------------------------------------------------------
    u_x   = PID_X(x)     # control to reduce (tx - x)
    u_y   = PID_Y(y)     # control to reduce (ty - y)
    u_z   = PID_Z(z)     # control to reduce (tz - z)
    u_yaw = PID_YAW(yaw) # control to reduce (tyaw - yaw)

    # ------------------------------------------------------
    # 3) Deadzone around target (to avoid twitching)
    # ------------------------------------------------------
    # Position deadzone
    if abs(tx - x) < error_threshold:
        u_x = 0.0
    if abs(ty - y) < error_threshold:
        u_y = 0.0
    if abs(tz - z) < error_threshold:
        u_z = 0.0

    # Yaw deadzone (e.g. 3 degrees)
    yaw_error = tyaw - yaw
    if abs(yaw_error) < math.radians(3.0):
        u_yaw = 0.0

    # ------------------------------------------------------
    # 4) Map control outputs to Tello RC axes
    #
    # send_rc_control(lr, fb, ud, yaw)
    #   lr  > 0 : move right
    #   fb  > 0 : move forward
    #   ud  > 0 : move up
    #   yaw > 0 : rotate CCW
    #
    # Assuming marker coordinates:
    #   x: right  (+x → marker right)
    #   y: down   (+y → marker down)
    #   z: out of marker towards drone
    #
    # We want:
    #   x error → left/right   (lr)
    #   y error → up/down      (ud)
    #   z error → forward/back (fb)
    # ------------------------------------------------------

    lr = int(u_x)       # type: ignore # +u_x → move right to increase x
    fb = int(u_z)       # type: ignore # +u_z → move forward to decrease distance error
    ud = int(-u_y)      # type: ignore # if +y means "down", then -u_y: positive → move up
    yw = int(u_yaw)     # type: ignore # +u_yaw → CCW rotation

    return lr, fb, ud, yw



# ---- Global calibration state (lazy loaded from calibration.xml) ----
CAMERA_MTX: np.ndarray
DIST_COEFFS: np.ndarray

# Physical side length of the ArUco marker (change this!)
MARKER_LENGTH = 0.15  # e.g. 0.10 = 10 cm

def __undistort_frame(self, frame: np.ndarray) -> np.ndarray:
    if self.K is not None and self.dist is not None:
        return cv2.undistort(frame, self.K, self.dist)
    return frame
def _load_calibration(calibration_file: str = "calibration.xml") -> None:
    global CAMERA_MTX, DIST_COEFFS

    fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Cannot open calibration file: {calibration_file}")

    CAMERA_MTX = fs.getNode("K").mat()
    DIST_COEFFS = fs.getNode("dist").mat()   # <--- HERE: use "dist", not "distCoeffs"
    fs.release()

    if CAMERA_MTX is None or DIST_COEFFS is None:
        raise ValueError("Calibration file does not contain 'K' or 'dist' nodes.")
def get_drone_position(frame: np.ndarray, marker_id: int) -> Tuple[float, float, float, float]:
    '''
    Compute the drone (camera) position in the marker coordinate system.

    :param frame: current BGR video frame from the drone
    :param marker_id: ArUco marker ID to track
    :return: (x, y, z, yaw) of the drone in marker coordinates
             units are the same as MARKER_LENGTH (e.g. meters)
             yaw is in radians
             returns (nan, nan, nan, nan) if marker not found
    '''

    global CAMERA_MTX, DIST_COEFFS

    # --- Ensure calibration is loaded ---
    if CAMERA_MTX is None or DIST_COEFFS is None:
        _load_calibration()

    # --- Undistort frame ---
    undistorted = cv2.undistort(frame, CAMERA_MTX, DIST_COEFFS)

    # --- Detect ArUco marker ---
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

    # 4x4 dictionary (choose the correct one if you used a different set)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        # No markers at all
        return float('nan'), float('nan'), float('nan'), float('nan')

    ids = ids.flatten()
    if marker_id not in ids:
        # Our marker not found
        return float('nan'), float('nan'), float('nan'), float('nan')

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

    rvec = rvecs[0]
    tvec = tvecs[0].reshape(3, 1)

    # Rotation from marker frame (world) to camera frame (body)
    R_marker_to_cam, _ = cv2.Rodrigues(rvec)

    # ---- Camera position in marker frame ----
    R_cam_to_marker = R_marker_to_cam.T
    t_cam_in_marker = -R_cam_to_marker @ tvec

    x, y, z = t_cam_in_marker.flatten().tolist()

    # ---- Yaw: rotation around marker Z, measured from X axis ----
    # Using standard yaw–pitch–roll (Z-Y-X) convention
    yaw = math.atan2(R_marker_to_cam[1, 0], R_marker_to_cam[0, 0])
    # If you prefer degrees:
    yaw_deg = math.degrees(yaw)

    return float(x), float(y), float(z), float(yaw)
