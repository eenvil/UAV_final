from typing import Tuple
import cv2
import numpy as np
import math
from simple_pid import PID


def detect_doll(frame) -> int:
    '''
    Docstring for detect_doll
    
    :param frame: the current video frame
    :return: 0 no doll detected, 1 kana, 2 melody
    :rtype: int
    '''
    pass
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

def track_marker(frame,target_pos,marker_id) -> Tuple[int, int, int, int]:
    '''
    Docstring for track_marker
    
    :param frame: the current video frame
    :return: (lr, fb, ud, yw) velocities to track the marker
    :rtype: Tuple[int, int, int, int]
    '''
    x,y,z,yaw = get_drone_position(frame, marker_id)



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
