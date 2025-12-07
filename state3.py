from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np
from cv2.typing import MatLike

# ----------------- CONFIG -----------------
LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([180, 255, 100])
THICKNESS_THRESHOLD_PX = 8.0
MIN_OCCUPANCY_FRACTION = 0.01      # for 3x3 grid

# Gains for converting pixel error -> velocity [-100, 100]
K_LR = 100.0      # left/right gain
K_UD = 100.0      # up/down gain
K_FB = 2.0        # forward/backward gain based on width error

# Size of ROI around image center for width estimation
WIDTH_ROI_HALF = 40
# ------------------------------------------


def _clamp(v: float, lo: int = -100, hi: int = 100) -> int:
    return int(max(lo, min(hi, v)))


def _detect_intersection(occ: np.ndarray) -> Tuple[bool, str]:
    """Simple intersection detection using 3x3 occupancy grid."""
    c = occ[1, 1]
    if not c:
        return False, "center not on road"

    up    = occ[0, 1]
    down  = occ[2, 1]
    left  = occ[1, 0]
    right = occ[1, 2]

    neighbors = sum([up, down, left, right])

    if up and c and down and not (left or right):
        return False, "straight vertical"

    if left and c and right and not (up or down):
        return False, "straight horizontal"

    if neighbors >= 3:
        return True, "T or cross intersection"

    if (up and left) or (up and right) or (down and left) or (down and right):
        return True, "corner / L intersection"

    return False, "no intersection"


def _compute_target_from_direction(
    direction: str,
    occ: np.ndarray,
    centroids: list[list[tuple[float, float] | None]],
    w: int,
    h: int,
) -> tuple[float, float, float, float, tuple[float, float]]:
    """
    Given desired direction ('up', 'down', 'left', 'right'),
    occupancy grid and per-cell centroids, find target point.

    Returns: dx, dy, tx, ty
    - dx, dy: pixel error from image center
    - tx, ty: target coordinate
    """
    dir_to_cell = {
        "up":    (0, 1),
        "down":  (2, 1),
        "left":  (1, 0),
        "right": (1, 2),
    }

    cx_img, cy_img = w / 2.0, h / 2.0

    if direction not in dir_to_cell:
        # No movement if invalid direction
        return 0.0, 0.0, cx_img, cy_img

    target_i, target_j = dir_to_cell[direction]

    # First try: exact desired neighbor cell
    tx = ty = None
    if occ[target_i, target_j] and centroids[target_i][target_j] is not None:
        tx, ty = centroids[target_i][target_j]
    else:
        # Fallback: closest occupied cell to desired cell
        best_dist = None
        best_pt: tuple[float, float] | None = None
        for i in range(3):
            for j in range(3):
                if not occ[i, j] or centroids[i][j] is None:
                    continue
                di = i - target_i
                dj = j - target_j
                dist = di * di + dj * dj
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_pt = centroids[i][j]
        if best_pt is not None:
            tx, ty = best_pt

    # If still nothing, stay at center
    if tx is None or ty is None:
        tx, ty = cx_img, cy_img

    dx = tx - cx_img
    dy = ty - cy_img
    return dx, dy, tx, ty, (tx, ty)


def line_tracking_state(
    frame: MatLike,
    turn_direction: str,
    line_width: float,
) -> tuple[tuple[int, int, int, int], bool, MatLike]:
    """
    Process one frame and return:
      - (left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)
      - is_intersection (bool)
      - debug_frame (with overlay)

    left_right_velocity:     -100..100 (left/right)
    forward_backward_velocity:-100..100 (forward/back, used to control width)
    up_down_velocity:        -100..100 (up/down)
    yaw_velocity:            -100..100 (here fixed to 0)
    """

    # Optional resize – remove if you want full resolution
    # frame = cv2.resize(frame, (960, 540))
    h, w = frame.shape[:2]

    # --- 1) Detect black road in HSV ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_black = cv2.inRange(hsv, LOWER_BLACK, UPPER_BLACK)

    # --- 2) Clean up noise ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # --- 3) Distance transform & thickness filtering ---
    dist = cv2.distanceTransform(mask_clean, distanceType=cv2.DIST_L2, maskSize=5)
    _, thick_mask = cv2.threshold(dist, THICKNESS_THRESHOLD_PX, 255, cv2.THRESH_BINARY)
    thick_mask = thick_mask.astype(np.uint8)

    # --- 4) Keep only largest connected component (main road) ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        thick_mask, connectivity=8
    )

    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + int(np.argmax(areas))
        main_mask = np.zeros_like(thick_mask)
        main_mask[labels == largest_idx] = 255
    else:
        main_mask = thick_mask

    # --- 5) Estimate road width near image center ---
    cx_img, cy_img = w // 2, h // 2
    x0w = max(cx_img - WIDTH_ROI_HALF, 0)
    x1w = min(cx_img + WIDTH_ROI_HALF, w)
    y0w = max(cy_img - WIDTH_ROI_HALF, 0)
    y1w = min(cy_img + WIDTH_ROI_HALF, h)

    dist_roi = dist[y0w:y1w, x0w:x1w]
    mask_roi = main_mask[y0w:y1w, x0w:x1w] > 0

    width_px = 0.0
    if np.any(mask_roi):
        local_radii = dist_roi[mask_roi]
        local_radius = float(np.median(local_radii))
        width_px = 2.0 * local_radius

    # --- 6) Build 3x3 occupancy grid + per-cell centroids ---
    occ = np.zeros((3, 3), dtype=bool)
    cell_centroids: list[list[tuple[float, float] | None]] = [
        [None for _ in range(3)] for _ in range(3)
    ]
    cell_h = h // 3
    cell_w = w // 3

    for i in range(3):       # rows (y)
        for j in range(3):   # cols (x)
            y0 = i * cell_h
            y1 = h if i == 2 else (i + 1) * cell_h
            x0 = j * cell_w
            x1 = w if j == 2 else (j + 1) * cell_w

            cell = main_mask[y0:y1, x0:x1]
            cell_area = cell.size
            road_pixels = cv2.countNonZero(cell)

            if road_pixels > cell_area * MIN_OCCUPANCY_FRACTION:
                occ[i, j] = True
                ys, xs = np.where(cell > 0)
                if len(xs) > 0:
                    cx = x0 + float(xs.mean())
                    cy = y0 + float(ys.mean())
                    cell_centroids[i][j] = (cx, cy)

    # --- 7) Intersection detection ---
    is_intersection, desc = _detect_intersection(occ)

    # --- 8) Compute target and dx, dy for the requested direction ---
    dx_pixels, dy_pixels, tx, ty, _ = _compute_target_from_direction(
        turn_direction, occ, cell_centroids, w, h
    )

    # --- 9) Convert pixel errors to velocities [-100, 100] ---

    # Normalize dx, dy by half image size so 1.0 = edge
    norm_dx = dx_pixels / (w / 2.0)
    norm_dy = dy_pixels / (h / 2.0)

    # Left/right: positive -> move right, negative -> move left
    left_right_velocity = _clamp(K_LR * norm_dx)

    # Up/down: positive -> move down, negative -> move up
    up_down_velocity = _clamp(K_UD * norm_dy)

    # Forward/back: use width error (desired - measured)
    # If line appears too thin (width_px < line_width) -> move forward (positive)
    width_error = line_width - width_px
    forward_backward_velocity = _clamp(K_FB * width_error)

    # Yaw is fixed to 0 as requested
    yaw_velocity = 0

    # --- 10) Build debug frame ---
    vis = frame.copy()

    # Draw 3x3 grid
    for i in range(3):
        for j in range(3):
            y0 = i * cell_h
            y1 = h if i == 2 else (i + 1) * cell_h
            x0 = j * cell_w
            x1 = w if j == 2 else (j + 1) * cell_w

            color = (0, 255, 0) if occ[i, j] else (0, 0, 255)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)

    # Center and target
    cv2.circle(vis, (cx_img, cy_img), 5, (255, 255, 0), -1)
    cv2.circle(vis, (int(tx), int(ty)), 5, (0, 255, 255), -1)
    cv2.arrowedLine(vis, (cx_img, cy_img), (int(tx), int(ty)),
                    (0, 255, 255), 2, tipLength=0.2)
    
    # Width bar at center
    if width_px > 0:
        half_w = int(width_px / 2)
        y_bar = cy_img
        x_start = max(cx_img - half_w, 0)
        x_end = min(cx_img + half_w, w - 1)
        cv2.line(vis, (x_start, y_bar), (x_end, y_bar), (0, 255, 255), 2)

    # Text overlays
    cv2.putText(vis,
                f"Dir: {turn_direction} | LR={left_right_velocity} FB={forward_backward_velocity} UD={up_down_velocity} Yaw={yaw_velocity}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis,
                f"Intersection: {is_intersection} ({desc})",
                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis,
                f"Width px: {width_px:.1f} | target: {line_width:.1f}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)

    velocities = (left_right_velocity,
                  forward_backward_velocity,
                  -up_down_velocity, # cv2 y-axis is inverted
                  yaw_velocity)
    
    return velocities, is_intersection, vis
#line tracking until detec until detecting aruco marker 2
def state31(frame):
    # todo: implement state3 behavior
    raise NotImplementedError("state3 behavior not implemented yet")
    return [lr, fb, ud, yw], debug_frame, next_state
def state32(frame):
    raise NotImplementedError("state3.2 behavior not implemented yet")
    return [lr, fb, ud, yw], debug_frame, next_state

if __name__ == "__main__":
    # Direction plan: switch to next after each intersection
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.avi", fourcc, 30.0, (720, 1280))
    direction_sequence = ["left", "up", "left", "up", "left", "down", "left"]
    dir_index = 0
    turn_direction = direction_sequence[dir_index]
    VIDEO_PATH = 'input.mp4'  # Path to your test video
    # Desired line width in pixels (tune this to your video)
    target_width_px = 40.0

    # To avoid switching multiple times on the same intersection,
    # use a small cooldown after each switch.
    intersection_cooldown_frames = 30
    cooldown = 0
    initial_cooldown = 50  # frames to wait at start
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: cannot open {VIDEO_PATH}")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (lr, fb, ud, yaw), is_intersection, dbg = line_tracking_state(
            frame, turn_direction, target_width_px
        )

        # Here you would send (lr, fb, ud, yaw) to your drone API.
        # For now we just print them occasionally:
        print(f"Dir={turn_direction:5s} | LR={lr:4d} FB={fb:4d} UD={ud:4d} Yaw={yaw:4d} | Intersection={is_intersection}")

        # Handle direction switching when intersection is detected
        if cooldown > 0 or initial_cooldown > 0:
            if cooldown > 0:
                cooldown -= 1
            if initial_cooldown > 0:
                initial_cooldown -= 1
        else:
            if is_intersection and dir_index < len(direction_sequence) - 1:
                dir_index += 1
                turn_direction = direction_sequence[dir_index]
                cooldown = intersection_cooldown_frames
                print(f"--> Switched to next direction: {turn_direction}")

        cv2.imshow("Line tracking debug", dbg)
        out.write(dbg)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()

