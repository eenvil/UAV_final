from func import get_drone_position
from typing import Tuple

# using aruco to fly to the fixed position
def state4(frame):
    # todo: implement state1 behavior
    raise NotImplementedError("state4 behavior not implemented yet")
    return [lr, fb, ud, yw], debug_frame, next_state

if __name__ == "__main__":
    from djitellopy import Tello
    import cv2
    import time

    tello = Tello()

    # ---- CONNECT TO TELLO ----
    try:
        tello.connect()
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Make sure you are connected to Tello WiFi and firewall is not blocking UDP.")
        exit(1)

    tello.streamon()
    frame_read = tello.get_frame_read()

    marker_id = 1  # Change to your marker ID

    print("Starting pose tracking...  Press ESC to exit")

    while True:
        frame = frame_read.frame

        if frame is None:
            continue

        # ---- Compute pose ----
        pose = get_drone_position(frame, marker_id)

        if pose is not None and not any(map(lambda v: np.isnan(v), pose)):
            x, y, z, yaw = pose
            print(f"[POSE] x={x:.2f}  y={y:.2f}  z={z:.2f}  yaw={yaw:.2f} rad")
        else:
            print("[POSE] Marker not detected")

        # ---- Visualization ----
        cv2.imshow("Frame", frame)

        # ESC to quit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        time.sleep(0.03)  # ~30 FPS loop smoothing

    print("Stopping stream...")
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()