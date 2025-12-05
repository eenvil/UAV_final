from djitellopy import Tello
from pynput import keyboard
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np
from state1 import state1
from state2 import state2
from state3 import state31,state32
from state4 import state4
from state5 import state5
from state6 import state6
from state7 import state71,state72
# --------- GLOBAL STATE ---------
debug_frame: Optional[np.ndarray] = None
debug_frame_lock = threading.Lock()

# Current mode: "manual" or "auto"
mode = {"value": "manual"}

# Movement state for manual control
movement = {
    "left_right": 0,     # A/D
    "forward_back": 0,   # W/S
    "up_down": 0,        # SPACE/SHIFT
    "yaw": 0,            # Q/E
}

# Optional: state for your auto controller
auto_state = {
    "t": 0.0,  # example time variable you might want to use
    "current_state": "state1",  # example state variable
}

listener: Optional[keyboard.Listener] = None
# --------- END GLOBAL STATE ---------
# --------- AUTO MODE BEHAVIOR (YOU EDIT HERE) ---------
cap = None
def auto_step(tello: Tello, state: dict) -> Tuple[int, int, int, int]:
    """
    Compute velocities (lr, fb, ud, yaw) in AUTO mode.

    You can change this logic to whatever you want.
    This function is called ~20 times per second when:
      - mode == "auto"
      - and there is NO manual key pressed (manual override).
    """
    # Example: simple "do nothing" behavior
    # Replace this with your own logic
    state["t"] += 0.05
    global cap, debug_frame
    lr = fb = ud = yw = 0
    debug_frame = None
    if state["current_state"] == "state1":
        [lr, fb, ud, yw], debug_frame, next_state = state1(cap.frame)
        state["current_state"] = next_state
    elif state["current_state"] == "state2":
        [lr, fb, ud, yw], debug_frame, next_state = state2(cap.frame)
        state["current_state"] = next_state
    elif state["current_state"] == "state31":
        [lr, fb, ud, yw], debug_frame, next_state = state31(cap.frame)
        state["current_state"] = next_state
    elif state["current_state"] == "state32":
        [lr, fb, ud, yw], debug_frame, next_state = state32(cap.frame)
        state["current_state"] = next_state
    elif state["current_state"] == "state4":
        [lr, fb, ud, yw], debug_frame, next_state = state4(cap.frame)
        state["current_state"] = next_state
    elif state["current_state"] == "state5":
        [lr, fb, ud, yw], debug_frame, next_state = state5(cap.frame)
        state["current_state"] = next_state
    elif state["current_state"] == "state6":
        [lr, fb, ud, yw], debug_frame, next_state = state6(cap.frame)
        state["current_state"] = next_state
    elif state["current_state"] == "state71":
        [lr, fb, ud, yw], debug_frame, next_state = state71(cap.frame)
        state["current_state"] = next_state
    elif state["current_state"] == "state72":
        [lr, fb, ud, yw], debug_frame, next_state = state72(cap.frame)
        state["current_state"] = next_state
    else:
        # Unknown state, do nothing
        lr = fb = ud = yw = 0
    if cap is not None:
        frame = cap.frame
        # Save a debug frame for the main thread to show
        with debug_frame_lock:
            debug_frame = frame.copy()
            debug_frame = cv2.putText(
                debug_frame, "AUTO MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
        
    return lr, fb, ud, yw


# --------- KEYBOARD HANDLERS ---------

def on_press(key) -> None:
    global listener

    ch: Optional[str] = None
    if isinstance(key, keyboard.KeyCode):
        ch = key.char.lower() if key.char is not None else None

    # ----- MODE SWITCHING -----
    if ch == '1':
        mode["value"] = "manual"
        print("[MODE] Switched to MANUAL")
    elif ch == '2':
        mode["value"] = "auto"
        print("[MODE] Switched to AUTO")

    # ----- MANUAL MOVEMENT KEYS -----
    # Forward / back
    if ch == 'w':
        movement["forward_back"] = 50
    elif ch == 's':
        movement["forward_back"] = -50

    # Left / right
    if ch == 'a':
        movement["left_right"] = -50
    elif ch == 'd':
        movement["left_right"] = 50

    # Rotate yaw
    if ch == 'q':
        movement["yaw"] = -50
    elif ch == 'e':
        movement["yaw"] = 50

    # Up / down
    if key == keyboard.Key.space:
        movement["up_down"] = 50
    elif key == keyboard.Key.shift:
        movement["up_down"] = -50

    # Quit with ESC
    if key == keyboard.Key.esc:
        print("Exiting…")
        if listener is not None:
            listener.stop()


def on_release(key) -> None:
    ch: Optional[str] = None
    if isinstance(key, keyboard.KeyCode):
        ch = key.char.lower() if key.char is not None else None

    # Forward / back
    if ch in ('w', 's'):
        movement["forward_back"] = 0

    # Left / right
    if ch in ('a', 'd'):
        movement["left_right"] = 0

    # Yaw rotation
    if ch in ('q', 'e'):
        movement["yaw"] = 0

    # Up / down
    if key in (keyboard.Key.space, keyboard.Key.shift):
        movement["up_down"] = 0


# --------- CONTROL LOOP ---------

def any_manual_input() -> bool:
    return (
        movement["left_right"] != 0 or
        movement["forward_back"] != 0 or
        movement["up_down"] != 0 or
        movement["yaw"] != 0
    )


def rc_loop(tello: Tello, running_flag: list[bool]) -> None:
    """
    Runs in a separate thread.
    Sends rc_control commands ~20 times per second.
    - If any manual keys are pressed: MANUAL has priority.
    - Else if mode == "auto": use auto_step().
    - Else: stop (0,0,0,0).
    """
    global cap, debug_frame
    while running_flag[0]:
        with debug_frame_lock:
            debug_frame = None  # Clear debug frame each loop
        if any_manual_input():
            # Manual override always wins
            lr = movement["left_right"]
            fb = movement["forward_back"]
            ud = movement["up_down"]
            yw = movement["yaw"]
        else:
            if mode["value"] == "auto":
                lr, fb, ud, yw = auto_step(tello, auto_state)
            else:
                lr = fb = ud = yw = 0
        if debug_frame is None:
            # If no debug frame set by auto_step, get current frame
            if cap is not None:
                frame = cap.frame
                with debug_frame_lock:
                    debug_frame = frame.copy()
        tello.send_rc_control(lr, fb, ud, yw)
        time.sleep(0.05)  # 20 Hz control loop


# --------- MAIN ---------

def main() -> None:
    global listener, cap

    tello = Tello()
    # set tello logging to not display
    tello.set_debug_mode(False)
    tello.connect()
    tello.streamon()
    cap = tello.get_frame_read()
    print("Battery:", tello.get_battery())

    tello.takeoff()
    print("[INFO] Takeoff complete. Press 1=MANUAL, 2=AUTO, ESC=quit.")

    running = [True]

    # Start RC loop thread
    t = threading.Thread(target=rc_loop, args=(tello, running), daemon=True)
    t.start()

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # ---- MAIN THREAD: show debug frame while waiting for ESC ----
    while listener.is_alive() and running[0]:
        # get a local copy of the frame
        with debug_frame_lock:
            frame = None if debug_frame is None else debug_frame.copy()

        if frame is not None:
            cv2.imshow("Debug", frame)

        # Allow OpenCV window to process events
        key = cv2.waitKey(1) & 0xFF
        # Optional: also allow quitting via window ESC
        if key == 27:  # ESC
            listener.stop()
            break

        time.sleep(0.01)  # small sleep to avoid busy-looping

    # -------------------------------------------------------------

    running[0] = False

    # Make sure drone stops & lands
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    cv2.destroyAllWindows()
    print("Landed and exited.")


if __name__ == "__main__":
    main()