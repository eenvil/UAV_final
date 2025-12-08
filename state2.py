# locate aruco and detect doll in frame
import func
import numpy as np
import cv2
import torch
import datetime

doll_detected = 0  # 0: none, 1: carna, 2: melody
doll_detect_thresh = 0.5  # detection threshold (seconds)
last_doll_detect_time = 0
last_doll_detected = 0
last_detect_time = 0
marker_id = 1
location = np.array([0.0, 0.07, 0.85,0.0])  # x, y, z offsets for marker id 1
location_time_thresh = 0.5  # seconds
last_arrival_time = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

def state2(frame):
    # detect doll every 0.1 seconds
    global doll_detected, last_detect_time, marker_id, last_doll_detected, last_doll_detect_time
    global location, last_arrival_time
    next_state = "state2"
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
    lr, fb, ud, yw = func.track_marker(frame, location, marker_id)
    if any(v != 0 for v in [lr, fb, ud, yw]) or func.get_drone_position(frame,marker_id) is None:
        last_arrival_time = current_time
    if current_time - last_arrival_time >= location_time_thresh:
        # reached location
        lr = fb = ud = yw = 0
        if doll_detected != 0:
            next_state = "state31" if doll_detected == 1 else "state32"
    
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




if __name__ == "__main__":

    SOURCE_VIDEO = 'demo_vid.mp4' # 你的測試影片
    
    # 初始化裝置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 開啟影片
    cap = cv2.VideoCapture(SOURCE_VIDEO)
    if not cap.isOpened():
        print(f"Error: Could not open video file {SOURCE_VIDEO}")
        exit()

    print(f"Processing {SOURCE_VIDEO}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break
        
        # ---------------------------------------------------------
        # 關鍵點：這裡呼叫你的函式，就像無人機在飛一樣
        # ---------------------------------------------------------
        result = func.detect_doll(frame)

        # ---------------------------------------------------------
        # 根據結果在畫面上印出文字
        # ---------------------------------------------------------
        # 預設文字與顏色 (灰色 NONE)
        text_str = "DETECT: NONE"
        text_color = (128, 128, 128) 

        if result == 1:
            text_str = "DETECT: CARNA (Target 1)"
            text_color = (0, 255, 0) # 綠色
        elif result == 2:
            text_str = "DETECT: MELODY (Target 2)"
            text_color = (0, 0, 255) # 紅色

        # 將文字畫在左上角
        cv2.putText(frame, text_str, (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)

        # 顯示畫面
        cv2.imshow("Detection Test", frame)

        # 按 'q' 離開
        if cv2.waitKey(1) == ord('q'):
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()