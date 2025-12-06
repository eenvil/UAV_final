# locate aruco and detect doll in frame
import func
import numpy as np
import cv2
import torch
def state2(frame):
    # todo: implement state2 behavior
    raise NotImplementedError("state2 behavior not implemented yet")
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