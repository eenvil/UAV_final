import time
import cv2
import re
import glob
import os
from datetime import datetime

"""
state1: replay manual flight from a recorded debug .log file.

假設 log 格式像：

DEBUG:root:Time: 2025-12-08 18:31:13.711, LR: 0, FB: 0, UD: 0, YW: 0

這個版本很單純：
- 每一行 log -> 一個 step
- 每個 step duration 固定 0.05 秒
- 不做「連續同指令合併」，所以 step 數大約 = log 行數
"""

# 找不到 log 或解析失敗時的備案
FALLBACK_SEQUENCE = [
    (1.0, 0, 0, 0, 0),
    (1.0, 0, 0, 50, 0),
    (1.0, 0, 0, 0, 0),
]

# 狀態機內部變數
_sequence_index = 0
_step_start_time = None
_state1_start_time = None
_flight_sequence = None
_sequence_loaded = False

# 解析 log 的 regex（符合你的 DEBUG line）
LOG_LINE_RE = re.compile(
    r"Time:\s*([0-9\-]+\s+[0-9:\.]+),\s*LR:\s*(-?\d+),\s*FB:\s*(-?\d+),\s*UD:\s*(-?\d+),\s*YW:\s*(-?\d+)"
)


def _parse_log_to_sequence(log_path):
    """
    從單一 log 檔讀出 (Time, LR, FB, UD, YW)，
    轉成 [(duration, LR, FB, UD, YW), ...]
    這裡我們不管實際 dt，直接每行給 0.05 秒。
    """
    entries = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOG_LINE_RE.search(line)
            if not m:
                continue
            timestr, lr, fb, ud, yw = m.groups()

            # 轉時間只是驗證用，其實後面沒有用到
            ts = None
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                try:
                    ts = datetime.strptime(timestr, fmt)
                    break
                except ValueError:
                    continue
            if ts is None:
                continue

            entries.append((ts, (int(lr), int(fb), int(ud), int(yw))))

    if not entries:
        return None

    # 每一筆都當成一個 step，duration 固定 0.05 秒
    seq = []
    for _, cmd in entries:
        lr, fb, ud, yw = cmd
        seq.append((0.05, lr, fb, ud, yw))

    return seq if seq else None


def _load_latest_log_sequence():
    """
    在 state1.py 同一個資料夾中找最新的 tello_debug_*.log，
    解析成 flight sequence。
    """
    base_dir = os.path.dirname(__file__)
    logs = glob.glob(os.path.join(base_dir, "tello_debug_*.log"))
    if not logs:
        print("[state1] No log files found, using FALLBACK_SEQUENCE")
        return None

    latest = max(logs, key=os.path.getmtime)
    print(f"[state1] Using log file: {os.path.basename(latest)}")
    try:
        seq = _parse_log_to_sequence(latest)
        if seq is None:
            print("[state1] Failed to parse log, using FALLBACK_SEQUENCE")
        else:
            print(f"[state1] Parsed {len(seq)} steps from log.")
        return seq
    except Exception as e:
        print(f"[state1] Exception while parsing log: {e}")
        return None


def state1(frame):
    """
    State 1: 依照最新 log 檔重播你手飛時的 RC 指令。

    回傳:
      [lr, fb, ud, yw], debug_frame, next_state
    """
    global _sequence_index, _step_start_time, _state1_start_time
    global _flight_sequence, _sequence_loaded

    # 第一次進來 state1 時載入 sequence
    if not _sequence_loaded:
        _flight_sequence = _load_latest_log_sequence() or FALLBACK_SEQUENCE
        _sequence_loaded = True
        _sequence_index = 0
        _step_start_time = None
        _state1_start_time = None

    # 初始化時間
    if _state1_start_time is None:
        _state1_start_time = time.time()
        _step_start_time = time.time()

    current_time = time.time()
    step_elapsed = current_time - _step_start_time

    # 預設指令是 0（保險）
    lr = fb = ud = yw = 0

    if _sequence_index < len(_flight_sequence):
        duration, lr, fb, ud, yw = _flight_sequence[_sequence_index]
        # 這個 step 的時間到了就進入下一步
        if step_elapsed >= duration:
            _sequence_index += 1
            _step_start_time = current_time
            if _sequence_index < len(_flight_sequence):
                duration, lr, fb, ud, yw = _flight_sequence[_sequence_index]
            else:
                lr = fb = ud = yw = 0  # 播完了就停在原地
    else:
        lr = fb = ud = yw = 0  # safety

    # 畫 debug 畫面
    debug_frame = frame.copy() if frame is not None else None
    if debug_frame is not None:
        src = "log" if _flight_sequence is not FALLBACK_SEQUENCE else "fallback"
        cv2.putText(
            debug_frame,
            f"State1 ({src}): Step {_sequence_index}/{len(_flight_sequence)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            debug_frame,
            f"LR:{lr} FB:{fb} UD:{ud} YW:{yw}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # 你現在不想切到 state2，那就一直留在 state1
    next_state = "state1"
    return [lr, fb, ud, yw], debug_frame, next_state
