
from traitlets import Tuple
import numpy as np
import torch
import cv2
from torchvision import transforms
from yolo.models.experimental import attempt_load
from yolo.utils.datasets import letterbox
from yolo.utils.general import non_max_suppression_kpt, scale_coords
from yolo.utils.plots import plot_one_box
from yolo.models.experimental import attempt_load
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


def track_marker(frame,marker_id) -> Tuple[int, int, int, int]:
    '''
    Docstring for track_marker
    
    :param frame: the current video frame
    :return: (lr, fb, ud, yw) velocities to track the marker
    :rtype: Tuple[int, int, int, int]
    '''
    pass
    return 0, 0, 0, 0

def get_drone_position(frame, marker_id) -> Tuple[float, float, float]:
    '''
    Docstring for get_drone_position
    
    :param frame: the current video frame
    :return: (x, y, z) position of the drone
    :rtype: Tuple[float, float, float]
    '''
    pass
    return
