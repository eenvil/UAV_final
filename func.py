
from traitlets import Tuple


def detect_doll(frame) -> int:
    '''
    Docstring for detect_doll
    
    :param frame: the current video frame
    :return: 0 no doll detected, 1 kana, 2 melody
    :rtype: int
    '''
    pass
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

