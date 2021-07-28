import cv2
import platform

class Video(object):
    def __init__(self, id=None, width=None, height=None, fps=None):
        self.id = id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
    
    def start(self):
        if platform.system() != 'Windows':
            self.cap = cv2.VideoCapture(self.id)
        else:
            self.cap = cv2.VideoCapture(self.id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
    
    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return ret, cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
            else:
                return ret, None

    def stop(self):
        if self.cap.isOpened():
            self.cap.release()
