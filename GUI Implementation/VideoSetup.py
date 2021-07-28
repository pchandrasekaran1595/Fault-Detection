import cv2
import platform

class Video(object):
    def __init__(self, id=None, width=None, height=None, fps=None):
        """
            id     : Device ID of the capture object
            width  : Width of the capture frame
            height : Height of the capture frame
            fps    : FPS of the capture object
        """
        self.id = id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
    
    def start(self):
        """
            Initialize the capture object
        """
        if platform.system() != 'Windows':
            self.cap = cv2.VideoCapture(self.id)
        else:
            self.cap = cv2.VideoCapture(self.id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
    
    def get_frame(self):
        """
            Read a frame from the capture object
        """
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return ret, cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
            else:
                return ret, None

    def stop(self):
        """
            Stop the capture object
        """
        if self.cap.isOpened():
            self.cap.release()
