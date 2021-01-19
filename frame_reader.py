import cv2


class FrameReader:
    """
    read frame from camera.
    """

    def __init__(self, width=640, height=480, camera_id=0):
        self.width = width
        self.height = height
        self.capture = cv2.VideoCapture(camera_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        ret, img = self.capture.read()
        if not ret:
            raise RuntimeError("fail to read frame!")
        return img  # img in gbr format
