import cv2
import numpy as np

from ponit_selector import PointSelector

class SquareDetector:
    def __init__(self, size=500):
        self.size = size
        self.mask = []
    
    def getMask(self, points, image):
        half_size = self.size / 2
        for i in range(len(points)):    
            # Calculate the coordinates of the square's corners
            top_left = (int(points[i][0] - half_size), int(points[i][1] - half_size))
            bottom_right = (int(points[i][0] + half_size), int(points[i][1] + half_size))
            self.mask.append((top_left, bottom_right))
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
        return self.mask, image

    def selectSquares(self, image):
        selector = PointSelector(image, "Selecciona los puntos")
        selector.show_and_wait()
        roi = selector.points
        point = np.array(roi)
        self.mask , result_image= self.getMask(point, image)
        cv2.imshow('Result', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self.mask    
