import numpy as np
import cv2


def calc_centroid(image,thresh):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        blob_thresh = 4
        for i in contours:
            try:
                (x, y), r = cv2.minEnclosingCircle(i)
                centeroid = (int(x), int(y))
                r = int(r)
                if (r > blob_thresh):
                    cv2.circle(image, centeroid, r, (0, 0, 255), 2)
                    coords = np.array([[x], [y]])
                    centroids.append(np.round(coords))
            except ZeroDivisionError:
                pass
        return centroids

class Detect(object):
    def __init__(self):
        self.bgd = cv2.createBackgroundSubtractorMOG2()

    def get_centroid(self, image):
        g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        try:
            cv2.imshow('Gray Scaled', g)
        except:
            print("End")

        f = self.bgd.apply(g)
        e = cv2.Canny(f, 50, 190, 3)
        _, thresh = cv2.threshold(e, 127, 255, 0)
        return calc_centroid(image, thresh)