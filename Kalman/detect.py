import numpy as np
import cv2




class Detect(object):
    def __init__(self):
        self.bgd = cv2.createBackgroundSubtractorMOG2()  # initializes the bgd variable with a background subtractor

    def get_centroid(self, image):
        """
        Inputs:
            - image: The input image on which centroid detection will be performed.

        Output:
            - Returns the output of the calculate_centroid function by passing the input image and the thresholded image 
                obtained by applying background subtraction and Canny edge detection to the grayscale version of the input image.
        """
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts the input image to grayscale.
        
        try:
            cv2.imshow('Gray Scaled', gray_img)
        except:
            print("End")

        f = self.bgd.apply(gray_img)  # applies background subtraction to the grayscale image.
        canned_img = cv2.Canny(f, 50, 190, 3)  # performs Canny edge detection on the background subtracted image.
        _, thresh = cv2.threshold(canned_img, 127, 255, 0)  # 127: The threshold value. Pixels with intensity values greater than this threshold will be set to the maximum value (255) specified in the next parameter.
                                                            # 255: The maximum value to be assigned to pixels that exceed the threshold.
                                                            # 0: The type of thresholding to be applied. In this case, it uses a binary thresholding method, where pixels that are above the threshold value are set to the maximum value, and pixels below the threshold are set to zero.
        return calculate_centroid(image, thresh)  # passing the original input image and the thresholded image obtained from the previous steps, and returns the result

def calculate_centroid(image,thresh):
    """
    Inputs:
        - image: The input image on which the centroid calculation will be performed.
        - thresh: The thresholded image used for contour detection.

    Output:
        - centroids: A list of 2D coordinates representing the centroids of the detected contours.
    """
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # detect contours in the thresholded image.
    centroids = []
    blob_thresh = 4
    for i in contours:
        try:
            (x, y), r = cv2.minEnclosingCircle(i)  # for each contour, it tries to find the minimum enclosing circle
            centeroid = (int(x), int(y))
            r = int(r)
            if (r > blob_thresh):  # if the radius of the enclosing circle is greater than the threshold blob_thresh, 
                                    
                cv2.circle(image, centeroid, r, (0, 0, 255), 2)   # draws a circle on the input image around the centroid
                coords = np.array([[x], [y]])  # appends the rounded centroid coordinates to the centroids list.
                centroids.append(np.round(coords))
        except ZeroDivisionError:
            pass
    return centroids