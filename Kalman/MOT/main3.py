
import copy
import matplotlib.pyplot as plt
import cv2
from tracker import ObjectTracker
from Test import *

def main():
    
    cap = cv2.VideoCapture("C:/Users/aa10098/Desktop/CV_project\MOT_Kalman_DeepSORT_StrongSORT/Data/pedestrians.mp4")

    tracker = ObjectTracker(160, 8, 3,1)
    
    labels = open('C:/Users/aa10098/Desktop/CV_project/YOLOv3-Object-Detection-with-OpenCV/yolov3-coco/coco-labels').read().strip().split('\n')
    net = cv2.dnn.readNetFromDarknet('C:/Users/aa10098/Desktop/CV_project/YOLOv3-Object-Detection-with-OpenCV/yolov3-coco/yolov3.cfg',
                                        'C:/Users/aa10098/Desktop/CV_project/YOLOv3-Object-Detection-with-OpenCV/yolov3-coco/yolov3.weights')
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    print("layer names are gotton")
    height, width = None, None
    writer = None
    
    def save_video(writer, image):
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter('./output.avi', fourcc, 30, (image.shape[1], image.shape[0]), True)
        writer.write(image)
        return writer 

    while(True):
    
        ret, frame = cap.read()
        if not ret or frame is None:
            print("frame is none")
            break

        if width is None or height is None:
            height, width = frame.shape[:2]
        
        orig_frame = copy.copy(frame)

        centers = infer_image(net, layer_names, height, width, frame, labels)
        ("centers are now here")
        print(f"length of centers {len(centers)}")

        if (len(centers) > 0):

            tracker.Update(centers)

            draw_line(tracker,frame)
            
            cv2.imshow('Tracking', frame)

            # save_video(writer,frame)
            writer = save_video(writer, frame)  # Call the save_video function inside the loop to save each frame

        # cv2.imshow('Original', orig_frame)
        cv2.waitKey(5)

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()