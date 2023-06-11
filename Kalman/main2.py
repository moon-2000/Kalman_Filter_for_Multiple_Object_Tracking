
import copy
import matplotlib.pyplot as plt
import cv2
from tracker import ObjectTracker
from Test import *

def main():
    
    cap = cv2.VideoCapture("../Data/pedestrians.mp4")

    tracker = ObjectTracker(160, 8, 3,1)

    labels = open('../yolov5/data/coco.yaml',  encoding='utf-8').read().strip().split('\n')
    # Give the weight files to the model and load the network
    modelWeights = "../yolov5/models/YOLOv5s.onnx"
    net = cv2.dnn.readNetFromONNX(modelWeights)
    
    layer_names = net.getLayerNames()
    layer_indices = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in layer_indices]
    

    height, width = None, None
    writer = None
    
    while(True):
        
        ret, frame = cap.read()
        if width is None or height is None:
            height, width = frame.shape[:2]
        
        orig_frame = copy.copy(frame)
        
        if not ret:
            break

        centers = infer_image(net, output_layers, height, width, frame, labels)
        # print(f" centers {centers}")

        if (len(centers) > 0):

            tracker.Update(centers)

            draw_line(tracker,frame)
            
            cv2.imshow('Tracking', frame)

            save_video(writer,frame)



        cv2.imshow('Original', orig_frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break




    writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()