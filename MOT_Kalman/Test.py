

import copy
import numpy as np
import cv2
from Tracker import ObjectTracker

### apply the data video here
def main():
    cap = cv2.VideoCapture("123.mp4")
    tracker = ObjectTracker(160, 8, 3,1)
    labels = open('./yolov3-coco/coco-labels').read().strip().split('\n')
    net = cv2.dnn.readNetFromDarknet('./yolov3-coco/yolov3.cfg', './yolov3-coco/yolov3.weights')
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    height, width = None, None
    writer = None
    
    while(True):
        ret, frame = cap.read()
        if width is None or height is None:
            height, width = frame.shape[:2]

        orig_frame = copy.copy(frame)
        if not ret:
            break

        centers = infer_image(net, layer_names, height, width, frame, labels)
        if (len(centers) > 0):
            tracker.Update(centers)
            draw_line(tracker,frame)
            cv2.imshow('Tracking', frame)
            save_video(writer,frame)

        cv2.imshow('Original', orig_frame)
        cv2.waitKey(50)

    writer.release()
    cap.release()
    cv2.destroyAllWindows()

def infer_image(net, layer_names, height, width, img, labels, boxes=None, confidences=None, classids=None, idxs=None):
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)
    boxes, confidences, classids,center = generate_boxes(outs, height, width, 0.5,labels)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    img = draw_boxes(img, boxes, confidences, classids, idxs, labels)
    return center

def generate_boxes(outs, height, width, tconf,labels):
    boxes = []
    confidences = []
    classids = []
    center=[]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            
            if confidence > tconf and labels[classid]=='person':
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                boxes.append([x, y, int(bwidth), int(bheight)])
                center.append(np.array([[x], [y]]))
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids,center

def draw_boxes(img, boxes, confidences, classids, idxs, labels):
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            color = (255,0,0)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    return img

def save_video(writer,image):
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter('./output5.avi', fourcc, 30, (image.shape[1], image.shape[0]), True)
    writer.write(image)

def draw_line(tracker,image):
    for i in range(len(tracker.objects)):
        if (len(tracker.objects[i].line) > 1):
            for j in range(len(tracker.objects[i].line)-1):
                x1 = tracker.objects[i].line[j][0][0]
                y1 = tracker.objects[i].line[j][1][0]
                x2 = tracker.objects[i].line[j+1][0][0]
                y2 = tracker.objects[i].line[j+1][1][0]
                clr = tracker.objects[i].object_id
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)),(0,0,0), 2)
            cv2.putText(image,str(clr), (int(x1)-10,int(y1)-20),0, 0.5, (0,0,0),2)


if __name__ == "__main__":

    main()