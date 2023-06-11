

import copy
import numpy as np
import cv2



def infer_image(net, layer_names, height, width, img, labels, boxes=None, confidences=None, classids=None, idxs=None):
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)
    boxes, confidences, classids, center = generate_boxes(img, outs, height, width, 0.5, labels)
    # print(f"center is {center}")
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # print(f"idx is {idxs}")

    img = draw_boxes(img, boxes, confidences, classids, idxs, labels)
    return center

def generate_boxes(img, outs, height, width, tconf, labels):
    boxes = []
    confidences = []
    classids = []
    center = []

    for out in outs[0]:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > tconf:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                cv2.circle(img ,(center_x,center_y),10,(0,255,0),2)

                x = int(center_x-w/2)
                y = int(center_y - h/2)
                
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                classids.append(class_id)
                center.append((center_x, center_y))  # Append center coordinates


    return boxes, confidences, classids, center



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


