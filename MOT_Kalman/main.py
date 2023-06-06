
import numpy as np
import matplotlib.pyplot as plt
import cv2
from detect import Detect
from kalman import Kalman2D

def main():
    # dt, a_x, a_y, sd_acceleration, x_sd, y_sd
    KF = Kalman2D(0.1, 1, 1, 1, 0.1,0.1)
    Video = cv2.VideoCapture('../Data/pedestrians.mp4')
    detection = Detect()

    E=[]
    t=[]
    G=[[] , [] , [] , [], [], []]
    X=[[]]
    i=0
    writer=None
    
    while(True):
        ret, frame = Video.read()
        if not ret:
            break
    
        centers = detection.get_centroid(frame)
        (x, y) = KF.predict()
        X.append([x,y])
        
        if (len(centers) > 0):
            cv2.circle(frame, (int(centers[0][0]), int(centers[0][1])), 10, (0, 191, 255), 2)
            x,y=int(x),int(y)
            G[4].append(x)
            G[5].append(y)

            (x1, y1) = KF.update(centers[0])
            x1=int(x1)
            y1=int(y1)
            G[0].append(x1)
            G[1].append(y1)
            x,y=int(x),int(y)
            
            G[2].append(int(centers[0][0]))
            G[3].append(int(centers[0][1]))
            E.append(((int(centers[0][0])-x1)**2 + (int(centers[0][1])-y1)**2)**0.5)
            t.append(i)
            cv2.rectangle(frame, (x1 - 14, y1 - 14), (x1 + 14, y1 + 14), (255, 0, 0), 2)
            cv2.putText(frame, "Estimated Position", (x1 + 14, y1 + 9), 0, 0.5, (0, 100, 255), 2)

            cv2.putText(frame, "Measured Position", (int(centers[0][0]) + 15, int(centers[0][1]) - 15), 0, 0.5, (0,255,100), 2)
        if i>3:
            cv2.line(frame, (int(X[i][0]), int(X[i][1])), (int(X[i-3][0]), int(X[i-3][1])),(0,0,255), 2)
        i+=1
        
        cv2.imshow('image', frame)
        
        if writer is None:
            writer = cv2.VideoWriter('./output6.avi', cv2.VideoWriter_fourcc(*"MJPG") , 30, (frame.shape[1], frame.shape[0]), True)
        writer.write(frame)
        
        cv2.waitKey(70)
    Video.release()
    cv2.destroyAllWindows()

    plt.plot(G[0],G[1],label='Estimated')
    plt.plot(G[2],G[3],label='Measurement')
    
    plt.xlabel('X', fontsize=20)
    plt.ylabel('Y', fontsize=20)
    plt.legend()
    plt.show()
    
    plt.plot(G[4],G[5],label='Predicted')
    plt.plot(G[2],G[3],label='Measurement')
    
    plt.xlabel('X', fontsize=20)
    plt.ylabel('Y', fontsize=20)
    plt.legend()
    plt.show()
    
    plt.plot(G[0],G[1],label='Estimated')
    plt.plot(G[4],G[5],label='Predicted')
    plt.plot(G[2],G[3],label='Measurement')
    
    plt.xlabel('X', fontsize=20)
    plt.ylabel('Y', fontsize=20)
    plt.legend()
    plt.show()
    
    plt.plot(t,E)
    S= str(sum(E)/len(E))
    plt.title("Average Deviation from Measured "+ S[:3])
    plt.ylabel('Deviation', fontsize=20)
    plt.xlabel('time step', fontsize=20)
    plt.show()



if __name__ == "__main__":
    main()