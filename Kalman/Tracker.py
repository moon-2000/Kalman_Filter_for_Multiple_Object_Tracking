import numpy as np
from kalman import Kalman2D
from Hungarian import get_minimum_cost_assignment




class Object(object):

    def __init__(self, detect , ID):
        
        self.prediction = np.asarray(detect)
        self.object_id = ID 
        self.KF = Kalman2D(0.1, 1, 1, 1, 0.1,0.1)
        self.skip_count = 0 
        self.line = [] 


class ObjectTracker(object):

    def __init__(self, min_dist , max_skip ,line_length , object_id):
    
        self.min_dist  = min_dist  #  minimum distance threshold to consider a detection as a match
        self.max_skip = max_skip  #  maximum number of consecutive frames to skip before considering an object as lost.
        self.line_length = line_length  # maximum number of historical positions to store in the object's line.
        self.objects = []  # list to store the tracked objects.
        self.object_id = object_id  #  an identifier for the next object to be tracked.

    def Update(self, detections):
        if self.objects ==[]:
            
            for i in range(len(detections)):
                self.objects.append( Object(detections[i], self.object_id))
                self.object_id += 1
        
        N , M = len(self.objects), len(detections)
        print(f"N and M is {N} and {M}")
        cost_matrix = np.zeros(shape=(N, M))
        for i in range(N):
            for j in range(M):
                diff = self.objects[i].prediction - detections[j]
                # print(f" diff dimensions are  {diff.shape}")
                scaling_factor = 1e-5  # Adjust the scaling factor as needed
                normalized_diff = diff / scaling_factor
                cost_matrix[i][j] = np.sqrt(np.sum(normalized_diff ** 2))
                # cost_matrix[i][j] = np.sqrt(diff[0]*diff[0] + diff[1]*diff[1])
                # cost_matrix[i][j] = np.sqrt(diff[0][0]*diff[0][0] +diff[1][0]*diff[1][0])

        cost_matrix = (0.5) * cost_matrix 
        assign = []
        for _ in range(N):
            assign.append(-1)
            
        rows, cols = get_minimum_cost_assignment(cost_matrix)
        for i in range(len(rows)):
            assign[rows[i]] = cols[i]

        unassign = []
        for i in range(len(assign)):
            if (assign[i] != -1):
                if (cost_matrix[i][assign[i]] > self.min_dist):
                    assign[i] = -1
                    unassign.append(i)
            else:
                self.objects[i].skip_count += 1

        del_objects = []
        for i in range(len(self.objects)):
            if (self.objects[i].skip_count > self.max_skip):
                del_objects.append(i)
        if len(del_objects) > 0: 
            for id in del_objects:
                if id < len(self.objects):
                    del self.objects[id]
                    del assign[id]         

        for i in range(len(detections)):
                if i not in assign:
                    self.objects.append( Object( detections[i], self.object_id )  )
                    self.object_id += 1

        for i in range(len(assign)):
            self.objects[i].KF.predict()

            if(assign[i] != -1):
                self.objects[i].skip_count = 0
                self.objects[i].prediction = self.objects[i].KF.update( detections[assign[i]])
            else:
                self.objects[i].prediction = self.objects[i].KF.update( np.array([[0], [0]]))

            if(len(self.objects[i].line) > self.line_length):
                for j in range( len(self.objects[i].line) - self.line_length):
                    del self.objects[i].line[j]

            self.objects[i].line.append(self.objects[i].prediction)
            self.objects[i].KF.lastResult = self.objects[i].prediction



