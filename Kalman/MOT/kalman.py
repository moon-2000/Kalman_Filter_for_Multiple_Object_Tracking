

import numpy as np
import matplotlib.pyplot as plt

class Kalman2D(object):
    # implements a 2D Kalman filter for tracking the position and velocity of an object in a Cartesian coordinate system.

    def __init__(self, dt, a_x, a_y, sd_acceleration, x_sd, y_sd):
        
        #  time step between consecutive measurements.
        self.dt = dt

        # the acceleration which is essentially a from the state update equation, 
        # representing the acceleration components as a column vector.
        self.a = np.matrix([[a_x],[a_y]])


        #  the state transition matrix that predicts the next state based on the previous state and acceleration.
        self.A = np.matrix([[1, 0, self.dt, 0],[0, 1, 0, self.dt],[0, 0, 1, 0],[0, 0, 0, 1]])

        # the control input transition matrix that incorporates the effect of acceleration on the state.
        self.B = np.matrix([[(self.dt**2)/2, 0],[0,(self.dt**2)/2],[self.dt,0],[0,self.dt]])

        # the measurement matrix that maps the state vector to the measurements (position).
        self.H = np.matrix([[1, 0, 0, 0],[0, 1, 0, 0]])

        # processs covariance that for our case depends solely on the acceleration, that models the uncertainty in the state transition due to acceleration.
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],[0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],[0, (self.dt**3)/2, 0, self.dt**2]]) * sd_acceleration**2

        # measurement covariance matrix that represents the uncertainty in the measurements.
        self.R = np.matrix([[x_sd**2,0],
                            [0, y_sd**2]])

        # the error covariance matrix that is Identity for now. It gets updated based on Q, A and R.
        # initialized as an identity matrix, representing the initial uncertainty in the state estimate.
        self.P = np.eye(self.A.shape[1])
        
        # State vector representing the position and velocity of the object [ x position ;  y position ; x velocity ; y velocity ; ]
        self.x = np.matrix([[0], [0], [0], [0]])


    def predict(self):
        # The state update : X(t) = A*X_(t-1) + B*a 
        # here acceleration is a 

        # updates the state estimate by multiplying the state transition matrix (self.A) with the current state (self.x)
        #  and adding the control input (self.B) multiplied by the acceleration self.a
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.a)
        
        # updates the error covariance matrix self.P using the prediction equation.
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        # returns the predicted position from the updated state vector self.x
        return self.x[0:2]


    def update(self, z):
        """
        Inputs:
            - z: Measurement vector containing the actual position of the object.
        """
        # calculates the innovation covariance matrix S 
        # by multiplying the measurement matrix self.H, the error covariance matrix self.P, and their transposes, and then adding the measurement covariance matrix self.R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # calculates the Kalman gain K by multiplying self.P, the transpose of self.H, and the inverse of S
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) 

        # updates the state estimate self.x using the measurement update equation and rounding the result.
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))  

        # calculates the identity matrix I
        I = np.eye(self.H.shape[1])

        # updates the error covariance matrix self.P using the Joseph form of the covariance update equation.
        self.P = (I -(K*self.H))*self.P  
        
        # returns the updated position from the updated state vector self.x.
        return self.x[0:2]