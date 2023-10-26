# Kalman_Filter_for_Multiple_Object_Tracking_(MOT)


#### "Kalman filters are recursive algorithms used to estimate the state of a linear dynamic system. 
In the context of MOT, Kalman filters can be used to predict the future state (position, velocity, ..etc) of each object based on its previous states and measurements, even in the presence of noisy or incomplete data."


### Steps of the project:
#### 1 - State Representation
In this step, the state vector for each object was defined. For instance, in the context of tracking objects in 2D space, the state vector included position (x, y) and velocity (vx, vy).
Initialization

Each object was initialized with a Kalman filter, providing the initial state and covariance matrix. This matrix represented the uncertainty associated with the initial state estimate.

#### 2- Prediction Step
The Kalman filter prediction step was applied to estimate the future state of each object. This step utilized the system dynamics to predict the next state of the objects based on their previous states.

#### 3- Measurement Update Step
When new measurements became available from sensors or other sources, the Kalman filter measurement update step was employed to refine the state estimates. This step incorporated measurement information and updated the state estimates based on the Kalman gain, balancing prediction and measurement information.

#### 4- Data Association
To address the challenge of associating measurements with predicted object states, the Hungarian algorithm weas used for data association. These methods helped determine which measurement corresponded to which object.
