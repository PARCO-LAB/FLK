import numpy as np
from lib.RNN import GRU
from lib.utils import compute_distance

class LKF():
    def __init__(self,s):
        self.X = np.array([[s]])
        self.P = np.array([[1]])
        self.F = np.array([[1]])
        self.Q = np.eye(self.X.shape[0])*0.05
        self.Y = np.array([s])
        self.H = np.array([1]).reshape(1,1)
        self.R = 1 
        self.K = 0.5
        self.wasnan = False

    def predict(self,next,Q=0.05):
        self.Q = Q
        self.X = next
        self.P = np.dot(self.F, np.dot(self.P,self.F.T)) + self.Q

    def update(self,Y,R):
        self.Y = Y
        self.R = R
        self.K = np.dot(self.P,self.H.T) / ( R + np.dot(self.H,np.dot(self.P,self.H.T))) 
        self.X = self.X + self.K * ( self.Y - np.dot(self.H,self.X))
        self.P = np.dot((np.eye(self.X.shape[0])- np.dot(self.K,self.H)),self.P)
        self.Y = float(np.dot(self.H,self.X))

    def get_output(self):
        return float(np.dot(self.H,self.X))


class AKF():
    
    def __init__(self,skeleton,keypoints,model_path):

        # Initialize a Linear Kalman Filter for each coordinate of all joints
        self.lkf = [LKF(skeleton[i]) for i in range(3*len(keypoints))]
        
        # Configure the prediction model
        if model_path:
            self.model = GRU(model_path = model_path, len_size=64, names=keypoints)
            self.is_RNN_enabled = True
            self.model.append(skeleton)
        else:
            self.is_RNN_enabled = False

        self.alpha = 0.01
        self.theta = 0.75

    def predict(self):
        if self.is_RNN_enabled:
            next, has_predicted = self.model.predict()
            next = next.reshape( (self.model.model_to_generic.shape[0],))
            return next
        else:
            return None        

    def correct(self,old_skeleton):
        for j in range(0,len(self.lkf),3):
            v = abs(compute_distance(out[j:j+3],old_skeleton[j:j+3]))/dt
            c =  1 / (self.alpha*( v**2)+1)
