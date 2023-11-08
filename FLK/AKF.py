from typing import List
import numpy as np

class FakeModel():
    raw: List[int]
    def __init__(self) -> None:
        self.raw = []
    
    def reset(self):
        self.raw = []
    
    def append(self, x):
        self.raw.append(x)
    
    def __len__(self):
        return len(self.raw)
    def __getitem__(self, i):
        return self.raw[i]
        
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
    
    def __init__(self,fs,skeleton,keypoints,model_path):
        
        # Initialize a Linear Kalman Filter for each coordinate of all joints
        self.lkf = [LKF(skeleton[i]) for i in range(3*len(keypoints))]
        self.fs = fs
        # Configure the prediction model
        if model_path:
            from .RNN import GRU
            self.model = GRU(model_path = model_path, len_size=64, names=keypoints)
            self.is_RNN_enabled = True
            self.model.append(skeleton)
        else:
            self.model = FakeModel()
            self.is_RNN_enabled = False
        self.model.append(skeleton)

        self.alpha = 0.01
        self.theta = 0.75
        self.old_skeleton = skeleton.copy()

    def predict(self):
        if self.is_RNN_enabled:
            next, has_predicted = self.model.predict()
            next = next.reshape( (self.model.model_to_generic.shape[0],))
            return next
        else:
            return self.model.raw[-1]        

    def compute_distance(self,a,b):
        return np.sqrt( np.power(a[0]-b[0],2)+np.power(a[1]-b[1],2)+np.power(a[2]-b[2],2) )

    def reset(self):
        if self.is_RNN_enabled:
            self.model.reset()

    def correct(self,skeleton):

        # First of all predict the current frame
        pred = self.predict()
        confidence = []

        out = skeleton.copy()

        # Then correct the input skeleton with the 
        for j in range(0,len(self.lkf),3):
            v = abs(self.compute_distance(skeleton[j:j+3],self.old_skeleton[j:j+3]))/(1/self.fs)
            v_pred = abs(self.compute_distance(pred[j:j+3],self.old_skeleton[j:j+3]))/(1/self.fs)
            c =  1 / (self.alpha*( v**2)+1)
            confidence.append(c)
            for k in range(0,3):
                if c < self.theta or np.isnan(c):
                    if not np.isnan(skeleton[j+k]):
                        self.lkf[j+k].predict( pred[j+k], [(self.alpha-1)*np.exp(-self.alpha*v)+1])
                        if self.lkf[j+k].wasnan:
                            self.lkf[j+k].update(skeleton[j+k],0)
                            self.lkf[j+k].wasnan = False
                        else:
                            self.lkf[j+k].update(skeleton[j+k],[self.alpha*v**2])
                    else:
                        self.lkf[j+k].predict( pred[j+k], [(self.alpha-1)*np.exp(-self.alpha*v_pred)+1])
                        self.lkf[j+k].wasnan = True
                    skeleton[j+k] = self.lkf[j+k].get_output()
                else:
                    self.lkf[j+k].X = np.array(skeleton[j+k])


        self.model.append(skeleton.copy())
        return skeleton
