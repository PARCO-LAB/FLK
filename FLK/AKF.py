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
        self.wasnan = True

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
    num_dimension: int
    def __init__(self,fs,skeleton,keypoints,model_path, num_dimension):
        self.num_dimension = num_dimension
        
        # Initialize a Linear Kalman Filter for each coordinate of all joints
        self.lkf = [LKF(skeleton[i]) for i in range(self.num_dimension*len(keypoints))]
        
        # Each keypoints initialized with NaN is nan
        for i in range(self.num_dimension*len(keypoints)):
            if not np.isnan(skeleton[i]):
                self.lkf[i].wasnan = False
        
        self.fs = fs
        # Configure the prediction model
        if model_path:
            from .RNN import GRU
            self.model = GRU(model_path = model_path, len_size=64, names=keypoints,num_dimension=self.num_dimension)
            self.is_RNN_enabled = True
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
        return np.sqrt( sum([np.power(a[i]-b[i],2) for i in range(self.num_dimension)]) )

    def reset(self):
        if self.is_RNN_enabled:
            self.model.reset()

    def correct(self,skeleton):
        
        # First of all predict the current frame
        pred = self.predict()
        confidence = []

        # Then correct the input skeleton
        for j in range(0,len(self.lkf),self.num_dimension):
            v = abs(self.compute_distance(skeleton[j:j+self.num_dimension],self.old_skeleton[j:j+self.num_dimension]))/(1/self.fs)
            v_pred = abs(self.compute_distance(pred[j:j+self.num_dimension],self.old_skeleton[j:j+self.num_dimension]))/(1/self.fs)
            c =  1 / (self.alpha*(v**2)+1)
            confidence.append(c)
            for k in range(0,self.num_dimension):
                # If the confidence it's above the threshold or it's invalid
                if c < self.theta or np.isnan(c):
                    # If there is a keypoint in input but it's confidence is very low
                    if not np.isnan(skeleton[j+k]):
                        if self.lkf[j+k].wasnan:
                            self.lkf[j+k] = LKF(skeleton[j+k])
                        else:
                            # old = self.lkf[j+k].X
                            self.lkf[j+k].predict( pred[j+k], [(self.alpha-1)*np.exp(-self.alpha*v)+1])
                            if self.lkf[j+k].wasnan:
                                self.lkf[j+k].update(skeleton[j+k],0)
                                self.lkf[j+k].wasnan = False
                            else:
                                self.lkf[j+k].update(skeleton[j+k],[self.alpha*v**2])
                            # print(old,self.lkf[j+k].X)
                            skeleton[j+k] = self.lkf[j+k].get_output()
                    # If there isn't a keypoint
                    else:
                        # If before there wasn't any keypoints, keep it
                        if not self.lkf[j+k].wasnan:
                            self.lkf[j+k].predict( pred[j+k], [(self.alpha-1)*np.exp(-self.alpha*v_pred)+1])
                            skeleton[j+k] = self.lkf[j+k].get_output()
                            self.lkf[j+k].wasnan = True
                # Keep the keypoint in input as is
                else:
                    if self.lkf[j+k].wasnan:
                        self.lkf[j+k] = LKF(skeleton[j+k])
                        self.lkf[j+k].wasnan = False
                    self.lkf[j+k].X = np.array(skeleton[j+k])
        self.model.append(skeleton.copy())
        return skeleton