import os
import numpy as np
import keras

# Disable boring logging of keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#keras.utils.disable_interactive_logging()

class GRU():
    
    def __init__(self,model_path,len_size,names):
        self.history = []
        self.raw = []
        self.window = len_size        
        self.model = keras.models.load_model(model_path)
        #print(self.model.summary())
        self.keypoints = ['RHip','RKnee','RAnkle','LHip','LKnee','LAnkle','LShoulder','LElbow','LWrist','RShoulder','RElbow','RWrist']
        
        # Create conversion maps from the input model to H36M13
        self.generic_to_model = np.zeros((len(self.keypoints)*3,1))-1
        self.model_to_generic = np.zeros((3*len(names),1))-1
        for i in range(len(self.keypoints)):
            for j in range(len(names)):
                if names[j] == self.keypoints[i]:
                    self.generic_to_model[3*i,0] = 3*j
                    self.generic_to_model[3*i+1,0] = 3*j+1
                    self.generic_to_model[3*i+2,0] = 3*j+2

                    self.model_to_generic[3*j,0] = 3*i
                    self.model_to_generic[3*j+1,0] = 3*i+1
                    self.model_to_generic[3*j+2,0] = 3*i+2

    def remove_hip_center(self):
        sk = self.history[0]
        # Find right Hip and left Hip
        rx = 0
        lx = 9
        cx = (sk[rx]+sk[lx]) / 2
        cy = (sk[rx+1]+sk[lx+1]) / 2
        cz = (sk[rx+2]+sk[lx+2]) / 2
        
        self.hip = np.transpose(np.array([cx,cy,cz]))
        mask = np.tile(self.hip,(self.window,int(len(sk)/3)))
        data = np.array(self.history).reshape((self.window,36))-mask
        
        self.mu = np.mean(data)
        self.sd = np.std(data)

        self.data = (data - self.mu) / self.sd

        
    def predict(self):
        
        if len(self.history) == self.window:

            input = self.data.reshape(1,self.data.shape[0], self.data.shape[1])
            
            y = self.model(input).numpy()
            y = y.reshape(y.shape[1],1)
            y = y *self.sd + self.mu
            y = self.translate(y,self.hip)

            return self.transform(y,self.model_to_generic,self.raw[-1]), True
        
        else:
            out = np.array(self.raw[-1]) 
            return out, False

    def reset(self):
            self.history = []
            self.raw = []


    def translate(self,sk,p):
        new_sk = np.zeros(sk.shape)
        for i in range(0,len(sk),3):
            new_sk[i]   = sk[i] + p[0,0]
            new_sk[i+1] = sk[i+1] + p[0,1]
            new_sk[i+2] = sk[i+2] + p[0,2]
        return new_sk

    def transform(self,sk,conv, default=None):
        new = np.full(conv.shape, np.nan)
        for i in range(conv.shape[0]):
            if int(conv[i]) != -1:
                new[i,0] = sk[int(conv[i])]
            else:
                new[i,0] = default[i]
        return new

    def append(self,e):
        self.raw.append(e)
        new_e = self.transform(e,self.generic_to_model)
        self.history.append(new_e)
        if len(self.history) == self.window:
            self.remove_hip_center()
        if len(self.history) > self.window:
            self.history.pop(0)
            self.raw.pop(0)
            self.remove_hip_center()
            
    def reset(self):
        self.history = []
        self.raw = []
