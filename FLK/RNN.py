import os
import numpy as np
import keras
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# Disable boring logging of keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#keras.utils.disable_interactive_logging()

class GRU():
    num_dimension: int
    def __init__(self,model_path,len_size,names, num_dimension):
        self.num_dimension = num_dimension
        self.history = []
        self.raw = []
        self.window = len_size        
        self.model = keras.models.load_model(model_path)
        #print(self.model.summary())
        self.keypoints = ['RHip','RKnee','RAnkle','LHip','LKnee','LAnkle','LShoulder','LElbow','LWrist','RShoulder','RElbow','RWrist']
        
        # Create conversion maps from the input model to H36M13
        self.generic_to_model = np.zeros((len(self.keypoints)*self.num_dimension,1))-1
        self.model_to_generic = np.zeros((self.num_dimension*len(names),1))-1
        for i in range(len(self.keypoints)):
            for j in range(len(names)):
                if names[j] == self.keypoints[i]:
                    for idx in range(self.num_dimension):
                        self.generic_to_model[self.num_dimension*i+idx,0] = self.num_dimension*j+idx
                        self.model_to_generic[self.num_dimension*j+idx,0] = self.num_dimension*i+idx

    def remove_hip_center(self):
        sk = self.history[0]
        # Find right Hip and left Hip
        rx = 0
        lx = 9

        self.hip = np.transpose(np.array([(sk[rx+1]+sk[lx+1]) / 2 for i in range(self.num_dimension)]))
        mask = np.tile(self.hip,(self.window,int(len(sk)/self.num_dimension)))
        # TODO: parametrize by joint numbers?
        data = np.array(self.history).reshape((self.window,12 * self.num_dimension))-mask
        
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
        for i in range(0,len(sk),self.num_dimension):
            for j in range(self.num_dimension):
                new_sk[i+j] = sk[i+j] + p[0,j]
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
