from scipy.signal import butter, filtfilt
import numpy as np

class BF:
    def __init__(self,latency,order,cutoff,fs) :
            self.latency = latency
            self.b, self.a = butter(N=order, Wn=cutoff, btype='low', analog=False, fs=fs)

class EMA:
    def __init__(self,alpha,prev) -> None:
        self.alpha = alpha
        self.prev = prev
    def correct(self,skeleton):
        val = np.array([e*self.alpha for e in skeleton])
        valold = np.array([(1-self.alpha)*e for e in self.prev])
        #return val+valold 
        res = []
        for i in range(len(val)):
            if np.isnan(valold[i]):
                res.append(skeleton[i])
            else:
                res.append(val[i]+valold[i])
        return res