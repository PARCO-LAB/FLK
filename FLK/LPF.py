from scipy.signal import butter, filtfilt, lfilter
import numpy as np

# class BF:
#     def __init__(self,latency,order,cutoff,fs) :
#         order = 4
#         cutoff_frequency = 5.0
#         sampling_frequency = 100.0
#         self.latency = latency
#         self.b, self.a = butter(order, cutoff_frequency / (sampling_frequency / 2), btype='low')
    
#     def correct(self,skeleton):
#         val = filtfilt(self.b, self.a, skeleton)
        
        
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
        self.prev = res
        return res
    
