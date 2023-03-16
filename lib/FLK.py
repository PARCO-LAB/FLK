from lib.AKF import AKF
from lib.BCA import BCA
from lib.LPF import EMA

class FLK:
    
    def __init__(self,fs,skeleton,keypoints,model_path=None,latency=0):

        self.AKF = AKF(fs,skeleton,keypoints,model_path)
                
        # Default latency is set to 0
        self.latency = latency
        self.keypoints = keypoints
        self.BCA = BCA(skeleton,keypoints)
        
        self.LPF = EMA(0.99,skeleton)

        # Activate LPF
        if self.latency > 0:
            pass

    def reset(self):
        self.AKF.reset()
        self.BCA.reset()

    def predict(self):
        skeleton =  self.AKF.predict()
        after_BCA = self.BCA.correct(skeleton,self.keypoints)
        return after_BCA

    def correct(self,skeleton):
        
        
        filtered = self.AKF.correct(skeleton)

        filtered = self.BCA.correct(filtered,self.keypoints)

        filtered = self.LPF.correct(filtered)

        if self.latency == 0:
            self.AKF.old_skeleton = filtered
            return filtered
        
        else:
            pass