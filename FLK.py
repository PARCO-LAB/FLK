from lib.AKF import AKF
from lib.BCA import BCA

class FLK:
    
    def __init__(self,fs,skeleton,keypoints,model_path=None):

        self.AKF = AKF(fs,skeleton,keypoints,model_path)
                
        # Default latency is set to 0
        self.latency = 0
        self.keypoints = keypoints
        self.BCA = BCA(skeleton,keypoints)
        
    def predict(self):
        return self.AKF.predict()

    def correct(self,skeleton):
        
        
        filtered = self.AKF.correct(skeleton)

        after_BCA = self.BCA.correct(filtered,self.keypoints)

        if self.latency == 0:
            self.AKF.old_skeleton = after_BCA
            return after_BCA
        
        else:
            print("In this simple demo the LPF version is not included.")