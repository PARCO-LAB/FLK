from lib.AKF import AKF
from lib.BCA import BCA

class FLK:
    
    def __init__(self,skeleton,keypoints,model_path=None):

        self.AKF = AKF(skeleton,keypoints,model_path)
                
        # Default latency is set to 0
        self.latency = 0

        self.BCA = BCA(skeleton,keypoints)
        
    def predict(self):
        return self.AKF.predict()

    def correct(self):
        pass