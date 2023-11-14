from typing import Dict, Optional, List, Any
import numpy as np
import numpy.typing as npt

from .AKF import AKF
from .BCA import BCA
from .LPF import EMA

Skeleton = npt.NDArray[Any] or List[int or float]

class FakeDict(object):
    _dict: Dict[Any, Any]
    def __init__(self):
        object.__setattr__(self, '_dict', {})
    
    def __getattribute__(self, __name: str) -> Any:
        d = object.__getattribute__(self, '_dict')
        return d[__name]
    
    def __setattr__(self, __name: str, __value: Any) -> None:
        d = object.__getattribute__(self, '_dict')
        d[__name] = __value
        
class FLK:
    akf: AKF
    bca: Optional[BCA] = None
    lpf: Optional[EMA] = None
    num_dimension: int

    extra_data: Any
    
    def __init__(self,
                 fs : float,
                 skeleton: Skeleton,
                 keypoints: List[str],
                 model_path: Optional[str]=None,
                 latency: float=0,
                 enable_bones: bool=True,
                 enable_lowpass_filter: bool=True,
                 ema_filter_value: float=0.99,
                 num_dimension: int = 3):
        skeleton = skeleton.ravel() if type(skeleton) is np.ndarray else skeleton
        self.akf = AKF(fs,skeleton,keypoints,model_path, num_dimension)
                
        # Default latency is set to 0
        self.latency = latency
        self.keypoints = keypoints
        if enable_bones:
            self.bca = BCA(skeleton,keypoints, num_dimension)
        
        if enable_lowpass_filter:
            self.lpf = EMA(ema_filter_value,skeleton)

        # Activate LPF
        if self.latency > 0:
            pass

        self.num_dimension = num_dimension
        self.extra_data = FakeDict()

    def reset(self):
        self.akf.reset()
        if self.bca is not None:
            self.bca.reset()

    def predict(self):
        skeleton =  self.akf.predict()
        if self.bca is not None:
            skeleton = self.bca.correct(skeleton,self.keypoints)
        return skeleton

    def correct(self,skeleton: Skeleton):
        
        filtered = self.akf.correct(skeleton)

        if self.bca is not None:
            filtered = self.bca.correct(filtered,self.keypoints)

        
        if self.lpf is not None:
            filtered = self.lpf.correct(filtered)
        
        if self.latency == 0:
            self.akf.old_skeleton = filtered
            return filtered
        
        else:
            pass
    
    def correct_np(self,skeleton_np: npt.NDArray[Any]):

        skeleton = skeleton_np.ravel()
        
        filtered = self.akf.correct(skeleton)

        if self.bca is not None:
            filtered = self.bca.correct(filtered,self.keypoints)

        if self.lpf is not None:
            filtered = self.lpf.correct(filtered)

        if self.latency == 0:
            self.akf.old_skeleton = filtered
            return filtered.reshape(12,self.num_dimension)
        
        else:
            pass