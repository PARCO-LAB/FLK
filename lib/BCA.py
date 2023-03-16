from lib.utils import *


class Node():
    def __init__(self,name,pos):
        self.name = name
        self.pos = pos


class Link():
    def __init__(self,name,from_kp,to_kp,type="fixed"):
        self.name = name
        self.from_kp = from_kp
        self.to_kp = to_kp
        self.length = None
        self.lengths = []

    def reset(self):
        self.lengths = []
        self.length = None

    def append(self,l):
        self.lengths.append(l) 

        # At least 20 samples with a variance less than 2 cm
        if self.length or ( len(self.lengths) > 20 and np.std(np.array(self.lengths), axis=0) < 0.02 ):
            self.length = np.mean(np.array(self.lengths), axis=0)

class DAG():
    
    def compute_distance(self,a,b):
        return np.sqrt( np.power(a[0]-b[0],2)+np.power(a[1]-b[1],2)+np.power(a[2]-b[2],2) )

    def __init__(self, skeleton, keypoints):
        
        # For each keypoint create a node ad set the initial position
        self.joints = [ Node(keypoints[i],skeleton[3*i:3*i+3]) for i in range(len(keypoints))]

        self.bones = []
        self.bones.append(Link("RHumerus","RShoulder","RElbow"))
        self.bones.append(Link("RForearm","RElbow","RWrist"))
        self.bones.append(Link("LHumerus","LShoulder","LElbow"))
        self.bones.append(Link("LForearm","LElbow","LWrist"))
        self.bones.append(Link("RFemur"  ,"RHip","RKnee"))
        self.bones.append(Link("RTibia"  ,"RKnee","RAnkle"))
        self.bones.append(Link("RFoot"   ,"RAnkle","RFoot"))
        self.bones.append(Link("LFemur"  ,"LHip","LKnee"))
        self.bones.append(Link("LTibia"  ,"LKnee","LAnkle"))
        self.bones.append(Link("LFoot"   ,"LAnkle","LFoot"))

        self.update_bone_length(skeleton,keypoints)

    def reset(self):
        for b in self.bones:
            b.reset()

    def update_bone_length(self,s,names):
        for b in self.bones:
            try:
                i1 = [idx for idx, l in enumerate(names) if b.from_kp in l][0]
                i2 = [idx for idx, l in enumerate(names) if b.to_kp in l][0]
                b.append(self.compute_distance(s[3*i1:3*i1+3],s[3*i2:3*i2+3]))
            except:
                pass
    


class BCA():
    def __init__(self,skeleton, keypoints):
        self.DAG = DAG(skeleton,keypoints)
        self.epsilon = 0.01 # bone-length error treshold
        
    def reset(self):
        self.DAG.reset()

    def correct(self,s,names):
        
        # Update with the current measurement
        self.DAG.update_bone_length(s,names)
        
        # Check if there is anything to correct
        for b in self.DAG.bones:
            if b.length and abs(b.length - b.lengths[-1])/b.length > self.epsilon:
                a_i = names.index(b.from_kp)
                b_i = names.index(b.to_kp)
                A = s[3*a_i:3*a_i+3]
                B = s[3*b_i:3*b_i+3]
                s[3*b_i:3*b_i+3] = b.length * (B-A) / np.linalg.norm(B-A) + A
        
        return s           
