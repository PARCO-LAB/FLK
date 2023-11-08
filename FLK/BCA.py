from .utils import *


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
    num_dimension: int
    def compute_distance(self,a,b):
        # return np.sqrt( np.power(a[0]-b[0],2)+np.power(a[1]-b[1],2)+np.power(a[2]-b[2],2) )
        return np.sqrt( sum([np.power(a[i]-b[i],2) for i in range(self.num_dimension)]) )

    def __init__(self, skeleton, keypoints, num_dimension):
        self.num_dimension = num_dimension
        
        # For each keypoint create a node ad set the initial position
        self.joints = [ Node(keypoints[i],skeleton[num_dimension*i:num_dimension*i+num_dimension]) for i in range(len(keypoints))]

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
                b.append(self.compute_distance(s[self.num_dimension*i1:self.num_dimension*i1+self.num_dimension],s[self.num_dimension*i2:self.num_dimension*i2+self.num_dimension]))
            except:
                pass
    


class BCA():
    num_dimension: int
    def __init__(self,skeleton, keypoints, num_dimension):
        self.DAG = DAG(skeleton,keypoints, num_dimension)
        self.epsilon = 0.01 # bone-length error treshold
        self.num_dimension = num_dimension
        
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
                A = s[self.num_dimension*a_i:self.num_dimension*a_i+self.num_dimension]
                B = s[self.num_dimension*b_i:self.num_dimension*b_i+self.num_dimension]
                s[self.num_dimension*b_i:self.num_dimension*b_i+self.num_dimension] = b.length * (B-A) / np.linalg.norm(B-A) + A
        
        return s           
