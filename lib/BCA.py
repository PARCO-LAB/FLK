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

    def append(self,l):
        self.lengths.append(l) 

        # At least 20 samples with a variance less than 2 cm
        if self.length or ( len(self.lengths) > 20 and np.std(np.array(self.lengths), axis=0) < 0.02 ):
            self.length = np.mean(np.array(self.lengths), axis=0)

class DAG():
    
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

    def update_bone_length(self,s,names):
        for b in self.bones:
            try:
                i1 = [idx for idx, l in enumerate(names) if b.from_kp in l][0]
                i2 = [idx for idx, l in enumerate(names) if b.to_kp in l][0]
                b.append(compute_distance(s[3*i1:3*i1+3],s[3*i2:3*i2+3]))
            except:
                pass
    
    def correct(self):
        pass

class BCA():
    
    def __init__(self,skeleton, keypoints):
        self.DAG = DAG(skeleton,keypoints)
        print([b.lengths for b in self.DAG.bones])
    
    