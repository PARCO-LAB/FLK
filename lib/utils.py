import numpy as np

def find_common_keypoints(ref,src):
    return list(set([e.split(':')[0] for e in ref.columns.intersection(src.columns).tolist() if e != 'time' and e != 'Unnamed: 0' ]))

def calculate_mpjpe(ref, src,kps):
    # Build the N*|kps|*3 matrices
    N = ref.shape[0]
    R = np.zeros(shape=(N, len(kps),3))
    S = np.zeros(shape=(N, len(kps),3))
    
    for i in range(len(kps)):
        R[:,i,0] = ref[kps[i]+':X'].values
        R[:,i,1] = ref[kps[i]+':Y'].values
        R[:,i,2] = ref[kps[i]+':Z'].values
        S[:,i,0] = src[kps[i]+':X'].values
        S[:,i,1] = src[kps[i]+':Y'].values
        S[:,i,2] = src[kps[i]+':Z'].values
    
    mpjpe = np.mean(np.sqrt(np.sum(np.square(S-R), axis=2)))

    return mpjpe

def evaluate(input, gt):
    kps = find_common_keypoints(input,gt)
    kps =  ['RKnee', 'LWrist', 'RHip', 'RShoulder',  'LElbow', 'LHip', 'RElbow', 'RWrist', 'LKnee', 'LShoulder', 'RAnkle', 'LAnkle']
    mpjpe = calculate_mpjpe(input,gt,kps)
    print(round(mpjpe*1000,2))