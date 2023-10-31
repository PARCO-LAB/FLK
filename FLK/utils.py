import numpy as np

def find_common_keypoints(ref,src):
    return list(set([e.split(':')[0] for e in ref.columns.intersection(src.columns).tolist() if e != 'time' and e != 'Unnamed: 0' and e != 'frame' ]))

def calculate_error(ref, src,kps, dim):
    # Build the N*|kps|*3 matrices
    N = min(ref.shape[0],src.shape[0])
    R = np.zeros(shape=(N, len(kps),dim))
    S = np.zeros(shape=(N, len(kps),dim))
    
    if dim == 3:
        for i in range(len(kps)):
            R[:,i,0] = ref[kps[i]+':X'].values[:N]
            R[:,i,1] = ref[kps[i]+':Y'].values[:N]
            R[:,i,2] = ref[kps[i]+':Z'].values[:N]
            S[:,i,0] = src[kps[i]+':X'].values[:N]
            S[:,i,1] = src[kps[i]+':Y'].values[:N]
            S[:,i,2] = src[kps[i]+':Z'].values[:N]
    elif dim == 2:
        for i in range(len(kps)):
            R[:,i,0] = ref[kps[i]+':U'].values[:N]
            R[:,i,1] = ref[kps[i]+':V'].values[:N]
            S[:,i,0] = src[kps[i]+':U'].values[:N]
            S[:,i,1] = src[kps[i]+':V'].values[:N]

    missings = (100*np.count_nonzero(np.isnan(S))/dim)  / (N*len(kps))
    mpjpe = np.nanmean(np.sqrt(np.sum(np.square(S-R), axis=2)))

    # (N-2)x14x3
    accel_gt = R[:-2] - 2 * R[1:-1] + R[2:]
    accel_pred = S[:-2] - 2 * S[1:-1] + S[2:]

    acc=np.nanmean(np.linalg.norm(accel_pred - accel_gt, axis=2), axis=1)
    
    acc =  np.nanmean(acc)

    return mpjpe, acc, missings

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

def evaluate(input, gt, type = None):
    kps = find_common_keypoints(input,gt)
    if type is None:
        # TODO: kept old behaviour, to update with the correct value
        kps =  ['RKnee', 'LWrist', 'RHip', 'RShoulder',  'LElbow', 'LHip', 'RElbow', 'RWrist', 'LKnee', 'LShoulder', 'RAnkle', 'LAnkle']
        return calculate_mpjpe(input,gt,kps)
    elif type == "3D":
        mpjpe, accel, missings = calculate_error(gt,input,kps, 3)
        return round(mpjpe*1000,2), round(accel*1000,1), int(missings)
    elif type == "2D":
        mpjpe, accel, missings = calculate_error(gt,input,kps, 2)
        return round(mpjpe,2), round(accel,1), int(missings)