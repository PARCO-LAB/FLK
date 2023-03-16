import pandas as pd
import numpy as np
from lib.FLK import FLK
from lib.utils import evaluate
import matplotlib.pyplot as plt

def main():
    
    # Keypoints number, name and order in the file
    keypoints = ["Hip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","Spine","Neck","Nose",
                 "Head","LShoulder","LElbow","LWrist","RShoulder","RElbow","RWrist"]

    # Load the file
    data = pd.read_csv("data/S9_Walking_Ray3D_CPN_cam1.csv")
    refined = data.copy()
    ground_truth = pd.read_csv("data/S9_Walking_ground_truth.csv")

    evaluate(data,ground_truth)

    # Get the first skeleton of the sequence
    first_skeleton = data.iloc[0,:].values[1:]

    # Initialize FLK
    flk = FLK( fs=50, skeleton=first_skeleton, keypoints=keypoints, model_path="models/GRU.h5" ,latency=0)
    #flk.AKF.is_RNN_enabled = False
    flk.latency = 0

    for k in range(1, data.shape[0]):
        skeleton = data.iloc[k,:].values[1:]
        if np.any(skeleton):
            # It is robust to NaN (flk.correct comprehends the prediction)
            filtered_skeleton = flk.correct(skeleton)
        else:
            filtered_skeleton = flk.predict()
        
        refined.iloc[k,:].values[1:] = filtered_skeleton
    
    flk.reset()

    evaluate(refined,ground_truth)
    
    plt.plot( data.iloc[:,0].values,data.iloc[:,1].values)
    plt.plot( data.iloc[:,0].values,refined.iloc[:,1].values)
    plt.show()
    refined.to_csv("data/S9_Walking_output.csv")

if __name__ == "__main__":
    main()