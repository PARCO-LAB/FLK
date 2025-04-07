import pandas as pd
import numpy as np
from FLK import FLK
from FLK import evaluate
import matplotlib.pyplot as plt

def main():
    
    # Keypoints number, name and order in the file
    # keypoints = ["Hip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","Spine","Neck","Nose",
    #              "Head","LShoulder","LElbow","LWrist","RShoulder","RElbow","RWrist"]
    
    keypoints = ["RHip","RKnee","RAnkle","RFoot","LHip","LKnee","LAnkle","LFoot","Neck","Head","LShoulder","LElbow","LWrist","RShoulder","RElbow","RWrist"]

    # Load the file
    data = pd.read_csv("data/S9_Walking_random25.csv").sort_values(by='time')
    refined = data.copy(deep=True)
    
    ground_truth = pd.read_csv("data/S9_Walking_ground_truth.csv")
    plt.plot( data.iloc[:,0].values,data.iloc[:,1].values)
    
    # Get the first skeleton of the sequence
    first_skeleton = data.iloc[0,:].values[1:]

    # Initialize FLK
    flk = FLK( fs=50, skeleton=first_skeleton, keypoints=keypoints, model_path="models/GRU.h5" ,latency=0)
    flk.akf.is_RNN_enabled = False
    # flk.latency = 0

    for k in range(1, refined.shape[0]):
        skeleton = refined.iloc[k,:].values[1:]
        # print(skeleton)
        # exit()
        if np.any(skeleton):
            # It is robust to NaN (flk.correct comprehends the prediction)
            filtered_skeleton = flk.correct(skeleton)
        else:
            filtered_skeleton = flk.predict()
        refined.iloc[k,:].values[1:] = filtered_skeleton
    flk.reset()


    
    print("input:   ",evaluate(data,ground_truth,'3D'))
    print("refined: ",evaluate(refined,ground_truth,'3D'))
    
    plt.plot( refined.iloc[:,0].values,refined.iloc[:,1].values)
    plt.plot( ground_truth.iloc[:,0].values,ground_truth.iloc[:,1].values)
    plt.savefig('figure.png')
    refined.to_csv("data/S9_Walking_output.csv")

if __name__ == "__main__":
    main()