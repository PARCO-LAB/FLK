import pandas as pd
import numpy as np
from FLK import FLK
from FLK import evaluate
import matplotlib.pyplot as plt

def main():
    
    # Load the file
    data = pd.read_csv("../openpose_valid.csv")
    
    refined = data.copy()
    
    # Keypoints number, name and order in the file
    keypoints = ["LShoulder","RShoulder","LElbow ","RElbow ","LWrist ","RWrist","LHip ","RHip ","LKnee","RKnee","LAnkle","RAnkle"]

    initialization_needed = True

    for k in range(1,data.shape[0]): # ,143):#
        skeleton = data.iloc[k,:].values[1:]
        
        # If there are no keypoints detected go to the next frame
        if np.all(np.isnan(skeleton)):
            #refined.iloc[k,:].values[1:] = skeleton
            initialization_needed = True
            try: flk.reset()
            except: pass
            continue
        
        # If it's the first time that a person come into the scene
        if initialization_needed:
            flk = FLK( fs=15, skeleton=skeleton, keypoints=keypoints, model_path=None ,\
                        latency=0, num_dimension=2, enable_bones=False, enable_lowpass_filter=True,
                        ema_filter_value=0.995)
            flk.latency = 0
            initialization_needed = False
            continue
        filtered_skeleton = flk.correct(skeleton)
        refined.iloc[k,1:] = filtered_skeleton
    
    flk.reset()
    
    refined.to_csv("../test_flk.csv",na_rep='nan')
    
if __name__ == "__main__":
    main()