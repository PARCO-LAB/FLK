import pandas as pd

from FLK import FLK

def main():
    
    # Keypoints number, name and order in the file
    keypoints = ["Hip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","Spine","Neck","Nose",
                 "Head","LShoulder","LElbow","LWrist","RShoulder","RElbow","RWrist"]

    # Load the file
    data = pd.read_csv("data/S9_Walking.csv")

    # Get the first skeleton of the sequence
    first_skeleton = data.iloc[0,:].values[1:]

    # Initialize FLK
    flk = FLK( skeleton=first_skeleton, keypoints=keypoints, model_path="models/GRU" )
    flk.latency = 0

    #for k in range(1,len(data.shape[0])):
    print(flk.predict())


if __name__ == "__main__":
    main()
    