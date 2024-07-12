import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

import warnings
warnings.filterwarnings("ignore")

import glob

path = 'data/aist/train/gt' # trained on ground truth


OP_H36M_NAMES = ['']*12

OP_H36M_NAMES[0]  = 'RHip'
OP_H36M_NAMES[1]  = 'RKnee'
OP_H36M_NAMES[2]  = 'RAnkle'
OP_H36M_NAMES[3]  = 'LHip'
OP_H36M_NAMES[4]  = 'LKnee'
OP_H36M_NAMES[5]  = 'LAnkle'
OP_H36M_NAMES[6]  = 'LShoulder'
OP_H36M_NAMES[7]  = 'LElbow'
OP_H36M_NAMES[8]  = 'LWrist'
OP_H36M_NAMES[9] = 'RShoulder'
OP_H36M_NAMES[10] = 'RElbow'
OP_H36M_NAMES[11] = 'RWrist'


def center_on_hip_first(df : pd.DataFrame, kp_center = 'Hip'):
    columns = df.columns
    df.reset_index(inplace = True)
    row = df.loc[0]
    
        #find MidHip
    RHip = [row['R' + kp_center +':X'],row['R' + kp_center +':Y'],row['R' + kp_center +':Z']]
    LHip = [row['L' + kp_center +':X'],row['L' + kp_center +':Y'],row['L' + kp_center +':Z']]
    MidHip = [(RHip[0] + LHip[0])/2, (RHip[1] + LHip[1])/2, (RHip[2] + LHip[2])/2]
    for i in range(len(df)):
        row = df.loc[i]
        for j in columns:
            if j.split(':')[-1] == 'X':
                row[j] = row[j] - MidHip[0]
            elif j.split(':')[-1] == 'Y':
                row[j] = row[j] - MidHip[1]
            elif j.split(':')[-1] == 'Z':
                row[j] = row[j] - MidHip[2]

        
        df.loc[i] = row

    return df

def load_from_csv(files : list, timesteps : int, scaler : StandardScaler = None, sample = None):
    
    final_X = []
    final_y = []
    for filename in files:
        print("processing",filename)
        df = pd.read_csv(filename)
        if 'time' in df:
            df = df.drop(columns = ['time'])
        X = []
        y = []
        columns_full = list(df.columns)
        to_drop = []
        for i in columns_full:
            if i.split(':')[0] not in OP_H36M_NAMES:
                to_drop.append(i)

        df = df.drop(columns = to_drop)
        if not scaler:
            for j in range(len(df) - timesteps):
                in_seq = df[j: j + timesteps].values.tolist()
                out = df.loc[j + timesteps].values.tolist()
                X.append(in_seq)
                y.append(out)
            X = np.array(X)
            y = np.array(y)
        else:
            df_values = df.values
            
            for j in range(len(df) - timesteps):
                sequenza = df[j: j + timesteps + 1]
                sequenza = center_on_hip_first(sequenza)
                sequenza = sequenza.drop(columns=['index'])
                s_mean = np.nanmean(sequenza)
                s_std = np.nanstd(sequenza)
                sequenza = (sequenza - s_mean)/s_std
                in_seq = sequenza[:-1].values.tolist()
                out = sequenza.values.tolist()[-1]
                X.append(in_seq)
                y.append(out)
            
            if len(X) > sample:
                subset_index = np.random.choice(list(range(0, len(X))),size = sample, replace = False)
                X = [X[i] for i in subset_index]
                y = [y[i] for i in subset_index]

            final_X += X
            print(np.array(final_X).shape)
            final_y += y

    subset_index = np.random.choice(list(range(0, len(final_X))),size = len(final_X), replace = False)
    final_X = [final_X[i] for i in subset_index]
    final_y = [final_y[i] for i in subset_index]
    X = np.array(final_X)
    y = np.array(final_y)


    split_index = int(len(X) * 0.8)  # 80% training, 20% validation
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    if scaler:
        return X_train, X_val, y_train, y_val, scaler
    else:
        return X_train, X_val, y_train, y_val, None


scaler = StandardScaler()


files = glob.glob(path + '/*.csv')

#files = files[:74]
files_final=[]
for i in files:
    if 'S9' in i or 'S11' in i:
        pass
    else:
        files_final.append(i)

length = 64
_sample = 10
dataset = 'aist'
X_train, X_val, y_train, y_val, scaler = load_from_csv(files_final,length, scaler, sample = _sample)

np.save('dataset/training_FLK/' + dataset + '/X_train_' + str(length) + '_' + str(_sample)+  '.npy', X_train)
np.save('dataset/training_FLK/' + dataset + '/X_val_' + str(length) + '_' + str(_sample)+  '.npy', X_val)
np.save('dataset/training_FLK/' + dataset + '/y_train_' + str(length) + '_' + str(_sample)+  '.npy', y_train)
np.save('dataset/training_FLK/' + dataset + '/y_val_' + str(length) + '_' + str(_sample)+  '.npy', y_val)