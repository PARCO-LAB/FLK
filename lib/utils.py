import numpy as np

def compute_distance(a,b):
    return np.sqrt( np.power(a[0]-b[0],2)+np.power(a[1]-b[1],2)+np.power(a[2]-b[2],2) )

def compute_velocity(now,old,dt):
    vel = []
    for j in range(0,len(now),3):
        dist = abs(compute_distance(now[j:j+3],old[j:j+3]))
        vel.append(float(dist)/dt)
    return vel

