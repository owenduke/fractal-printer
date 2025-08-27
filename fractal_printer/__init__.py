import numpy as np
import quaternion

def iterate_voxels(distance_function, levels, size=1, center = np.quaternion(0,0,0,0), cutoff = 0, kwargs = {}):
    origin = center - np.quaternion(0,1,1,1)*size/2
    steps = 3**levels
    resolution = size/steps
    mask = np.zeros((steps,steps,steps),dtype=np.int8)

    for n in range(levels-1,-1,-1):
        stride = 3**n
        span = (stride-1)//2
        points = range(span,steps-span,stride)
        for i in points:
            for j in points:
                for k in points:
                    if mask[i,j,k] == 0:
                        z = np.quaternion(0,i+1/2, j+1/2, k+1/2)*resolution + origin 
                        d = distance_function(z,**kwargs)
                        r = int(np.floor(np.abs(d)/np.sqrt(3)/resolution)-0.5)
                        if r >= span:
                            mask[i-span:i+span+1,j-span:j+span+1,k-span:k+span+1] = np.sign(d)
    return mask <= cutoff
