import numpy as np
"""
Averaging several frames together reduces the impact of readout
noise and gives a more accurate estimate of the bias level. 
The master bias frame produced from this averaging - Dealing with CCD Data

Frame combine Steps:
    - Takes 2D arrays (currently takes them written out)
    - Stacks arrays on 3rd axis (now 3d matrix)
    - Takes element-wise mean of each array
    - Returns 2d array of each element-wise averages
"""

def frame_combine(file_list):
    data=np.dstack(file_list)
    #assuming all data will be 2D arrays
    master_frame=np.mean(data,2)
    return master_frame