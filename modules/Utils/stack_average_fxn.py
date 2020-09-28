import numpy as np
"""
Averaging several frames together reduces the impact of readout
noise and gives a more accurate estimate of the bias level. 
The master bias frame produced from this averaging - Dealing with CCD Data

Stack Average Steps:
    1. Takes 2D arrays (currently takes them written out)
    2. Stacks arrays on 3rd axis (now 3d matrix)
    3. Takes element-wise mean of each array
    4. Returns 2d array of each element-wise averages

"""
def stack_average(flat_file_list):
    data=np.dstack(flat_file_list)
    #assuming all data will be 2D arrays
    mean_data=np.mean(data,2)
    return mean_data