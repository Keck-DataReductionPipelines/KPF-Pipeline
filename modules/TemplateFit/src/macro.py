
#
# -- Template fitting algorithm macros --
# This file stores datatypes, constants, and common helper functions 
# for the template fitting algorithms. 
import typing as tp 
import numpy as np

import os
import sys

ECHELLE_SHAPE = [72, 4096]
C_SPEED = 2.99792458e5 # [km/s] speed of light
SEC_ACC = 4.49e-3 # [km/s^2] acceleration


# Data type
EchelleData_TYPE = tp.NewType('EchelleData_TYPE', np.ndarray)
EchellePair_TYPE = tp.Tuple[EchelleData_TYPE, EchelleData_TYPE]

# Result type
ALPHA_TYPE = tp.NewType('alpha_TYPE', np.ndarray)

ord_range = [i for i in range(71) if i > 24 and i != 57 and i != 66 ]

# Some helpful general purpose function

def findfiles(fpath, extension):
    '''
    find all the files in the sub directories with relevant extension
    '''
    lst_fname = []
    for dirpath,_, filenames in os.walk(fpath):
        for filename in [f for f in filenames if f.endswith(extension)]:
            lst_fname.append(os.path.join(dirpath, filename))
    return lst_fname


def common_range(x, y, w) -> EchellePair_TYPE:
    '''
    find the common range of values between x, y
    '''
    idx = np.where(np.logical_and(
        x < np.amax(y),
        x > np.amin(y)
    ))
    return x[idx], w[idx]