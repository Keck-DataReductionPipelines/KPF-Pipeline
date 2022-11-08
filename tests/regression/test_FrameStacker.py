import numpy as np
from modules.Utils.frame_stacker import FrameStacker

def test_compute():

    """
    Test compute method of FrameStacker class.
    """

    print(test_compute.__doc__)

    fs = FrameStacker(a,nsigma)
    frame_avg,frame_var,frame_cnt,frame_unc = fs.compute()

    print("frame_avg =",frame_avg)
    print("frame_var =",frame_var)
    print("frame_cnt =",frame_cnt)
    print("frame_unc =",frame_unc)

if __name__ == '__main__':

    nsigma = 2.5

    a = np.arange(24)
    a.shape = (4,2,3)

    a[3][1][2] = 42.0    # Stick in an outlier.

    print("a=",a)

    test_compute()
