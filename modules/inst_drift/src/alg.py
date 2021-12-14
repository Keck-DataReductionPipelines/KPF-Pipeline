import numpy as np
from astropy.io import fits

# two algorithms: 1) using existing polynomial solutions, interpolate for given obs time:
#   compute a mean RV based on the pixel-level differences
# 2) using CCF to compute overall RV, interpolate for given obs time

class InstDrift(object):
    """
    """
    def __init__(self, config=None, logger=None):
        self.config=config
        self.logger=logger

    def plot_poly_coefs(self, coef_num, order_num):
        """
        (to implement in future): polynomail coeffs vs time for specific order (inputs: poly num, order num)
        """
        pass

    def plot_drift(self, calibration_fitsfile_array):
        """
        overall RV of cal data vs time for array of input files
        """
        pass

    def calcdrift_ccf(self, obstime, calfile1, calfile2):
        pass

    def calcdrift_polysolution(self, obstime, calfile1, calfile2):

        calfile1 = fits.open(calfile1)
        wls1 = calfile1['SCIWAVE'].data
        time1 = calfile1[0].OBSJD # TODO: is this correct time to use for interpolation?

        calfile2 = fits.open(calfile2)
        wls2 = calfile2['SCIWAVE'].data
        time2 = calfile2[0].OBSJD

        wl_diff = np.mean(wls2 - wls1)

        # TODO: interpolate based on time diffs, return
