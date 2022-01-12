import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import units as u, constants as cst

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

    def plot_drift(self, wlpixelfile1, wlpixelfile2, savename):
        """
        overall RV of cal data vs time for array of input files
        """
        # for each file in array, calculate drift using func below between ith file and 0th file
        # then plot drift vs time
        
        drift = self.calcdrift_polysolution(wlpixelfile1, wlpixelfile2)

        obsname1 = wlpixelfile1.split('_')[1]
        obsname2 = wlpixelfile2.split('_')[1]

        fig, ax = plt.subplots()
        ax.axhline(0, color='grey', ls='--')

        plt.plot(
            drift[:,0], drift[:,1], 'ko', ls='-'
        )
        plt.title('Inst. drift: {} to {}'.format(obsname1, obsname2))
        plt.xlabel('order')
        plt.ylabel('drift [cm s$^{-1}$]')
        plt.savefig(savename, dpi=250)

    def calcdrift_ccf(self, obstime, calfile1, calfile2):
        pass

    def calcdrift_polysolution(self, wlpixelfile1, wlpixelfile2):

        peak_wavelengths_ang1 = np.load(
            wlpixelfile1, allow_pickle=True
        ).tolist()

        peak_wavelengths_ang2 = np.load(
            wlpixelfile2, allow_pickle=True
        ).tolist()

        orders = peak_wavelengths_ang1.keys()

        drift_all_orders = np.empty((len(orders),2))

        # make a dataframe and join on wavelength
        for i, order_num in enumerate(orders):

            order_wls1 = pd.DataFrame(
                data = np.transpose([
                    peak_wavelengths_ang1[order_num]['known_wavelengths_vac'],
                    peak_wavelengths_ang1[order_num]['line_positions']
                ]),
                columns=['wl', 'pixel1']
            )

            order_wls2 = pd.DataFrame(
                data = np.transpose([
                    peak_wavelengths_ang2[order_num]['known_wavelengths_vac'],
                    peak_wavelengths_ang2[order_num]['line_positions']
                ]),
                columns=['wl', 'pixel2']
            )

            order_wls = order_wls1.set_index('wl').join(order_wls2.set_index('wl'))

            delta_lambda = order_wls.index.values[1:] - order_wls.index.values[:-1]
            delta_pixel = order_wls.pixel1.values[1:] - order_wls.pixel1.values[:-1]

            drift_pixels = order_wls['pixel2'] - order_wls['pixel1']

            drift_wl = drift_pixels.values[1:] / delta_pixel * delta_lambda

            alpha = (drift_wl / order_wls.index.values[1:])

            drifts_cms = (alpha**2 + 2 * alpha) / (alpha**2 + 2 * alpha + 2) * cst.c.to(u.cm/u.s).value

            drift_all_orders[i,0] = order_num
            drift_all_orders[i,1] = np.mean(drifts_cms)

        return drift_all_orders

if __name__ == '__main__':
    myI = InstDrift()
    file1 = '/code/KPF-Pipeline/outputs/neidL1_20191217T023129_wls_pixels.npy'
    file2 = '/code/KPF-Pipeline/outputs/neidL1_20191217T023815_wls_pixels.npy'
    drift = myI.plot_drift(file1, file2, '/code/KPF-Pipeline/outputs/drift.png')
