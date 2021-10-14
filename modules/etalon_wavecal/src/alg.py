import numpy as np
from astropy.io import fits
import os.path
import matplotlib.pyplot as plt

from modules.wavelength_cal.src.alg import LFCWaveCalibration
from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0
from keckdrpframework.models.arguments import Arguments

class EtalonWaveCalAlg:

    def __init__(self, config=None, logger=None):
        configpull = ConfigHandler(config,'PARAM')
        self.min_order = configpull.get_config_value('min_order', 40)
        self.max_order = configpull.get_config_value('max_order', 45)
        self.height_sigma = configpull.get_config_value('height_sigma', .5)
        self.saveplots = configpull.get_config_value('saveplots', 'Etalon_plots')
        self.flux_ext = configpull.get_config_value('flux_ext','CALFLUX')
        self.wave_ext = configpull.get_config_value('wave_ext','SCIWAVE')

    def run_on_all_orders(self,flux):
        """Run wavelength calibration on all orders.

        Args:
            flux (np.array): Flux data
        """
        if not os.path.exists(self.saveplot):
            os.system('mkdir {}'.format(self.saveplots))

        if self.saveplots is not None:
            plt.figure()
            im = plt.imshow(flux,aspect='auto')
            im.set_clim(0,20000)
            plt.savefig('{}/extracted_spectrum.pdf'.format(self.saveplots),dpi=250)
            #pdf vs png, preference?

        for order in np.arange(self.min_order,self.max_order):

            if self.saveplots is not None:
                plot_dir = '{}/order{}'.format(self.saveplots,order)
                if not os.path.exists(plot_dir)
                    os.system('mkdir {}'.format(plot_dir))

                plt.figure()
                plt.plot(flux[order])
                plt.savefig('{}/extracted_spectrum.pdf'.format(plot_dir),dpi=250)

            else:
                plot_dir = None

            new_peaks,peaks,peak_heights,gauss_coeffs = LFCWaveCalibration.find_peaks_in_order(
                flux[order],self.height_sigma, plot_path=plot_dir)

    def etalon_alg(self,hdul):
        #make sure fits file is opened elsewhere
        """Set-up and running Etalon algorithm.

        Args:
            hdul (HDUList): HDUList of FITS file
        """

        assert hdul[0].header['CAL-OBJ'].startswith('Etalon')

        calflux = hdul[self.flux_ext]
        calflux = np.nan_to_num(calflux)

        #neid masking
        calflux[:,425:450] = 0

        run_on_all_orders(calflux)

