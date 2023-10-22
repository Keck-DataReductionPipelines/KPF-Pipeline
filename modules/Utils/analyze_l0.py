import re
import time
import numpy as np
import matplotlib.pyplot as plt
from modules.Utils.kpf_parse import HeaderParse, get_data_products_L0
from modules.Utils.utils import DummyLogger
from datetime import datetime

class AnalyzeL0:
    """
    Description:
        This class contains functions to analyze L0 images (storing them
        as attributes) and functions to plot the results.

    Arguments:
        L0 - an L0 object

    Attributes:
        amp_names (list) - names of the amplifier regions present in the L0 image
        gain (dictionary) - gain values for amplifier regions (determined by Ashley Baker)
        regions (list of strings) - list of amplifier regions in L0
        nregions_green (int) - number of amplifier regions present in Green CCD image
        nregions_red (int) - number of amplifier regions present in Red CCD image
        green_present (boolean) - True if the Green CCD image is in the L0 object
        red_present (boolean) - True if the Red CCD image is in the L0 object
        read_noise_overscan (dictionary) - read noise estimates in e- determined by std of overscan regions.  
                                           The dictionary is over the list of regions
        
    """

    def __init__(self, L0, logger=None):
        self.logger = logger if logger is not None else DummyLogger()
        self.L0 = L0
        self.data_products = get_data_products_L0(L0)
        primary_header = HeaderParse(L0, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        self.ObsID = primary_header.get_obsid()
        if ('Green' in self.data_products) or ('Red' in self.data_products):
            self.regions = [key for key in self.L0.extensions if 'AMP' in key]
            bad_regions = [] # remove regions with no data
            for region in self.regions:
                if len(self.L0[region].data) == 0:
                    bad_regions.append(region) # the bad region can't be removed in this loop, so a list is created
            if bad_regions != []:
                for region in bad_regions:
                    self.regions.remove(region)
            self.determine_regions() # sets self.green_present, self.red_present, nregions_green, nregions_red
            self.gain = {
                 'GREEN_AMP1': 5.175,
                 'GREEN_AMP2': 5.208,
                 'GREEN_AMP3': 5.52,
                 'GREEN_AMP4': 5.39,
                 'RED_AMP1': 5.02,
                 'RED_AMP2': 5.27,
                 'RED_AMP3': 5.32,
                 'RED_AMP4': 5.23,
            }
            self.read_noise_overscan = {} # initialize dictionary
            try:
                self.measure_read_noise_overscan()
            except Exception as e:
                print(e)
                print('measure_read_noise_overscan() failed on ' + self.ObsID)
            
            self.read_speed, self.green_acf, self.red_acf, self.green_read_time, self.red_read_time = \
                  primary_header.get_read_speed()
    
    def reject_outliers(self, data, n=5.0):
        """
        This method performs an iterative n-sigma outlier rejection 
        and is used in measure_read_noise_overscan().
        
        Parameters:
            data (list or np.array): Input data set
            n (float): Number of standard deviations from the mean. 
                       Data points outside this range are considered outliers.
        
        Returns:
            np.array: Data set with outliers removed.
        """
        data = np.array(data)
        
        while True:
            mean, std = np.mean(data), np.std(data)
            outliers = (data < mean - n * std) | (data > mean + n * std)
            if not np.any(outliers):
                break
            data = data[~outliers]
        
        return data


    def determine_regions(self):
        """
        This method determines if the Green and Red CCD images are present in the 
        AnalyzeL0 object and counts the number of amplifier regions in each CCD image.
        These results are written to attributes of the AnalyzeL0 object.
        
        Parameters:
            None 

        Attributes:
            self.green_present (boolean) - True if the Green CCD image (of non-zero size) is in the L0 object
            self.red_present (boolean) - True if the Red CCD image (of non-zero size) is in the L0 object
            self.nregions_green (int) - number of amplifier regions present in Green CCD image
            self.nregions_red (int) - number of amplifier regions present in Red CCD image

        Returns:
            None
        """
        
        # Determine if specific amplifier regions are present and count them
        for CHIP in ['GREEN', 'RED']:
            nregions = 0
            amp1_present = False
            amp2_present = False
            amp3_present = False
            amp4_present = False
            if CHIP + '_AMP1' in self.L0.extensions:
                if self.L0[CHIP + '_AMP1'].shape[0] > 0: # Sanity check that data is present
                    amp1_present = True
            if CHIP + '_AMP2' in self.L0.extensions:
                if self.L0[CHIP + '_AMP2'].shape[0] > 0:
                    amp2_present = True
            if CHIP + '_AMP3' in self.L0.extensions:
                if self.L0[CHIP + '_AMP3'].shape[0] > 0:
                    amp3_present = True
            if CHIP + '_AMP4' in self.L0.extensions:
                if self.L0[CHIP + '_AMP4'].shape[0] > 0:
                    amp4_present = True
            if not amp1_present:
                nregions = 0
            if amp1_present:
                nregions = 1
            if amp1_present & amp2_present:
                nregions = 2
            if amp1_present & amp2_present & amp3_present & amp4_present:
                nregions = 4
            if CHIP == 'GREEN':
                self.nregions_green = nregions
                if nregions >= 1:
                    self.green_present = True
                else:
                    self.green_present = True
            if CHIP == 'RED':
                self.nregions_red = nregions
                if nregions >= 1:
                    self.red_present = True
                else:
                    self.red_present = True


    def measure_read_noise_overscan(self, nparallel=30, nserial=50, nsigma=5.0): 
        """
        Measure read noise in the overscan region of a KPF CCD image. 
        Read noise is measured as the standard deviation of the pixel values, 
        after applying an n-sigma iterative outlier rejection.
        
        Args:
            nparallel (integer) - overscan length in parallel direction 
                                  (30 pixels was final value in 2023)
            nserial (integer) - overscan length in serial direction 
                                (30 pixels was final value in 2023)
    
        Attributes:
            TBD
    
        Returns:
            None
        """

        for region in self.regions:
            CHIP = region.split('_')[0]
            NUM = region.split('AMP')[1]
            gain = self.gain[CHIP+'_AMP'+NUM]
            data = self.L0[region]
            if np.nanmedian(data) > 200*2**16:  # divide by 2**16 if needed
                data /= 2**16
            if region == 'GREEN_AMP2':
                data = data[:,::-1]  # flip so that overscan is on the right
            if region == 'GREEN_AMP3':
                data = data[::-1,:]
            if region == 'GREEN_AMP4':
                data = data[::-1,::-1]
            if region == 'RED_AMP2':
                data = data[:,::-1]
            if region == 'RED_AMP3':
                data = data 
            if region == 'RED_AMP4':
                data = data[:,::-1]
            overscan_region = data[5:2040 + nparallel-5,2044 + 10:2044 + nserial-10]
            vals = self.reject_outliers(overscan_region.flat, nsigma)    
            self.read_noise_overscan[region] = gain * np.std(vals)


    def plot_L0_stitched_image(self, chip=None, fig_path=None, show_plot=False):
        """
        Generate a plot of the stitched L0 image.
        The image will be divided by 2^16, if appropriate.

        Args:
            chip (string) - "green" or "red"
            fig_path (string) - set to the path for a SNR vs. wavelength file
                to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it the current environment
            (e.g., in a Jupyter Notebook).
        """
        L0 = self.L0
        
        CHIP = chip.upper()
        if chip == 'green' or chip == 'red':
            if chip == 'green':
                regions = self.nregions_green
            if chip == 'red':
                regions = self.nregions_red

            # Assemble image
            if regions == 1:
                self.logger.debug("The code currently doesn't support single amplifier mode because this requires knowing which amplifer is used to get the proper orientation.")
                return
            if regions == 2:
                if chip == 'green':
                    image = np.flipud(np.concatenate((L0[CHIP + '_AMP1'].data, L0[CHIP + '_AMP2'].data), axis=1))
                if chip == 'red':
                    image = np.concatenate((L0[CHIP + '_AMP1'].data, L0[CHIP + '_AMP2'].data), axis=1)
            if regions == 4:
                image_bot = np.concatenate((L0[CHIP + '_AMP1'].data, L0[CHIP + '_AMP2'].data), axis=1)
                image_top = np.concatenate((L0[CHIP + '_AMP3'].data, L0[CHIP + '_AMP4'].data), axis=1)
                image = np.concatenate((image_bot, image_top), axis=0)

            # Determine if image needs to be divided by 2^16
            if np.nanmedian(image) > 200*2**16:
                twotosixteen = True
                image /= 2**16
            else:
                twotosixteen = False
        else:
            self.logger.debug('plot_L0_stitched_image: need to set chip="green" or "red"')
            return

        plt.figure(figsize=(10, 8), tight_layout=True)
        plt.imshow(image, cmap='viridis', origin='lower',
                   vmin=np.percentile(image,1),
                   vmax=np.percentile(image,99.5))
        if chip == 'green':
            plt.title('L0 (no processing) - Green CCD: ' + str(self.ObsID) + ' - ' + self.name, fontsize=14)
        if chip == 'red':
            plt.title('L0 (no processing) - Red CCD: ' + str(self.ObsID) + ' - ' + self.name, fontsize=14)
        plt.xlabel('Column (pixel number)', fontsize=14)
        plt.ylabel('Row (pixel number)', fontsize=14)
        cbar_label = 'ADU'
        if twotosixteen:
            cbar_label = cbar_label + r' / $2^{16}$'
        cbar = plt.colorbar(shrink=0.95, label=cbar_label)
        cbar.ax.yaxis.label.set_size(14)
        cbar.ax.tick_params(labelsize=12)
        plt.grid(False)

        # Create label for read noise
        rn_text = ''
        if self.read_noise_overscan != {}:
            rn_text = 'Read noise = '
            chip_regions = [item for item in self.regions if CHIP in item]
            for i, region in enumerate(chip_regions):
                if CHIP == 'GREEN':
                    nregions = self.nregions_green
                if CHIP == 'RED':
                    nregions = self.nregions_red
                if region.split('_')[0] == CHIP:
                    rn_text += f"{self.read_noise_overscan[region]:.2f}"
                    if i < nregions-1:
                        rn_text += ', '
            rn_text += ' e-'

        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    xytext=(100, -32), textcoords='offset points')
        plt.annotate(rn_text, xy=(0, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="left", va="bottom",
                    xytext=(-100, -32), textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)     

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=600, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')
