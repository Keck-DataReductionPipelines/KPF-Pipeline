import re
import time
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from kpfpipe.models.level0 import KPF0
from scipy.optimize import curve_fit
from modules.Utils.utils import DummyLogger
from modules.Utils.kpf_parse import HeaderParse, get_data_products_L0, get_datecode
from modules.quicklook.src.analyze_time_series import AnalyzeTimeSeries

class AnalyzeL0:
    """
    Description:
        This class contains functions to analyze L0 images (storing them
        as attributes) and functions to plot the results.

    Arguments:
        L0 - an L0 object

    Attributes:
        header (dictionary) - primary header of the L0
        name (string) - type of spectrum (e.g., 'Flat')
        ObsID (string) - Observation ID (e.g. 'KP.20240101.123456.12')
        data_products (array of strings) - list of data products available in the L0
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


    def measure_read_noise_overscan(self, nparallel=30, nserial=50, nsigma=5.0, verbose=False): 
        """
        Measure read noise in the overscan region of a KPF CCD image. 
        Read noise is measured as the standard deviation of the pixel values, 
        after applying an n-sigma outlier rejection.
        
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
            if verbose:
                self.logger.info(f'Read noise({region}) = {self.read_noise_overscan[region]}')


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
            plt.title('L0 - Green CCD: ' + str(self.ObsID) + ' - ' + self.name, fontsize=14)
        if chip == 'red':
            plt.title('L0 - Red CCD: ' + str(self.ObsID) + ' - ' + self.name, fontsize=14)
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

class AnalyzeL0Stack:
    """
    Description:
        This class contains functions to analyze sets of L0 images and functions 
        to plot the results.

    Arguments:
        L0_array - array of L0 filenames or L0 objects
        object_like (None or str) - object string to query time series database, e.g. 'flat-all'
        start_date (None or datetime object) - if not none, datetime of start of time range) 
        end_date (None or datetime object) - if not none, datetime of end of time range) 
        db_path (str) - path to time series database file (e.g., '/data/time_series/kpf_ts.db')
        first_n (False or int) - if False, all matching spectra are read

    Attributes:
        L0_array - array of L0 filenames or L0 objects
        #header_array (array of dictionaries) - array of primary headers of the L0s
        #name_array (string) - array of spectrum types (e.g., 'Flat')
        #ObsID_array (string)- array of Observation IDs (e.g. 'KP.20240101.123456.12')
        #data_products (array of strings) - list of data products available in the L0
        #amp_names (list) - names of the amplifier regions present in the first L0 image
        gain (dictionary) - gain values for amplifier regions (determined by Ashley Baker)
        #regions (list of strings) - list of amplifier regions in the first L0
        #nregions_green (int) - number of amplifier regions present in Green CCD image of the first L0
        #nregions_red (int) - number of amplifier regions present in Red CCD image of the first L0
        #green_present (boolean) - True if the Green CCD image is in the first L0 object
        #red_present (boolean) - True if the Red CCD image is in the first L0 object
        
    """

    def __init__(self, L0_array, logger=None, 
                       object_like=None, start_date=None, end_date=None,
                       db_path = '/data/time_series/kpf_ts.db', 
                       first_n=False):
        self.logger = logger if logger is not None else DummyLogger()

        # Create a list of L0 files using the times series DB, if needed
        if object_like != None:
            myTS = AnalyzeTimeSeries(db_path=db_path, logger=logger)
            cols = ['ObsID', 'OBJECT']
            df = myTS.dataframe_from_db(cols, object_like=object_like, 
                                              start_date=start_date, 
                                              end_date=end_date,
                                              not_junk=True)
            if first_n == False:
                self.nframes = len(df)
            else:
                self.nframes = int(first_n)
            self.L0_array = []
            for i in np.arange(self.nframes):
                ObsID = df['ObsID'].iloc[i]
                L0_file = '/data/L0/' + get_datecode(ObsID) + '/' + ObsID + '.fits'
                self.L0_array.append(L0_file)
        else:
            self.nframes = len(L0_array)
            self.L0_array = copy.deepcopy(L0_array)
        
        # Load L0 objects
        self.logger.info("Loading L0 objects")
        for i in tqdm(np.arange(self.nframes)):
            if type(self.L0_array[i]) == 'kpfpipe.models.level0.KPF0':
                pass
            else: # assume that non-L0 elements are paths to L0 files
                self.L0_array[i] = KPF0.from_fits(self.L0_array[i])

        #self.data_products = get_data_products_L0(L0)
        #primary_header = HeaderParse(L0, 'PRIMARY')
        #self.header = primary_header.header
        #self.name = primary_header.get_name()
        #self.ObsID = primary_header.get_obsid()
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
        
        # Assemble images (possibly move this code to the plotting routines 
        #                  because it is slow and not always needed for both chips)
#        for chip in ['green', 'red']:
#            if not hasattr(self, 'image_{chip}_array'):
#               self.make_image_array(chip)
        
        #if ('Green' in self.data_products) or ('Red' in self.data_products):
        #    self.regions = [key for key in self.L0.extensions if 'AMP' in key]
        #    bad_regions = [] # remove regions with no data
        #    for region in self.regions:
        #        if len(self.L0[region].data) == 0:
        #            bad_regions.append(region) # the bad region can't be removed in this loop, so a list is created
        #    if bad_regions != []:
        #        for region in bad_regions:
        #            self.regions.remove(region)
        #    self.determine_regions() # sets self.green_present, self.red_present, nregions_green, nregions_red
        
    def make_image_array(self, chip):
        """
        Description:
           Method to make in image array, which involves applying gain, 
           stitching amplifier regions, and subtracting bias.
        
        Arguments:
            chip (string) - 'green' or 'red'
        
        Attributes:
           image_green_array (array of 2D numpy arrays) - analagus to L0_array
           image_red_array   (array of 2D numpy arrays) - analagus to L0_array
        
        To-do: 
           make this work for the four-amplifier case
        """
        if chip == 'green':
             self.logger.info("Assembling Green CCD images")
             self.image_green_array = []
             for i in tqdm(np.arange(self.nframes)):
                 image_green_amp1 = np.array(self.L0_array[i]['GREEN_AMP1'].data) * self.gain['GREEN_AMP1'] / 2**16
                 image_green_amp2 = np.array(self.L0_array[i]['GREEN_AMP2'].data) * self.gain['GREEN_AMP2'] / 2**16
                 image = np.flipud(np.concatenate((image_green_amp1, image_green_amp2), axis=1))
                 bias = np.percentile(image, 1)
                 image -= bias
                 self.image_green_array.append(image)
        if chip == 'red':
             self.logger.info("Assembling Red CCD images")
             self.image_red_array = []
             for i in tqdm(np.arange(self.nframes)):
                 image_red_amp1 = np.array(self.L0_array[i]['RED_AMP1'].data) * self.gain['RED_AMP1'] / 2**16
                 image_red_amp2 = np.array(self.L0_array[i]['RED_AMP2'].data) * self.gain['RED_AMP2'] / 2**16
                 image = np.flipud(np.concatenate((image_red_amp1, image_red_amp2), axis=1))
                 bias = np.percentile(image, 1)
                 image -= bias
                 self.image_red_array.append(image)

    def make_normalized_data(self, image, xmin, xmax, ymin, ymax, order=4):
        """
        Method to normalize 1-D data by fitting an nth order polynomial and dividing it
        """
        region = image[ymin:ymax, xmin:xmax]
        spectrum = np.sum(region, axis=0)
        variance = spectrum
        params, x_filtered, y_filtered = self.fit_poly_with_outlier_rejection(np.arange(len(spectrum)), spectrum, order)
        y_fitted = self.poly_n(np.arange(len(spectrum)), *params)
        normalized_data = spectrum / y_fitted
        return normalized_data, variance

    def poly_n(self, x, *coeffs):
        """
        Method to create an nth order polynomial
        """
        return sum(c * x**i for i, c in enumerate(coeffs))
    
    def fit_poly_with_outlier_rejection(self, x, y, order, max_iter=10, threshold=3):
        """
        Method to fit polynomial and remove outliers
        """
        for i in range(max_iter):
            params, covariance = curve_fit(lambda x, *params: self.poly_n(x, *params), x, y, p0=[1]*(order+1))
            y_fitted = self.poly_n(x, *params)
            residuals = y - y_fitted
            std_residuals = np.std(residuals)
            outliers = np.abs(residuals) > threshold * std_residuals
            if not np.any(outliers):
                break
            x = x[~outliers]
            y = y[~outliers]
        return params, x, y                
    
    def plot_2D_surrounding_box(self, image, xmin, xmax, ymin, ymax, title=''):
        """
        Method to plot a 2D image. (should merge w/plot_2D_panel())
        """
        plt.figure(figsize=(10, 8), tight_layout=True)
        plt.imshow(image, cmap='viridis', origin='lower',
                   vmin=np.percentile(image,1),
                   vmax=np.percentile(image,99.5))
        plt.xlabel('Column (pixel number)', fontsize=14)
        plt.ylabel('Row (pixel number)', fontsize=14)
        cbar_label = 'e-'
        cbar = plt.colorbar(shrink=0.95, label=cbar_label)
        cbar.ax.yaxis.label.set_size(14)
        cbar.ax.tick_params(labelsize=12)
        plt.grid(False)
        plt.show()
        plt.close('all')
        
    def plot_2D_panel(self, ax, image, xmin, xmax, ymin, ymax, title=''):
        """
        Method to make a panel in a plot showing a 2D image with a 
        red box around the extraction region.
        """
        im = ax.imshow(image, cmap='viridis', origin='lower',
                   vmin=np.percentile(image, 1),
                   vmax=np.percentile(image, 99.5))
        ax.set_xlabel('Column (pixel number)', fontsize=14)
        ax.set_ylabel('Row (pixel number)', fontsize=14)
        cbar = plt.colorbar(im, ax=ax, shrink=0.75)
        cbar.set_label('e-', size=14)
        cbar.ax.tick_params(labelsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(False)
        
        # Draw a red box around the specified coordinates
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    def plot_zoomed_2D_panel(self, ax, image, xmin, xmax, ymin, ymax, buffer=50, height=50, title=''):
        """
        Method to make a panel in a plot showing a zoomed 2D image with a 
        red box around the extraction region.
        """
        y_buffer = (height - (ymax - ymin)) // 2
        xmin_zoom = max(xmin - buffer, 0)
        xmax_zoom = min(xmax + buffer, image.shape[1] - 1)
        ymin_zoom = max(ymin - y_buffer, 0)
        ymax_zoom = min(ymax + y_buffer, image.shape[0] - 1)
        
        im = ax.imshow(image, cmap='viridis', origin='lower',
                       vmin=np.percentile(image, 1),
                       vmax=np.percentile(image, 99.5))
        ax.set_xlabel('Column (pixel number)', fontsize=14)
        ax.set_ylabel('Row (pixel number)', fontsize=14)
        cbar = plt.colorbar(im, ax=ax, shrink=0.90)
        cbar.set_label('e-', size=14)
        cbar.ax.tick_params(labelsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(False)
    
        # Set the axis limits to the zoom-in range values with buffer
        ax.set_xlim(xmin_zoom, xmax_zoom)
        ax.set_ylim(ymin_zoom, ymax_zoom)
    
        # Set tick labels to reflect the desired range
        x_ticks = np.linspace(xmin_zoom, xmax_zoom, num=11, dtype=int)
        y_ticks = np.linspace(ymin_zoom, ymax_zoom, num=11, dtype=int)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks)
        ax.set_aspect(aspect='auto') 
    
        # Add a red box to show the exact region
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    def plot_1D_panel(self, ax, pixel_num, normalized_flux, title='', expected_SNR=0):
        """
        Method to make a panel in a plot showing a 1D extracted spectrum.
        """
        label = f'SNR = {int(1/np.std(normalized_flux))} (1/std(normalized flux))'
        if expected_SNR > 0:
            label = label + f'\nSNR (expected) = {int(expected_SNR)} (sqrt(median(variance))'
        ax.step(pixel_num, normalized_flux, label=label)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Pixel Number', fontsize=14)
        ax.set_ylabel('Normalized Flux', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize=12)
        ax.grid()
    
    def plot_2D_3_panels(self, image, chip, xmin, xmax, ymin, ymax, title=''):
        """
        Method to assemble a 3-panel plot with the L0 image, zoom of L0 image, and 1D extracted spectrum.
        """
        if not hasattr(self, f'image_{chip}_array'):
            self.make_image_array(chip)
        if type(image) == type(0):
            if chip == 'green':
                image = self.image_green_array[image]
            if chip == 'red':
                image = self.image_red_array[image]
        
        pixel_num = np.arange(xmin, xmax)
        normalized_data, variance = self.make_normalized_data(image, xmin, xmax, ymin, ymax)
        expected_SNR = np.sqrt(np.median(variance))
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [2.2, 1, 1]})
        title_panel1 = f'L0 Image ({chip})'
        title_panel2 = f'Zoomed L0 Image ({chip})'
        title_panel3 = f'Normalized Box-extracted Spectrum'
        if title != '':
            title_panel1 = title + f' - L0 Image ({chip})'
            title_panel2 = title + f' - Zoomed L0 Image ({chip})'
            title_panel3 = title + f' - Normalized Box-extracted Spectrum'
        self.plot_2D_panel(ax1, image, xmin, xmax, ymin, ymax, title=title_panel1)
        self.plot_zoomed_2D_panel(ax2, image, xmin, xmax, ymin, ymax, title=title_panel2)
        self.plot_1D_panel(ax3, pixel_num, normalized_data, title=title_panel3, expected_SNR=expected_SNR)
        
        plt.tight_layout()
        plt.show()
        
