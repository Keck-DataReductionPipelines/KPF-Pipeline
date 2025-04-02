import time
import copy
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from astropy.time import Time
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.stats import norm
from scipy.stats import median_abs_deviation
from modules.Utils.kpf_parse import HeaderParse, get_datecode_from_filename
from modules.Utils.utils import DummyLogger
from astropy.time import Time
from astropy.table import Table
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
from scipy.signal import correlate
from scipy.stats import median_abs_deviation
from kpfpipe.models.level0 import KPF0
#import emcee
#import corner

class Analyze2D:
    """
    This class contains functions to analyze 2D images (storing them
    as attributes) and functions to plot the results.
    Some of the functions need to be filled in

    Arguments:
        D2 - a 2D object

    Attributes:
        header - header of the PRIMARY extension of the 2D object
        name - name of source (e.g., 'Bias', 'Etalon', '185144')
        ObsID - observation  ID (e.g. 'KP.20230704.02326.27')
        exptime - exposure time (sec)
        green_dark_current_regions - dictionary specifying the regions where 
                                     dark current is measured on the Green CCD
        red_dark_current_regions - dictionary specifying the regions where 
                                   dark current is measured on the Red CCD
        green_coll_pressure_torr - ion pump pressure (Green CCD, collimator side)
        green_ech_pressure_torr  - ion pump pressure (Green CCD, echelle side)
        green_coll_current_a     - ion pump current (Green CCD, collimator side)
        green_ech_current_a      - ion pump current (Green CCD, echelle side)
        red_coll_pressure_torr   - ion pump pressure (Red CCD, collimator side)
        red_ech_pressure_torr    - ion pump pressure (Red CCD, echelle side)
        red_coll_current_a       - ion pump current (Red CCD, collimator side)
        red_ech_current_a        - ion pump current (Red CCD, echelle side)
        green_percentile_99      - 99th percentile flux of Green 2D image (e-)
        green_percentile_90      - 90th percentile flux of Green 2D image (e-)
        green_percentile_50      - 50th percentile flux of Green 2D image (e-)
        green_percentile_10      - 10th percentile flux of Green 2D image (e-)
        red_percentile_99        - 99th percentile flux of Red 2D image (e-)
        red_percentile_90        - 90th percentile flux of Red 2D image (e-)
        red_percentile_50        - 50th percentile flux of Red 2D image (e-)
        red_percentile_10        - 10th percentile flux of Red 2D image (e-)
    """

    def __init__(self, D2, logger=None):
        self.logger = logger if logger is not None else DummyLogger()
        self.D2 = copy.deepcopy(D2) # use D2 instead of 2D because variable names can't start with a number
        self.df_telemetry = self.D2['TELEMETRY']  # read as Table for astropy.io version of FITS
        primary_header = HeaderParse(D2, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        self.ObsID = primary_header.get_obsid()
        self.exptime = self.header['EXPTIME']
        self.green_dark_current_regions = None # Green CCD regions where dark current is measured, defined below
        self.red_dark_current_regions   = None # Red CCD regions where dark current is measured, defined below
        self.green_coll_pressure_torr = 0
        self.green_ech_pressure_torr  = 0
        self.green_coll_current_a     = 0
        self.green_ech_current_a      = 0
        self.red_coll_pressure_torr   = 0
        self.red_ech_pressure_torr    = 0
        self.red_coll_current_a       = 0
        self.red_ech_current_a        = 0
        try:
            self.green_percentile_99, self.green_percentile_90, self.green_percentile_50, self.green_percentile_10 = np.nanpercentile(np.array(D2['GREEN_CCD'].data),[99,90,50,10])
        except:
            self.logger.error('Problem computing SNR for Green 2D image')
        try:
            self.red_percentile_99, self.red_percentile_90, self.red_percentile_50, self.red_percentile_10 = np.nanpercentile(np.array(D2['RED_CCD'].data),[99,90,50,10])
        except:
            self.logger.error('Problem computing SNR for Green 2D image')


    def measure_master_age(self, kwd='BIASFILE', verbose=False):
        '''
        Computes the number of whole days between the observation and a master file 
        listed in the PRIMARY header.  

        Arguments:
            kwd - keyword name of WLS file (usually 
                  'BIASFILE', 'DARKFILE', or 'FLATFILE')
    
        Returns:
            master_wls_file - number of days between the observation and the
                              date of observations for the WLS files
        '''

        date_obs_str = self.header['DATE-OBS']
        date_obs_datetime = datetime.strptime(date_obs_str, "%Y-%m-%d").date()

        if verbose:
            self.logger.info(f'Date of observation: {date_obs_str}')
        
        try:
            if kwd in self.header:
                master_filename = self.header[kwd]
                master_filename_datetime = get_datecode_from_filename(master_filename, datetime_out=True)
                master_filename_datetime = master_filename_datetime.replace(hour=0, minute=0, second=0, microsecond=0).date()
                if verbose:
                    self.logger.info(f'Date of {kwd}: {master_filename_datetime.strftime("%Y-%m-%d")}')
                
                age_master_file = (master_filename_datetime - date_obs_datetime).days
                if verbose:
                    self.logger.info(f'Time between observation and {kwd}: {age_master_file}')
    
                return age_master_file
            else:
                age_master_file = -999 # standard value indicating keyword not available
                return age_master_file

        except KeyError as e:
            self.logger.info(f"KeyError: {e}")
            pass

        except Exception as e:
            self.logger.error(f"Problem with determining age of {kwd}: {e}\n{traceback.format_exc()}")
            return None


    def measure_2D_dark_current(self, chip=None):
        """
        This method measures the dark current (flux/hr) in 10 regions on either
        the green or red CCD.  The regions were chosen so to be sensitive to 
        apparent dark current from the ion pumps (ech and coll), six reference
        regions (ref1-ref6), and areas near the chip edges with significant 
        amplifier glow (amp1 and amp2).
        
        Args:
            chip (string) - 'green' or 'red'

        Attributes (set by this method):
            green_dark_current_regions - dictionary specifying the regions where 
                                         dark current [e-/hr] is measured on the Green CCD
            red_dark_current_regions - dictionary specifying the regions where 
                                       dark current [e-/hr] is measured on the Red CCD
            green_coll_pressure_torr - ion pump pressure (Green CCD, collimator side)
            green_ech_pressure_torr  - ion pump pressure (Green CCD, echelle side)
            green_coll_current_a     - ion pump current (Green CCD, collimator side)
            green_ech_current_a      - ion pump current (Green CCD, echelle side)
            red_coll_pressure_torr   - ion pump pressure (Red CCD, collimator side)
            red_ech_pressure_torr    - ion pump pressure (Red CCD, echelle side)
            red_coll_current_a       - ion pump current (Red CCD, collimator side)
            red_ech_current_a        - ion pump current (Red CCD, echelle side)

        Returns:
            None
        """
        D2 = self.D2
        self.df_telemetry = self.D2['TELEMETRY']  # read as Table for astropy.io version of FITS
        if not self.exptime > 1:
            exptime_hr = 1/3600 
        else:
            exptime_hr = self.exptime/3600

        # Read telemetry
        #df_telemetry = Table.read(D2, hdu='TELEMETRY').to_pandas() # need to refer to HDU by name
        num_columns = ['average', 'stddev', 'min', 'max']
        for column in self.df_telemetry:
            #df_telemetry[column] = df_telemetry[column].str.decode('utf-8')
            self.df_telemetry = self.df_telemetry.replace('-nan', 0)# replace nan with 0
            if column in num_columns:
                self.df_telemetry[column] = pd.to_numeric(self.df_telemetry[column], downcast="float")
            else:
                self.df_telemetry[column] = self.df_telemetry[column].astype(str)
        self.df_telemetry.set_index("keyword", inplace=True)

        reg = {'ref1': {'name': 'Reference Region 1',         'x1': 1690, 'x2': 1990, 'y1': 1690, 'y2': 1990, 'short':'ref1', 'med_elec':0, 'label':''},
               'ref2': {'name': 'Reference Region 2',         'x1': 1690, 'x2': 1990, 'y1': 2090, 'y2': 2390, 'short':'ref2', 'med_elec':0, 'label':''},
               'ref3': {'name': 'Reference Region 3',         'x1': 2090, 'x2': 2390, 'y1': 1690, 'y2': 1990, 'short':'ref3', 'med_elec':0, 'label':''},
               'ref4': {'name': 'Reference Region 4',         'x1': 2090, 'x2': 2390, 'y1': 2090, 'y2': 2390, 'short':'ref4', 'med_elec':0, 'label':''},
               'ref5': {'name': 'Reference Region 5',         'x1':   80, 'x2':  380, 'y1':  700, 'y2': 1000, 'short':'ref5', 'med_elec':0, 'label':''},
               'ref6': {'name': 'Reference Region 6',         'x1':   80, 'x2':  380, 'y1': 3080, 'y2': 3380, 'short':'ref6', 'med_elec':0, 'label':''},
               'amp1': {'name': 'Amplifier Region 1',         'x1':  300, 'x2':  500, 'y1':    5, 'y2':   20, 'short':'amp1', 'med_elec':0, 'label':''},
               'amp2': {'name': 'Amplifier Region 2',         'x1': 3700, 'x2': 3900, 'y1':    5, 'y2':   20, 'short':'amp2', 'med_elec':0, 'label':''},
               'coll': {'name': 'Ion Pump (Collimator side)', 'x1': 3700, 'x2': 4000, 'y1':  700, 'y2': 1000, 'short':'coll', 'med_elec':0, 'label':''},
               'ech':  {'name': 'Ion Pump (Echelle side)',    'x1': 3700, 'x2': 4000, 'y1': 3080, 'y2': 3380, 'short':'ech',  'med_elec':0, 'label':''}
              }
        if (chip.lower() == 'green'): 
            frame = np.array(D2['GREEN_CCD'].data)
            self.green_coll_pressure_torr = self.df_telemetry.at['kpfgreen.COL_PRESS', 'average']
            self.green_ech_pressure_torr  = self.df_telemetry.at['kpfgreen.ECH_PRESS', 'average']
            self.green_coll_current_a     = self.df_telemetry.at['kpfgreen.COL_CURR',  'average']
            self.green_ech_current_a      = self.df_telemetry.at['kpfgreen.ECH_CURR',  'average']
        if (chip.lower() == 'red'): 
            frame = np.array(D2['RED_CCD'].data)
            self.red_coll_pressure_torr = self.df_telemetry.at['kpfred.COL_PRESS', 'average']
            self.red_ech_pressure_torr  = self.df_telemetry.at['kpfred.ECH_PRESS', 'average']
            self.red_coll_current_a     = self.df_telemetry.at['kpfred.COL_CURR',  'average']
            self.red_ech_current_a      = self.df_telemetry.at['kpfred.ECH_CURR',  'average']

        for r in reg.keys():
            current_region = frame[reg[r]['y1']:reg[r]['y2'], reg[r]['x1']:reg[r]['x2']]
            reg[r]['med_elec'] = np.median(current_region)/exptime_hr
        if chip == 'green':
            self.green_dark_current_regions = reg
        if chip == 'red':
            self.red_dark_current_regions = reg


    def fit_double_gaussian_cdf(self, ngaussians=1, chip=None):
        """
        This method fits the cumulative distribution of intensity values with a 
        function that is the cumulative distribution of the sum of one or two 
        Gaussians with the same center. The current implementation is slow.
        
        Args:
            ngaussians - 1 (default) or 2
            chip (string) - 'green' or 'red'

#        Attributes (set by this method):
#            green_dark_current_regions - dictionary specifying the regions where 

        Returns:
            None
        """

        def one_gaussian_cdf(x, mu1, sigma1):
            return norm.cdf(x, mu1, sigma1) 

        def two_gaussian_cdfs(x, mu1, sigma1, A1, sigma2):
            return A1 * norm.cdf(x, mu1, sigma1) + (1-A1) * norm.cdf(x, mu1, sigma2)

#        def one_gaussian_cdf(x, mu1, sigma1, A1):
#            return A1 * norm.cdf(x, mu1, sigma1) 
#
#        def two_gaussian_cdfs(x, mu1, sigma1, A1, mu2, sigma2, A2):
#            return A1 * norm.cdf(x, mu1, sigma1) + A2 * norm.cdf(x, mu2, sigma2)
#
#        def one_gaussian_cdf(x, mu1, sigma1):
#            return norm.cdf(x, mu1, sigma1) 
#
#        def two_gaussian_cdfs(x, mu1, sigma1, sigma2):
#            return norm.cdf(x, mu1, sigma1) + norm.cdf(x, mu1, sigma2)

        D2 = self.D2
        if (chip.lower() == 'green'): 
            image = np.array(D2['GREEN_CCD'].data)
        if (chip.lower() == 'red'): 
            image = np.array(D2['RED_CCD'].data)
        intensities = image.flatten()
        intensities.sort()
        cdf = np.arange(len(intensities)) / float(len(intensities))


        # Define the new points where you want to sample the CDF
        ndownsample = 1000
        new_points = np.linspace(0, len(intensities) - 1, ndownsample, dtype=int)
        
        # Interpolate the CDF at these new points
        # Since we are dealing with indices here, we can use simple array indexing rather than interpolation
        downsampled_cdf = cdf[new_points]
        downsampled_intensities = intensities[new_points]
        
        #intensities = downsampled_intensities
        #downsampled_cdf = cdf

        #initial_params_1g = [np.mean(intensities), np.std(intensities), 1]
        initial_params_1g = [np.mean(intensities), np.std(intensities)]
        params_1g, _ = curve_fit(one_gaussian_cdf, intensities, cdf, p0=initial_params_1g)
        print(params_1g)
        
        initial_params_2g = [params_1g[0], params_1g[1], 1, 3*params_1g[1]]
        #initial_params_2g = [params_1g[0], params_1g[1], params_1g[2], params_1g[0], 3*params_1g[1], 0.001]
        #initial_params_2g = [params_1g[0], params_1g[1], 2*params_1g[1]]
        #bounds = ([-np.inf,np.inf],[-np.inf,np.inf],[],[])
        params_2g, _ = curve_fit(two_gaussian_cdfs, intensities, cdf, p0=initial_params_2g)#, bounds=bounds)
        print(params_2g)


    def measure_xdisp_offset(self, chip=None, ref_image=None, ref_extension=None, 
                             num_slices=31, slice_width=30, fit_half_width=8, 
                             fig_path=None, show_plot=False):
        """
        This method compares the self.D2 image (green or red) to a reference 
        2D image and determines the amount of vertical shift between then using
        cross-correlation functions of vertical slices of each image.
        
        Args:
            chip (string) - 'green' or 'red'
            ref_image (2D object, np.array, or filename) - another 2D object, 
                numpy array with the same dimensions as self.D2, 
                or the filename of a 2D object
            num_slices - how many vertical columns to sample across the image
            slice_width - how wide each vertical slice is, in pixels
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Attributes (set by this method):
            green_offset - offset in pixels between green 2D image an a reference
            green_offset_sigma - uncertainty in green_offset
            red_offset - offset in pixels between green 2D image an a reference
            red_offset_sigma - uncertainty in red_offset

        Returns:
            (optionally:) PNG plot in fig_path or shows the plot it in the 
            current environment (e.g., in a Jupyter Notebook).
        """

        def parabolic_peak(x, a, b, c):
            '''Parabola function'''
            return a * x**2 + b * x + c
        
        def subpixel_peak_position_fixed(corr, lags, fit_half_width=8):
            """
            Find peak of CCF.
            """
            peak_idx = np.argmax(corr)
            if peak_idx <= fit_half_width or peak_idx >= len(corr) - fit_half_width:
                return lags[peak_idx], lags[peak_idx]  # Edge case
        
            x_vals = lags[peak_idx - fit_half_width:peak_idx + fit_half_width + 1]
            y_vals = corr[peak_idx - fit_half_width:peak_idx + fit_half_width + 1]
        
            x_init = lags[peak_idx - 1:peak_idx + 2]
            y_init = corr[peak_idx - 1:peak_idx + 2]
            a, b, c = np.polyfit(x_init, y_init, 2)
            p0 = [a, b, c]
            try:
                popt, _ = curve_fit(parabolic_peak, x_vals, y_vals, p0=p0)
                a, b, _ = popt
                if np.abs(a) < 1e-6:
                    raise ValueError("Flat curvature")
                vertex = -b / (2 * a)
            except Exception:
                vertex = lags[peak_idx]
        
            peak_lag = lags[peak_idx]
            #print(f"peak_lag: {peak_lag:.1f}, offset: {vertex:.3f}, delta: {vertex - peak_lag:.3f}")
            return vertex, peak_lag
    
        plot_correlations = fig_path or show_plot
        
        
        if chip == 'green' or chip == 'red':
            CHIP = chip.upper()
            EXT = CHIP + '_CCD'
            img1 = self.D2[EXT]
            if ref_extension != None:
                EXT = CHIP + ref_extension
            if type(ref_image) == type(self.D2):
                img2 = ref_image[EXT]
            elif type(ref_image) == type(img1):
                img2 = ref_image
            elif type(ref_image) == type('abc'):
                D2_ref = KPF0.from_fits(ref_image)
                img2 = D2_ref[EXT]
        else:
            self.error('measure_xdisp_offset: chip not specified.  Returning.')
                       
        assert img1.shape == img2.shape, "Images must be the same shape"
        
        h, w = img1.shape
        xs = (w // 2 - (num_slices // 2) * slice_width) + np.arange(num_slices) * slice_width
    
        all_corrs = []
        all_lags = []
        offsets = []
        peak_lags = []
        lags = np.arange(-h + 1, h)
    
        for x in xs:
            slice1 = img1[:, x:x+slice_width].mean(axis=1)
            slice2 = img2[:, x:x+slice_width].mean(axis=1)
            slice1 -= np.mean(slice1)
            slice2 -= np.mean(slice2)
            corr = correlate(slice1, slice2, mode='full')
            # lags already defined globally
            all_lags.append(lags)  # ensure lags match corr
            all_corrs.append(corr)
    
            offset, peak_lag = subpixel_peak_position_fixed(corr, lags, fit_half_width=fit_half_width)
            offsets.append(offset)
            peak_lags.append(peak_lag)
    
        # Filter out large outliers
        offsets = np.array(offsets)
        if np.all(offsets == offsets[0]):
            filtered_offsets = offsets
        else:
            median = np.median(offsets)
            mad = median_abs_deviation(offsets)
            threshold = 5 * mad
            filtered_offsets = offsets[np.abs(offsets - median) < threshold]
        filtered_median = np.median(filtered_offsets)

    
        # If no offsets passed the outlier filter, use all offsets
        if len(filtered_offsets) == 0:
            filtered_offsets = offsets
    
        # Compute 1-sigma uncertainty from percentiles
        lower = np.percentile(filtered_offsets, 16)
        upper = np.percentile(filtered_offsets, 84)
        sigma = (upper - lower) / 2
     
        if chip == 'green':
            self.green_offset = filtered_median
            self.green_offset_sigma = sigma
        elif chip == 'red':
            self.red_offset = filtered_median
            self.red_offset_sigma = sigma

    
        if plot_correlations:
            plt.figure(figsize=(15, 4))
    
            # Plot all cross-correlations
            plt.subplot(1, 3, 1)
            for corr, lags in zip(all_corrs, all_lags):
                plt.plot(lags, corr, alpha=0.5)
            plt.axvline(filtered_median, color='r', linestyle='--', label=f'Median Offset: {filtered_median:.4f}')
            plt.title('All Cross-correlations')
            plt.xlabel('Lag')
            plt.xlim(-120, 120)
            plt.ylabel('Correlation')
            plt.legend()
            plt.grid(True)
    
            # Zoom-in around actual subpixel peak offsets
            plt.subplot(1, 3, 2)
            zoom_range = 5
            for corr, lags, offset in zip(all_corrs, all_lags, offsets):
                center_idx = np.argmin(np.abs(lags - offset))
                if center_idx - zoom_range >= 0 and center_idx + zoom_range + 1 <= len(corr):
                    zoom_lags = lags[center_idx - zoom_range:center_idx + zoom_range + 1]
                    zoom_corr = corr[center_idx - zoom_range:center_idx + zoom_range + 1]
                    plt.plot(zoom_lags, zoom_corr, marker='o', alpha=0.6)
            plt.axvline(filtered_median, color='r', linestyle='--', label='Median Offset')
            plt.title('Zoom-In Around Peaks')
            plt.xlabel('Lag')
            plt.legend()
            plt.grid(True)
    
            # Histogram of offsets
            plt.subplot(1, 3, 3)
            plt.hist(filtered_offsets, bins=20, edgecolor='k', alpha=0.7)
            plt.axvline(filtered_median, color='r', linestyle='--', label='Median Offset')
            plt.title('Offset Histogram')
            plt.xlabel('Offset (pixels)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
    
            plt.tight_layout()
            
            # Create a timestamp and annotate in the lower right corner
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            timestamp_label = f"KPF QLP: {current_time}"
            plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                        fontsize=8, color="darkgray", ha="right", va="bottom",
                        xytext=(0, -50), textcoords='offset points')
    
            # Display the plot
            if fig_path != None:
                plt.savefig(fig_path, dpi=300, facecolor='w')
            if show_plot == True:
                plt.show()
            plt.close('all')


    def plot_2D_image(self, chip=None, variance=False, data_over_sqrt_variance=False,
                            overplot_dark_current=False, blur_size=None, 
                            subtract_master_bias=False,
                            subtract_master_dark=False,
                            fig_path=None, show_plot=False):
        """
        Generate a plot of the a 2D image.  Overlay measurements of 
        dark current or bias measurements in preset regions, if commanded.
        
        Args:
            chip (string) - "green" or "red"
            variance - plot variance (VAR extensions) instead of signal (CCD extensions)
            data_over_sqrt_variance - plot data divided by sqrt(variance), an approximate SNR image
            overlay_dark_current - if True, dark current measurements are over-plotted
            subtract_master_bias - if not False, a master bias will be subtracted
                                   if True, then the path to the master bias is 
                                   the value from the BIASFILE keyword
                                   if his parameter is set to a path, that file 
                                   will be used 
            subtract_master_dark - same as subtract_master_bias
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """
    
        # Subtract master bias frame
        bias_read = False
        dark_read = False
        if subtract_master_bias == False:
            pass
        else:
            if subtract_master_bias == True:
                if ('BIASDIR' in self.header) and ('BIASFILE' in self.header):
                    if self.header['BIASDIR'].startswith("/masters"):
                        pre = '/data'
                    else:
                        pre = ''
                    bias_filename = pre + self.header['BIASDIR'] + '/' + self.header['BIASFILE']
            else:
            	bias_filename = subtract_master_bias
            
            try:
                D2_master_bias = KPF0.from_fits(bias_filename)
                bias_read = True
            except Exception as e:
                self.logger.error(f"Problem reading {bias_filename}: {e}\n{traceback.format_exc()}")
            if bias_read:
                for extension in ['GREEN_CCD', 'RED_CCD']:
                    if extension in D2_master_bias.extensions:
                        self.D2[extension] -= D2_master_bias[extension]
            else:
                subtract_master_bias = False
        
        # Subtract scaled master dark frame
        if subtract_master_dark == False:
            pass
        else:
            if subtract_master_dark == True:
                if ('DARKDIR' in self.header) and ('DARKFILE' in self.header):
                    if self.header['DARKDIR'].startswith("/masters"):
                        pre = '/data'
                    else:
                        pre = ''
                    dark_filename = pre + self.header['DARKDIR'] + '/' + self.header['DARKFILE']
            else:
            	dark_filename = subtract_master_dark
            
            try:
                D2_master_dark = KPF0.from_fits(dark_filename)
                dark_read = True
            except Exception as e:
                self.logger.error(f"Problem reading {dark_filename}: {e}\n{traceback.format_exc()}")
            if dark_read:
                for extension in ['GREEN_CCD', 'RED_CCD']:
                    if extension in D2_master_dark.extensions:
                        self.D2[extension] -= D2_master_dark[extension] * float(self.exptime)
            else:
                subtract_master_dark = False
        
        chip = chip.lower()

        # Set parameters based on the chip selected
        if chip == 'green' or chip == 'red':
            if chip == 'green':
                CHIP = 'GREEN'
                chip_title = 'Green'
                if overplot_dark_current:
                    reg = self.green_dark_current_regions
                    coll_pressure_torr = self.green_coll_pressure_torr
                    ech_pressure_torr = self.green_ech_pressure_torr
                    coll_current_a = self.green_coll_current_a
                    ech_current_a = self.green_ech_current_a
            if chip == 'red':
                CHIP = 'RED'
                chip_title = 'Red'
                if overplot_dark_current:
                    reg = self.red_dark_current_regions
                    coll_pressure_torr = self.red_coll_pressure_torr
                    ech_pressure_torr = self.red_ech_pressure_torr
                    coll_current_a = self.red_coll_current_a
                    ech_current_a = self.red_ech_current_a
            if variance:
                image = np.array(self.D2[CHIP + '_VAR'].data)
            elif data_over_sqrt_variance:
                image = np.array(self.D2[CHIP + '_CCD'].data) / np.sqrt(np.array(self.D2[CHIP + '_VAR'].data))
            else:
                image = np.array(self.D2[CHIP + '_CCD'].data)
            
        else:
            self.logger.debug('chip not supplied.  Exiting plot_2D_image')
            return
            
        if blur_size != None:
            from scipy import ndimage
            image = ndimage.median_filter(image, size=blur_size)

        # Generate 2D image
        plt.figure(figsize=(10,8), tight_layout=True)
        vmin = np.nanpercentile(image[100:-100,100:-100],0.1)
        vmax = np.nanpercentile(image[100:-100,100:-100],99)
        if variance or data_over_sqrt_variance:
            vmin = np.nanpercentile(image, 0.01)
            vmax = np.nanpercentile(image,99.99)
        plt.imshow(image, vmin=vmin, vmax=vmax, 
                          interpolation = 'None', 
                          origin = 'lower', 
                          cmap='viridis')
        plt.grid(False)
        title_txt = '2D - ' + chip_title + ' CCD: ' + str(self.ObsID) + ' - ' + self.name
        if variance:
            title_txt = title_txt + ' (Variance)'
        elif data_over_sqrt_variance:
            title_txt = title_txt + ' (SNR)'
        if bias_read:
            title_txt = title_txt + ' (Bias subtracted)'
        if dark_read:
            title_txt = title_txt + ' (Dark subtracted)'
        plt.title(title_txt, fontsize=14)
        plt.xlabel('Column (pixel number)', fontsize=18)
        plt.ylabel('Row (pixel number)', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        label = 'Counts (e-)'
        if variance:
            label = 'Variance (e-)'
        elif data_over_sqrt_variance:
            label = r'Counts (e-) / (Variance (e-))$^{0.5}$'
        cbar = plt.colorbar(label = label)
        cbar.ax.yaxis.label.set_size(18)
        cbar.ax.tick_params(labelsize=14)
        
        if overplot_dark_current:
            if self.exptime > 0:
                exptype = 'dark'
                timelabel = ' e$^-$ hr$^{-1}$'
                image *= (3600./self.exptime)  # convert to e- per hour
            else: 
                exptype = 'bias'
                timelabel = ' e$^-$'
            # Plot regions
            for r in reg.keys():
                plt.gca().add_patch(Rectangle((reg[r]['x1'],reg[r]['y1']),reg[r]['x2']-reg[r]['x1'],reg[r]['y2']-reg[r]['y1'],linewidth=1,edgecolor='r',facecolor='none'))
                plt.text(((reg[r]['short'] == 'ref3') or
                          (reg[r]['short'] == 'ref4') or
                          (reg[r]['short'] == 'ref5') or
                          (reg[r]['short'] == 'ref6') or
                          (reg[r]['short'] == 'amp1'))*(reg[r]['x1'])+
                         ((reg[r]['short'] == 'ref1') or
                          (reg[r]['short'] == 'ref2') or
                          (reg[r]['short'] == 'ech')  or
                          (reg[r]['short'] == 'coll') or
                          (reg[r]['short'] == 'amp2'))*(reg[r]['x2']),
                         (((reg[r]['y1'] < 2080) and (reg[r]['y1'] > 100))*(reg[r]['y1']-30)+
                          ((reg[r]['y1'] > 2080) or  (reg[r]['y1'] < 100))*(reg[r]['y2']+30)),
                         str(np.round(reg[r]['med_elec'],1)) + timelabel,
                         size=16,
                         weight='bold',
                         color='r',
                         ha=(((reg[r]['short'] == 'ref3') or
                              (reg[r]['short'] == 'ref4') or
                              (reg[r]['short'] == 'ref5') or
                              (reg[r]['short'] == 'ref6') or
                              (reg[r]['short'] == 'amp1'))*('left')+
                             ((reg[r]['short'] == 'ref1') or
                              (reg[r]['short'] == 'ref2') or
                              (reg[r]['short'] == 'ech')  or
                              (reg[r]['short'] == 'coll') or
                              (reg[r]['short'] == 'amp2'))*('right')),
                         va=(((reg[r]['y1'] < 2080) and (reg[r]['y1'] > 100))*('top')+
                             ((reg[r]['y1'] > 2080) or (reg[r]['y1'] < 100))*('bottom'))
                        )
            coll_text = 'Ion Pump (Coll): \n' + (f'{coll_pressure_torr:.1e}' + ' Torr, ' + f'{coll_current_a*1e6:.1f}' + ' $\\mu$A')*(coll_pressure_torr > 1e-9) + ('Off')*(coll_pressure_torr < 1e-9)
            ech_text  = 'Ion Pump (Ech): \n'  + (f'{ech_pressure_torr:.1e}'  + ' Torr, ' + f'{ech_current_a*1e6:.1f}'  + ' $\\mu$A')*(ech_pressure_torr  > 1e-9) + ('Off')*(ech_pressure_torr < 1e-9)
            now = datetime.now()
            plt.text(4080, -250, now.strftime("%m/%d/%Y, %H:%M:%S"), ha='right', color='gray')
            plt.text(4220,  500, coll_text, size=11, rotation=90, ha='center')
            plt.text(4220, 3000, ech_text,  size=11, rotation=90, ha='center')
            plt.text(3950, 1500, 'Bench Side\n (blue side of orders)', size=14, rotation=90, ha='center', color='white')
            plt.text( 150, 1500, 'Top Side\n (red side of orders)',    size=14, rotation=90, ha='center', color='white')
            plt.text(2040,   70, 'Collimator Side',                    size=14, rotation= 0, ha='center', color='white')
            plt.text(2040, 3970, 'Echelle Side',                       size=14, rotation= 0, ha='center', color='white')
         
        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    xytext=(0, -35), textcoords='offset points')
        if bias_read:
            bias_label = 'Bias: ' + bias_filename
            plt.annotate(bias_label, xy=(-0.1, 0), xycoords='axes fraction', 
                        fontsize=8, color="darkgray", ha="left", va="bottom",
                        xytext=(0, -52), textcoords='offset points')
        if dark_read:
            dark_label = 'Dark: ' + dark_filename
            plt.annotate(dark_label, xy=(-0.1, 0), xycoords='axes fraction', 
                        fontsize=8, color="darkgray", ha="left", va="bottom",
                        xytext=(0, -52), textcoords='offset points')
            plt.subplots_adjust(bottom=0.1)     

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=600, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')

# AWH will fix up this code at some point
# to make a 1:1 pixel relationship
#    def plot_2D_image_test(self, chip=None, overplot_dark_current=False, blur_size=None, 
#                            fig_path=None, show_plot=False):
#        """
#        Generate a plot of a 2D image.  Overlay measurements of 
#        dark current or bias measurements in preset regions, if commanded.
#        
#        Args:
#            chip (string) - "green" or "red"
#            overplot_dark_current - if True, dark current measurements are over-plotted
#            fig_path (string) - set to the path for the file to be generated.
#            show_plot (boolean) - show the plot in the current environment.
#
#        Returns:
#            PNG plot in fig_path or shows the plot it in the current environment 
#            (e.g., in a Jupyter Notebook).
#
#        """
#
#        # Set parameters based on the chip selected
#        if chip == 'green' or chip == 'red':
#            if chip == 'green':
#                CHIP = 'GREEN'
#                chip_title = 'Green'
#                if overplot_dark_current:
#                    reg = self.green_dark_current_regions
#                    coll_pressure_torr = self.green_coll_pressure_torr
#                    ech_pressure_torr = self.green_ech_pressure_torr
#                    coll_current_a = self.green_coll_current_a
#                    ech_current_a = self.green_ech_current_a
#            if chip == 'red':
#                CHIP = 'RED'
#                chip_title = 'Red'
#                if overplot_dark_current:
#                    reg = self.red_dark_current_regions
#                    coll_pressure_torr = self.red_coll_pressure_torr
#                    ech_pressure_torr = self.red_ech_pressure_torr
#                    coll_current_a = self.red_coll_current_a
#                    ech_current_a = self.red_ech_current_a
#            image = np.array(self.D2[CHIP + '_CCD'].data)
#        else:
#            self.logger.debug('chip not supplied.  Exiting plot_2D_image')
#            return
#            
#        if blur_size != None:
#            from scipy import ndimage
#            image = ndimage.median_filter(image, size=blur_size)
#
#        # Assuming image is a numpy array of shape (4080, 4080)
#        dpi = 100  # Set the DPI to 100, so the number of inches equals the number of pixels divided by DPI
#        figsize = image.shape[1] / dpi, image.shape[0] / dpi  # Calculate the figure size in inches
#        
#        # Create a figure with the exact size and DPI
#        fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
#        
#        # Add an axis at position [left, bottom, width, height] where all quantities are in fractions of figure width and height
#        ax = fig.add_axes([0, 0, 1, 1])
#        
#        # Hide the axes
#        ax.axis('off')
#        
#        # Display the image
#        im = ax.imshow(image, vmin = np.nanpercentile(image[100:-100,100:-100],0.1), 
#                               vmax = np.nanpercentile(image[100:-100,100:-100],99), 
#                               interpolation = 'None', 
#                               origin = 'lower', 
#                               cmap='viridis',
#                               aspect='equal')
#
#        # ... (rest of your plotting code, e.g., overplot_dark_current)
#
#        # Display the plot
#        if fig_path is not None:
#            t0 = time.process_time()
#            plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor='w')
#            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
#        if show_plot:
#            plt.show()
#        plt.close('all')


    def plot_2D_image_zoom(self, chip=None, fig_path=None, show_plot=False, 
                           zoom_coords=(3780, 3780, 4080, 4080)):
        """
        Generate a zoom-in plot of the a 2D image.  

        Args:
            zoom_coords - coordinates for zoom (xmin, ymin, xmax, ymax)
            chip (string) - "green" or "red"
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """
        chip = chip.lower()

        # Set parameters based on the chip selected
        if chip == 'green' or chip == 'red':
            if chip == 'green':
                CHIP = 'GREEN'
                chip_title = 'Green'
                reg = self.green_dark_current_regions
            if chip == 'red':
                CHIP = 'RED'
                chip_title = 'Red'
                reg = self.red_dark_current_regions
            image = np.array(self.D2[CHIP + '_CCD'].data)
        else:
            self.logger.debug('chip not supplied.  Exiting plot_2D_image')
            return
        
        # Plot and annotate
        plt.figure(figsize=(10,8), tight_layout=True)
        plt.imshow(image[zoom_coords[0]:zoom_coords[2], zoom_coords[1]:zoom_coords[3]], 
                   extent=[zoom_coords[0], zoom_coords[2], zoom_coords[1], zoom_coords[3]], 
                   vmin = np.nanpercentile(image,0.1), 
                   vmax = np.nanpercentile(image,99.9), 
                   interpolation = 'None', 
                   origin = 'lower')
        plt.title('2D - ' + chip_title + ' CCD: ' + str(self.ObsID) + ' - ' + self.name, fontsize=18)
        plt.xlabel('Column (pixel number)', fontsize=18)
        plt.ylabel('Row (pixel number)', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        cbar = plt.colorbar(label = 'Counts (e-)')
        cbar.ax.yaxis.label.set_size(18)
        cbar.ax.tick_params(labelsize=14)
        
        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=200, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def plot_2D_image_zoom_3x3(self, chip=None, fig_path=None, show_plot=False):
        """
        Generate a 3x3 array zoom-in plots of the a 2D image.  

        Args:
            chip (string) - "green" or "red"
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """
        chip = chip.lower()

        # Set parameters based on the chip selected
        if chip == 'green' or chip == 'red':
            if chip == 'green':
                CHIP = 'GREEN'
                chip_title = 'Green'
                #reg = self.green_dark_current_regions
            if chip == 'red':
                CHIP = 'RED'
                chip_title = 'Red'
                #reg = self.red_dark_current_regions
            image = np.array(self.D2[CHIP + '_CCD'].data)
        else:
            self.logger.info('chip not supplied.  Exiting plot_2D_image')
            return
                
        # Calculate center of the image and define offsets
        center_x, center_y = np.array(image.shape[:2]) // 2
        size = 400
        sep = 1840
        offsets = [-sep, 0, sep]
        
        # Generate the array of 2D images
        fig, axs = plt.subplots(3, 3, figsize=(10, 8)) 
        
        for i in range(3):
            for j in range(3):
                start_x = center_x - size // 2 + offsets[i]
                start_y = center_y - size // 2 + offsets[j]
                sub_img = image[start_x:start_x+size, start_y:start_y+size]
                im = axs[2-i, j].imshow(sub_img, origin='lower',
                                        extent=[start_y, start_y+size, start_x, start_x+size],
                                        vmin=np.nanpercentile(sub_img, 0.1),
                                        vmax=np.nanpercentile(sub_img, 99.9),
                                        interpolation='none', cmap='viridis')
                axs[2-i, j].grid(False)
                axs[2-i, j].tick_params(top=False, right=False, labeltop=False, labelright=False, labelsize=9)
                if i != 0:
                    axs[2-i, j].tick_params(labelbottom=False)
                if j != 0:
                    axs[2-i, j].tick_params(labelleft=False)
        
                # Add colorbar for each subplot
                cbar = fig.colorbar(im, ax=axs[2-i, j], fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=7)

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.15, hspace=0.15)
        
        # Use suptitle and fig.text for cleaner labels
        fig.suptitle(f'2D - {chip_title} CCD: {self.ObsID} - {self.name}', fontsize=14)
        fig.text(0.5, 0.04, 'Column (pixel number)', ha='center', fontsize=16)
        fig.text(0.06, 0.5, 'Row (pixel number)', va='center', rotation='vertical', fontsize=16)
        plt.subplots_adjust(top=0.945, bottom=0.10)  # or tweak to 0.91, 0.93, etc.
        
        # Timestamp annotation
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.95, 0.01, f"KPF QLP: {current_time}", fontsize=8, color="darkgray", ha="right", va="bottom")

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=300, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def plot_2D_order_trace2x2(self, chip=None, order_trace_master_file='auto',
                               fig_path=None, show_plot=False, 
                               width=200, height=200, 
                               start_x_arr='default', start_y_arr='default'):
        """
        Overlay the order trace on the 2D image in a 3x3 array of zoom-in plots.  

        Args:
            chip (string) - "green" or "red"
            order_trace_master_file (string) - path to order trace file; auto means use defaults
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).
        """
        chip = chip.lower()
        
        # Set parameters based on the chip selected
        obs_date = Time(self.header['DATE-MID'])
        service_mission_date1 = Time('2024-02-03', format='iso', scale='utc')
        if chip == 'green' or chip == 'red':
            if chip == 'green':
                CHIP = 'GREEN'
                chip_title = 'Green'
                if order_trace_master_file == 'auto':
                    if obs_date < service_mission_date1:
                        order_trace_master_file = '/data/reference_fits/kpf_20230920_master_flat_GREEN_CCD.csv'
                    else:
                        order_trace_master_file = '/data/reference_fits/kpf_20240206_master_flat_GREEN_CCD.csv'
                if start_x_arr == 'default':
                    start_x_arr = [ 0, 3600,  0, 3600]
                if start_y_arr == 'default':
                    start_y_arr = [1200, 1200,  545,  545]
            if chip == 'red':
                CHIP = 'RED'
                chip_title = 'Red'
                if order_trace_master_file == 'auto':
                    if obs_date < service_mission_date1:
                        order_trace_master_file = '/data/reference_fits/kpf_20230920_master_flat_RED_CCD.csv'
                    else:
                        order_trace_master_file = '/data/reference_fits/kpf_20240206_master_flat_RED_CCD.csv'
                if start_x_arr == 'default':
                    start_x_arr = [ 0, 3600,  0, 3600]
                if start_y_arr == 'default':
                    start_y_arr = [1538, 1538,  545,  550]
            image = np.array(self.D2[CHIP + '_CCD'].data)
            order_trace_master = pd.read_csv(order_trace_master_file)
        else:
            self.logger.info('chip not supplied.  Exiting plot_2D_image')
            return
                
        # Generate the array of 2D images
        fig, axs = plt.subplots(2, 2, figsize=(19,17), tight_layout=False)
        for i in range(2):
            for j in range(2):
                # Calculate the top left corner of each subimage
                start_x = start_x_arr[2*i+j]
                start_y = start_y_arr[2*i+j]
                end_x = min(start_x+width,  image.shape[0])
                end_y = min(start_y+height, image.shape[1])

                # Slice out and display the subimage
                sub_img = image[start_y:end_y, start_x:end_x]
                im = axs[i, j].imshow(sub_img, origin='lower', 
                                 extent=[start_x, end_x, start_y, end_y], # these indices appear backwards, but work
                                 vmin = np.nanpercentile(sub_img,0.1), 
                                 vmax = np.nanpercentile(sub_img,99.9),
                                 interpolation = 'None',
                                 cmap='viridis')
                axs[i, j].set_xlim(start_x, start_x+width)
                axs[i, j].set_ylim(start_y, start_y+height)
                
                # Overplot order trace
                for o in range(1,np.shape(order_trace_master)[0]-2,1):#[50]:#range(np.shape(order_trace)[0])
                    x_grid_master = np.linspace(order_trace_master.iloc[o]['X1'], 
                                                order_trace_master.iloc[o]['X2'], 
                                                int(order_trace_master.iloc[o]['X2']-order_trace_master.iloc[o]['X1']))
                    x_grid_master = x_grid_master[x_grid_master >= start_x]
                    x_grid_master = x_grid_master[x_grid_master <= start_x+width]
                    y_grid_master = order_trace_master.iloc[o]['Coeff0'] + \
                                    order_trace_master.iloc[o]['Coeff1'] * x_grid_master + \
                                    order_trace_master.iloc[o]['Coeff2'] * x_grid_master**2 + \
                                    order_trace_master.iloc[o]['Coeff3'] * x_grid_master**3         
                    axs[i, j].plot(x_grid_master, y_grid_master,                                          color='red',   linewidth=1.2, linestyle='--')
                    axs[i, j].plot(x_grid_master, y_grid_master-order_trace_master.iloc[o]['BottomEdge'], color='white', linewidth=1.2, linestyle='--', alpha=1)
                    axs[i, j].plot(x_grid_master, y_grid_master+order_trace_master.iloc[o]['TopEdge'],    color='white', linewidth=1.2, linestyle='--', alpha=1)                
                
                axs[i, j].grid(False)
                axs[i, j].tick_params(top=False, right=False, labeltop=False, labelright=False)
                axs[i, j].tick_params(axis='x', labelsize=14)
                axs[i, j].tick_params(axis='y', labelsize=14)
                cbar = fig.colorbar(im, ax=axs[i, j], fraction=0.04, pad=0.04) # Adjust the fraction and pad for proper placement
                cbar.ax.tick_params(labelsize=12)

        plt.grid(False)
        ax = fig.add_subplot(111, frame_on=False)
        ax.grid(False)
        ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        ax.set_title('Order Trace - ' + chip_title + ' CCD: ' + str(self.ObsID), fontsize=24)
        ax.set_xlabel('Column (pixel number)', fontsize=24, labelpad=20)
        ax.set_ylabel('Row (pixel number)', fontsize=24, labelpad=25)

        # Annotations
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=16, color="darkgray", ha="right", va="bottom",
                    xytext=(-50, -150), textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)     
        plt.annotate('Trace = ' + order_trace_master_file, xy=(0, 0), xycoords='axes fraction', 
                    fontsize=16, color="darkgray", ha="left", va="bottom",
                    xytext=(50, -150), textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=400, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def plot_bias_histogram(self, variance=False, data_over_sqrt_variance=False, chip=None, fig_path=None, show_plot=False):
        """
        Plot a histogram of the counts per pixel in a 2D image.  

        Args:
            variance - plot variance (VAR extensions) instead of signal (CCD extensions)
            data_over_sqrt_variance - plot data divided by sqrt(variance), an approximate SNR image
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """

        if chip == 'green' or chip == 'red':
            if chip == 'green':
                chip_title = 'Green'
                CHIP = 'GREEN'
            if chip == 'red':
                chip_title = 'Red'
                CHIP = 'RED'

        if variance:
            image = np.array(self.D2[CHIP + '_VAR'].data)
        elif data_over_sqrt_variance:
            image = np.array(self.D2[CHIP + '_CCD'].data) / np.sqrt(np.array(self.D2[CHIP + '_VAR'].data))
        else:
            image = np.array(self.D2[CHIP + '_CCD'].data)

        if not data_over_sqrt_variance and not variance:
            histmin = -40
            histmax = 40
        else:
            histmin = int(np.floor(np.percentile(image,0)))
            histmax = int(np.ceil(np.percentile(image,99.995)))

        flattened = image.flatten()
        flattened = flattened[(flattened >= histmin) & (flattened <= histmax)]
        
        # Fit a normal distribution to the data
        if not data_over_sqrt_variance and not variance:
            mu, std = norm.fit(flattened)
            median = np.median(flattened)

        #innermin = -15
        #innermax = 15
        #flattened_inner = flattened[(flattened >= innermin) & (flattened <= innermax)]
        #mu, std = norm.fit(flattened_inner)
        #median = np.median(flattened_inner)
        

#        # Define the model: sum of two Gaussians
#        def gaussian(x, mu, sigma, amplitude):
#            return amplitude * norm.pdf(x, mu, sigma)
#        
#        def model(params, x):
#            mu1, mu2, sigma1, sigma2, amplitude1, amplitude2 = params
#            return gaussian(x, mu1, sigma1, amplitude1) + gaussian(x, mu2, sigma2, amplitude2)
#        
#        # Define the log-probability function
#        def log_prob(params, x, y):
#            model_y = model(params, x)
#            #sigma = params[2] + params[3]
#            sigma_y = np.sqrt(y+1)
#            return -0.5 * np.sum((y - model_y) ** 2 / sigma_y ** 2 + np.log(sigma_y ** 2))
#        
#        # Run the MCMC estimation
#        ndim = 6  # number of parameters in the model
#        nwalkers = 50  # number of MCMC walkers
#        nburn = 1000  # "burn-in" period to let chains stabilize
#        nsteps = 5000  # number of MCMC steps to take
#        
#        # set up initial guess and run MCMC
#        guess = np.array([0, 0, 3, 10, 100000, 1000]) + 0.1 * np.random.randn(nwalkers, ndim)
#        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[flattened, model(guess, flattened)])
#        sampler.run_mcmc(guess, nsteps)
#        
#        # Discard burn-in and flatten the samples
#        samples = sampler.chain[:, nburn:, :].reshape(-1, ndim)
#        
#        # Make a corner plot with the posterior distribution
#        fig, ax = plt.subplots(ndim, figsize=(10, 7), tight_layout=True)
#        corner.corner(samples, labels=["mu1", "mu2", "sigma1", "sigma2", "amplitude1", "amplitude2"], truths=guess[0], ax=ax)
#        plt.show()        
        
        # Create figure with specified size
        fig, ax = plt.subplots(figsize=(7,5))
        
        # Create histogram with log scale
        n, bins, patches = plt.hist(flattened, bins=range(histmin, histmax+1), color='gray', log=True)
                
        if not data_over_sqrt_variance and not variance:
            # Plot the distribution
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, histmax-histmin+1)
            p = norm.pdf(x, mu, std) * len(flattened) * np.diff(bins)[0] # scale the PDF to match the histogram
            plt.plot(x, p, 'r', linewidth=2)
    
            # Add annotations
            textstr = '\n'.join((
                r'$\\mu=%.2f$ e-' % (mu, ),
                r'$\\sigma=%.2f$ e-' % (std, ),
                r'$\mathrm{median}=%.2f$ e-' % (median, )))
            props = dict(boxstyle='round', facecolor='red', alpha=0.15)
            plt.gca().text(0.98, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                    verticalalignment='top', horizontalalignment='right', bbox=props)
        
        # Set up axes
        if not data_over_sqrt_variance and not variance:
            ax.axvline(x=0, color='blue', linestyle='--')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(histmin, histmax)
        plt.ylim(5*10**-1, 10**7)
        #plt.title(str(self.ObsID) + ' - ' + self.name, fontsize=14)
        plt.title('2D - ' + chip_title + ' CCD: ' + str(self.ObsID) + ' - ' + self.name, fontsize=14)
        if variance:
            xlabel = 'Variance (e-)'
        elif data_over_sqrt_variance:
            xlabel = r'Counts (e-) / (Variance (e-))$^{0.5}$'
        else:
            xlabel = 'Counts (e-)'

        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel('Number of Pixels (log scale)', fontsize=14)
        plt.tight_layout()
 
        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    xytext=(0, -40), textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)     

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=200, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')

        
    def plot_bias_histogram2(self, chip=None, fig_path=None, show_plot=False):
        """
        Plot a histogram of the counts per pixel in a 2D image.  

        Args:
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """

        image = np.array(self.D2[CHIP + '_CCD'].data)
        if chip == 'green' or chip == 'red':
            if chip == 'green':
                CHIP = 'GREEN_CCD'
                chip_title = 'Green'
            if chip == 'red':
                CHIP = 'RED_CCD'
                chip_title = 'Red'

        histmin = -50
        histmax = 50
        flattened = self.D2[CHIP].data.flatten()
        flattened = flattened[(flattened >= histmin) & (flattened <= histmax)]
        
        # Fit a normal distribution to the data
        mu, std = norm.fit(flattened)
        median = np.median(flattened)

        innermin = -10
        innermax = 10
        flattened_inner = flattened[(flattened >= innermin) & (flattened <= innermax)]
        mu, std = norm.fit(flattened_inner)
        median = np.median(flattened_inner)
        
        # Create figure with specified size
        fig, ax = plt.subplots(figsize=(7,5))
        
        # Create histogram with log scale
        n, bins, patches = plt.hist(flattened, bins=range(histmin, histmax+1), color='gray', log=True)
        
        # Plot the distribution
        xmin, xmax = plt.xlim()
        x = np.linspace(innermin, innermax, innermax-innermin+1)
        p = norm.pdf(x, mu, std) * len(flattened) * np.diff(bins)[0] # scale the PDF to match the histogram
        plt.plot(x, p, 'r', linewidth=2)
        
        # Add annotations
        textstr = '\n'.join((
            r'$\\mu=%.2f$ e-' % (mu, ),
            r'$\\sigma=%.2f$ e-' % (std, ),
            r'$\mathrm{median}=%.2f$ e-' % (median, )))
        props = dict(boxstyle='round', facecolor='red', alpha=0.15)
        plt.gca().text(0.98, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        # Set up axes
        ax.axvline(x=0, color='blue', linestyle='--')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(histmin, histmax)
        plt.ylim(5*10**-1, 10**7)
        #plt.title(str(self.ObsID) + ' - ' + self.name, fontsize=14)
        plt.title('2D - ' + chip_title + ' CCD: ' + str(self.ObsID) + ' - ' + self.name, fontsize=14)
        plt.xlabel('Counts (e-)', fontsize=14)
        plt.ylabel('Number of Pixels (log scale)', fontsize=14)
        plt.tight_layout()

        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    xytext=(0, -40), textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)     

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=200, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')

    
    def plot_2D_image_histogram(self, chip=None, 
                                      subtract_master_bias=False,
                                      subtract_master_dark=False, 
                                      saturation_limit_2d=240000,
                                      fig_path=None, show_plot=False):
        """
        Make a histogram of the pixel intensities for a 2D image.

        Args:
            chip (string) - "green" or "red"
            subtract_master_bias - if not False, a master bias will be subtracted
                                   if True, then the path to the master bias is 
                                   the value from the BIASFILE keyword
                                   if his parameter is set to a path, that file 
                                   will be used 
            subtract_master_dark - same as subtract_master_bias
            fig_path (string) - set to the path for the file 
                to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """
    
        # Subtract master bias frame
        bias_read = False
        dark_read = False
        if subtract_master_bias == False:
            pass
        else:
            D2_subtracted = copy.deepcopy(self.D2)
            if subtract_master_bias == True:
                if ('BIASDIR' in self.header) and ('BIASFILE' in self.header):
                    if self.header['BIASDIR'].startswith("/masters"):
                        pre = '/data'
                    else:
                        pre = ''
                    bias_filename = pre + self.header['BIASDIR'] + '/' + self.header['BIASFILE']
            else:
            	bias_filename = subtract_master_bias
            
            try:
                D2_master_bias = KPF0.from_fits(bias_filename)
                bias_read = True
            except Exception as e:
                self.logger.error(f"Problem reading {bias_filename}: {e}\n{traceback.format_exc()}")
            if bias_read:
                for extension in ['GREEN_CCD', 'RED_CCD']:
                    if extension in D2_master_bias.extensions:
                        D2_subtracted[extension] -= D2_master_bias[extension]
            else:
                subtract_master_bias = False
        
        # Subtract scaled master dark frame
        if subtract_master_dark == False:
            pass
        else:
            D2_subtracted = copy.deepcopy(self.D2)
            if subtract_master_dark == True:
                if ('DARKDIR' in self.header) and ('DARKFILE' in self.header):
                    if self.header['DARKDIR'].startswith("/masters"):
                        pre = '/data'
                    else:
                        pre = ''
                    dark_filename = pre + self.header['DARKDIR'] + '/' + self.header['DARKFILE']
            else:
            	dark_filename = subtract_master_dark
            
            try:
                D2_master_dark = KPF0.from_fits(dark_filename)
                dark_read = True
            except Exception as e:
                self.logger.error(f"Problem reading {dark_filename}: {e}\n{traceback.format_exc()}")
            if dark_read:
                for extension in ['GREEN_CCD', 'RED_CCD']:
                    if extension in D2_master_dark.extensions:
                        D2_subtracted[extension] -= D2_master_dark[extension] * float(self.exptime)
            else:
                subtract_master_dark = False
        
        # Set parameters based on the chip selected
        chip = chip.lower()
        if chip == 'green' or chip == 'red':
            if chip == 'green':
                CHIP = 'GREEN'
                chip_title = 'Green'
            if chip == 'red':
                CHIP = 'RED'
                chip_title = 'Red'
            image = np.array(self.D2[CHIP + '_CCD'].data)
            # Flatten the image array for speed in histrogram computation
            flatten_image = image.flatten()
            
            if bias_read or dark_read:
                image_subtracted = np.array(D2_subtracted[CHIP + '_CCD'].data)
                flatten_image_subtracted = image_subtracted.flatten()
                mad_subtracted = median_abs_deviation(flatten_image_subtracted, nan_policy='omit')
        else:
            self.logger.info(f'chip not supplied. Exiting plot_2D_image_histogram()')
            return

        plt.figure(figsize=(8,5), tight_layout=True)
        bins = 100
        mad = median_abs_deviation(flatten_image, nan_policy='omit')
        if mad < 100:
            bins = 80
        if mad < 20:
            bins = 60
        if mad < 10:
            bins = 40
        plt.hist(flatten_image, 
                 bins=bins, 
                 label='2D: Median: ' + '%4.1f' % np.nanmedian(flatten_image) + ' e-; '
                       'Stddev: ' + '%4.1f' % np.nanstd(flatten_image) + ' e-; '
                       'MAD: '    + '%4.1f' % mad + ' e-; '
                       'Saturated? ' + str(np.nanpercentile(flatten_image,99.99)>saturation_limit_2d), 
                 alpha=0.5, 
                 density = False, 
                 range = (np.nanpercentile(flatten_image,  0.005),
                          np.nanpercentile(flatten_image, 99.995)))
        if bias_read or dark_read:
            plt.hist(flatten_image_subtracted, 
                     bins=bins, 
                     label='2D - master '  + bias_read*'bias' + dark_read*'dark' + ': Median: ' + '%4.1f' % np.nanmedian(flatten_image_subtracted) + ' e-; '
                           'Stddev: ' + '%4.1f' % np.nanstd(flatten_image_subtracted) + ' e-; '
                           'MAD: '    + '%4.1f' % mad + ' e-; '
                           'Saturated? ' + str(np.nanpercentile(flatten_image_subtracted,99.99)>saturation_limit_2d), 

                     histtype='step',      
                     color='red',          
                     linewidth=2.0,        
                     alpha=1.0,            
                     density=False,

                     range = (np.nanpercentile(flatten_image_subtracted,  0.005),
                              np.nanpercentile(flatten_image_subtracted, 99.995)))
        plt.title('2D - ' + chip_title + ' CCD: ' + str(self.ObsID) + ' - ' + self.name, fontsize=18)
        plt.xlabel('Counts (e-)', fontsize=16)
        plt.ylabel('Number of Pixels', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.yscale('log')
        if bias_read or dark_read:
            plt.legend(loc='lower right', fontsize=10)
        else:
            plt.legend(loc='lower right', fontsize=10)

         
        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    xytext=(0, -40), textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)     
        
        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=144, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')
        
        
    def plot_2D_column_cut(self, chip=None, fig_path=None, show_plot=False,
                           column_brightness_percentile=50, saturation_limit_2d=240000):
        """
        Create a plot of cuts through columns corresponding to the 10th, 50th, and 90th
        percentiles for total flux

        Args:
            chip (string) - "green" or "red"
            fig_path (string) - set to the path for the file 
                to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).
        """
        
        # Set parameters based on the chip selected
        if chip == 'green' or chip == 'red':
            if chip == 'green':
                CHIP = 'GREEN'
                chip_title = 'Green'
            if chip == 'red':
                CHIP = 'RED'
                chip_title = 'Red'
            image = np.array(self.D2[CHIP + '_CCD'].data)
            column_sum = np.nansum(image, axis = 0)
            p_10 = np.nanpercentile(column_sum, 10) # 10th percentile
            p_50 = np.nanpercentile(column_sum, 50) # 50th percentile
            p_90 = np.nanpercentile(column_sum, 99) # 99th percentile
            percentile = np.nanpercentile(column_sum, column_brightness_percentile) # nth percentile
            which_column_10 = np.argmin(np.abs(column_sum - p_10)) # index of 50th percentile
            which_column_50 = np.argmin(np.abs(column_sum - p_50)) # index of 50th percentile
            which_column_90 = np.argmin(np.abs(column_sum - p_90)) # index of 90th percentile
            which_column = np.argmin(np.abs(column_sum - percentile)) # index of nth percentile
        else:
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
            return
            
        # Determine if plot should be logarithmic or not
        if np.nanpercentile(column_sum, 90) / np.nanpercentile(column_sum, 10) > 20:
            log_plot = True
        else:
            log_plot = False

        plt.figure(figsize=(12,5), tight_layout=True)
        plt.plot(image[:,which_column_90], 
                 alpha = 0.5, 
                 linewidth =  0.75, 
                 label = 'Column ' + str(which_column_90) + ' (90% max brightness)', 
                 color = 'Red')
        plt.plot(image[:,which_column_50], 
                 alpha = 0.5, 
                 linewidth =  0.75, 
                 label = 'Column ' + str(which_column_50) + ' (50% max brightness)', 
                 color = 'Orange')
        plt.plot(image[:,which_column_10], 
                 alpha = 0.5, 
                 linewidth =  0.75, 
                 label = 'Column ' + str(which_column_10) + ' (10% max brightness)', 
                 color = 'Green')
         # Only show the saturation limit if the it's close
        if max(image[:,which_column_90]) > 0.3 * saturation_limit_2d: 
            plt.axhline(y=saturation_limit_2d, color='r', linestyle='-') # Saturation limit
            plt.text(100, 0.9*saturation_limit_2d, 'Saturation', fontsize=12, verticalalignment='top', color='r')
            plt.ylim(0.5, 1.2*saturation_limit_2d)
        plt.title('Column Cuts though 2D ' + chip_title + ' CCD: ' + str(self.ObsID) + ' - ' + self.name, fontsize=18)
        plt.ylabel('e-', fontsize=16)
        plt.xlabel('Row Number', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(0, len(column_sum))
        if log_plot:
            plt.yscale('log')
            y_lim = plt.ylim()
            plt.ylim(0.9, y_lim[1])
        plt.legend( fontsize=12)    
         
        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    xytext=(0, -40), textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)     

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=400, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


class Analyze2D_2files:
    """
    This class contains functions to analyze 2D images (storing them
    as attributes) and functions to plot the results.
    Some of the functions need to be filled in

    Arguments:
        D2a - first 2D object
        D2b - second 2D object

    Attributes:
        header_a - header of D2a
        header_b - header of D2b
        name_a - name of source (e.g., 'Bias', 'Etalon', '185144') of D2a
        name_b - name of source (e.g., 'Bias', 'Etalon', '185144') of D2b
        ObsID_a - observation  ID (e.g. 'KP.20230704.02326.27') of D2a
        ObsID_b - observation  ID (e.g. 'KP.20230704.02326.27') of D2b
    """

    def __init__(self, D2a, D2b, logger=None):
        self.logger = logger if logger is not None else DummyLogger()
        self.D2a = D2a 
        self.D2b = D2b
        primary_header_a = HeaderParse(D2a, 'PRIMARY')
        primary_header_b = HeaderParse(D2b, 'PRIMARY')
        self.header_a = primary_header_a.header
        self.header_b = primary_header_b.header
        self.name_a = primary_header_a.get_name()
        self.name_b = primary_header_b.get_name()
        self.ObsID_a = primary_header_a.get_obsid()
        self.ObsID_b = primary_header_b.get_obsid()

    def plot_2D_image_blink(self, chip=None, fig_path=None, zoom_coords=None, oversampling_factor=10):
        """
        Generate an animated GIF that blinks back and forth between the two CCD 
        images in the 2D objects.  The color bar will be scaled base on the 
        dynamic range of the first image.
        
        Args:
            chip (string) - "green" or "red"
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.
            zoom_coords (tuple) - None or tuple of coordinates (x0, y0, x1, y1) 
            	                  for a zoomed-in plot
            oversampling_factor (int) - the dimensions of the final plot will 
                                        be oversampled by this factor compared
                                        to the pixels in zoom_coords

        Returns:
            Animated GIF image in fig_path
        
        Example:
            ObsID1 = 'KP.20240131.01126.61' # flat
            ObsID2 = 'KP.20240131.18680.85' # wideflat - f/4
            D2_file1 = '/data/2D/' + get_datecode(ObsID1) + '/' + ObsID1 + '_2D.fits'
            D2_file2 = '/data/2D/' + get_datecode(ObsID2) + '/' + ObsID2 + '_2D.fits'
            D2_1 = KPF0.from_fits(D2_file1)
            D2_2 = KPF0.from_fits(D2_file2)
            myD2 = Analyze2D_2files(D2_1, D2_2)
            myD2.plot_2D_image_blink(chip='green', fig_path=ObsID1 + '_' + ObsID2 + '_blink.gif')
            myD2.plot_2D_image_blink(chip='green', zoom_coords=(1640, 1640, 2440, 2440), fig_path=ObsID1 + '_' + ObsID2 + '_zoom_blink.gif')
        """
            
        def update(frame):
            if frame % 2 == 0:
                im.set_array(image_a)
            else:
                im.set_array(image_b)
            return [im]
  
        # Set parameters based on the chip selected
        if chip == 'green' or chip == 'red':
            if chip == 'green':
                CHIP = 'GREEN'
                chip_title = 'Green'
            if chip == 'red':
                CHIP = 'RED'
                chip_title = 'Red'
        else:
            self.logger.debug('chip not supplied.  Exiting plot_2D_image_blink')
            return

        image_a = np.array(self.D2a[CHIP + '_CCD'].data)
        image_b = np.array(self.D2b[CHIP + '_CCD'].data)
        
        if zoom_coords != None:
            x0 = zoom_coords[0]
            y0 = zoom_coords[1]
            x1 = zoom_coords[2]
            y1 = zoom_coords[3]
            image_a = image_a[y0:y1, x0:x1] 
            image_b = image_b[y0:y1, x0:x1]
            vmin = np.nanpercentile(image_a,0.1)
            vmax = np.nanpercentile(image_a,99)
            figsize = (int(abs(x1-x0)*oversampling_factor/72), 
                       int(abs(y1-y0)*oversampling_factor/72))
        else:
            vmin = np.nanpercentile(image_a[100:-100,100:-100],0.1)
            vmax = np.nanpercentile(image_a[100:-100,100:-100],99)
            figsize = ( int(4100/72), int(4100/72) )

        # Generate 2D image
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
        im = ax.imshow(image_a, vmin = vmin, 
                                vmax = vmax, 
                                interpolation = 'None', 
                                origin = 'lower', 
                                cmap='viridis', 
                                animated=True)
        
        ax.grid(False)
        
        ani = FuncAnimation(fig, update, frames=range(20), blit=True)
        frames = []
        for i in range(2):
            update(i)
            frame = Image.fromarray((im.make_image(renderer=fig.canvas.get_renderer())[0]).astype(np.uint8))
            frames.append(frame)

        if fig_path != None:
            frames[0].save(
                fig_path,
                save_all=True,
                append_images=frames[1:],
                duration=2000,  # duration between frames in milliseconds
                loop=0,  # loop=0 for infinite loop
                facecolor='w'
            )
            self.logger.info(f'GIF saved to {fig_path}')
        plt.close('all')
