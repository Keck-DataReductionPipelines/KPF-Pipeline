import time
import copy
import traceback
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi
from datetime import datetime, timedelta
from modules.Utils.utils import DummyLogger
from astropy.time import Time
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from modules.Utils.kpf_parse import HeaderParse
from modules.Utils.kpf_parse import get_datecode_from_filename
from modules.Utils.kpf_parse import get_datetime_obsid
from modules.calibration_lookup.src.alg import GetCalibrations
from kpfpipe.models.level1 import KPF1
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline


class AnalyzeL1:
    """
    Description:
        This class contains functions to analyze L1 spectra (storing them
        as attributes) and functions to plot the results.

    Arguments:
        L1 - an L1 object

    Attributes from compute_l1_snr()):
        GREEN_SNR: Two-dimensional array of SNR values for the Green CCD.  
            The first array index specifies
            the spectral order (0-34 = green, 0-31 = red).
            The second array index specifies the orderlet:
            0=CAL, 1=SCI1, 2=SCI2, 3=SCI3, 4=SKY, 5=SCI1+SCI2+SCI3
        RED_SNR: Similar to GREEN_SNR, but for the Red CCD.
        GREEN_SNR_WAV: One-dimensional array of the wavelength of the middle 
            of the spectral orders on the green CCD.
        RED_SNR_WAV: Similar to GREEN_SNR, but for the Red CCD.

    Attributes from measure_orderlet_flux_ratios():
        w_g_sci1 - wavelength array for SCI1 (green)
        w_g_sci2 - wavelength array for SCI2 (green)
        w_g_sci3 - wavelength array for SCI3 (green)
        w_g_sky  - wavelength array for SKY  (green)
        w_g_cal  - wavelength array for CAL  (green)
        w_r_sci1 - wavelength array for SCI1 (red)
        w_r_sci2 - wavelength array for SCI2 (red)
        w_r_sci3 - wavelength array for SCI3 (red)
        w_r_sky  - wavelength array for SKY  (red)
        w_r_cal  - wavelength array for CAL  (red)
        f_g_sci1 - flux array for SCI1 (green)
        f_g_sci2 - flux array for SCI2 (green)
        f_g_sci3 - flux array for SCI3 (green)
        f_g_sky  - flux array for SKY  (green)
        f_g_cal  - flux array for CAL  (green)
        f_r_sci1 - flux array for SCI1 (red)
        f_r_sci2 - flux array for SCI2 (red)
        f_r_sci3 - flux array for SCI3 (red)
        f_r_sky  - flux array for SKY  (red)
        f_r_cal  - flux array for CAL  (red)
        f_g_sci1_int - interpolated SCI1 flux (green) onto SCI2 wavelengths
        f_g_sci3_int - interpolated SCI3 flux (green) onto SCI2 wavelengths
        f_g_sky_int  - interpolated SKY  flux (green) onto SCI2 wavelengths
        f_g_cal_int  - interpolated CAL  flux (green) onto SCI2 wavelengths
        f_r_sci1_int - interpolated SCI1 flux (red) onto SCI2 wavelengths
        f_r_sci3_int - interpolated SCI3 flux (red) onto SCI2 wavelengths
        f_r_sky_int  - interpolated SKY  flux (red) onto SCI2 wavelengths
        f_r_cal_int  - interpolated CAL  flux (red) onto SCI2 wavelengths
        ratio_g_sci1_sci2 - SCI1/SCI2 flux ratio per spectral order (green)
        ratio_g_sci3_sci2 - SCI3/SCI2 flux ratio per spectral order (green)
        ratio_g_sci1_sci3 - SCI1/SCI3 flux ratio per spectral order (green)
        ratio_g_sky_sci2  - SKY/SCI2  flux ratio per spectral order (green)
        ratio_g_cal_sci2  - CAL/SCI2  flux ratio per spectral order (green)
        ratio_r_sci1_sci2 - SCI1/SCI2 flux ratio per spectral order (red)
        ratio_r_sci3_sci2 - SCI3/SCI2 flux ratio per spectral order (red)
        ratio_r_sci1_sci3 - SCI1/SCI3 flux ratio per spectral order (red)
        ratio_r_sky_sci2  - SKY/SCI2  flux ratio per spectral order (red)
        ratio_r_cal_sci2  - CAL/SCI2  flux ratio per spectral order (red)
        w_g_order - central wavelengths for green spectral orders
        w_r_order - central wavelengths for red spectral orders

    Attributes set by compare_wave_to_reference():
        self.wave_diff_green   - (L1 - L1_ref diff) eval at p; [order, orderlet, p]; p = 0th pixel, middle pixel, last pixel
        self.wave_diff_red     - (L1 - L1_ref diff) eval at p; [order, orderlet, p]; p = 0th pixel, middle pixel, last pixel
        self.wave_median_green - median(L1 - L1_ref); [order, orderlet]
        self.wave_median_red   - median(L1 - L1_ref); [order, orderlet]
        self.wave_stddev_green - stddev(L1 - L1_ref); [order, orderlet]
        self.wave_stddev_red   - stddev(L1 - L1_ref); [order, orderlet]
        self.wave_mid_green    - wavelength of middle of order; [order]
        self.wave_mid_red      - wavelength of middle of order; [order]
        self.pix_diff_green   - same as self.wave_diff_green but in pixels
        self.pix_diff_red     - same as self.wave_diff_red   
        self.pix_median_green - same as self.wave_median_green 
        self.pix_median_red   - same as self.wave_median_red   
        self.pix_stddev_green - same as self.wave_stddev_green 
        self.pix_stddev_red   - same as self.wave_stddev_red   
        self.pix_mid_green    - same as self.wave_mid_green
        self.pix_mid_red      - same as self.wave_mid_red   

    Attributes set by measure_l1_snr():
        GREEN_SNR - Two-dimensional array of SNR values for the Green CCD.
            The first array index specifies the spectral order
            (0-34 = green, 0-31 = red).  The second array index
            specifies the orderlet:
            0=CAL, 1=SCI1, 2=SCI2, 3=SCI3, 4=SKY, 5=SCI1+SCI2+SCI3.
            For example, GREEN_SNR[1,2] is the SNR for order=1 and the
            SCI2 orderlet.
        RED_SNR - Similar to GREEN_SNR, but for the Red CCD.
        GREEN_PEAK_FLUX - Similar to GREEN_SNR, but it is an array of top-
            percentile counts instead of SNR.
        RED_PEAK_FLUX - Similar to GREEN_PEAK_FLUX, but for the Red CCD.
        GREEN_SNR_WAV - One-dimensional array of the wavelength of the
            middle of the spectral orders on the green CCD.
        RED_SNR_WAV - Similar to GREEN_SNR, but for the Red CCD.

    """

    def __init__(self, L1, logger=None):
        self.logger = logger if logger is not None else DummyLogger()
        self.L1 = copy.deepcopy(L1)
        primary_header = HeaderParse(L1, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        self.ObsID = primary_header.get_obsid()


    def measure_WLS_age(self, kwd='WLSFILE', verbose=False):
        '''
        Computes the number of days between the observation and the
        date of observations for the WLS files.  The age assumes the 
        following times for autocal calibrations:
            morn     = 18:48 UT  (HST morning cals)
            midnight = 09:30 UT  (HST midnight cals)
            eve      = 03:30 UT  (HST evening cals)

        Arguments:
            kwd - keyword name of WLS file (usually 'WLSFILE' or 'WLSFILE2')
    
        Returns:
            age_wls_file - number of days between the observation and the
                           date of observations for the WLS files
        '''

        date_mjd_str = self.header['MJD-OBS']
        date_obs_datetime = Time(date_mjd_str, format='mjd').datetime
        if verbose:
            self.logger.info(f'Date of observation: {date_obs_datetime.strftime("%Y-%m-%d %H:%M:%S")}')
        
        try:
            wls_filename = self.header[kwd]
            wls_filename_datetime = get_datecode_from_filename(wls_filename, datetime_out=True)
            if "morn" in wls_filename:
                wls_filename_datetime += timedelta(hours=18.8)
            elif "eve" in wls_filename:
                wls_filename_datetime += timedelta(hours=3.5)
            elif "midnight" in wls_filename:
                wls_filename_datetime += timedelta(hours=9.5)
            if verbose:
                self.logger.info(f'Date of {kwd}: {wls_filename_datetime.strftime("%Y-%m-%d %H:%M:%S")}')

            if type(wls_filename_datetime) == type(date_obs_datetime):
                age_wls_file = (wls_filename_datetime - date_obs_datetime).total_seconds() / 86400.0
            else:
                self.logger.info("Error comparing datetimes: ")
                self.logger.info("wls_filename_datetime = " + str(wls_filename_datetime))
                self.logger.info("date_obs_datetime = " + str(date_obs_datetime))
                age_wls_file = None

            if verbose:
                self.logger.info(f'Days between observation and {kwd}: {age_wls_file}')

            return age_wls_file

        except Exception as e:
            self.logger.error(f"Problem with determining age of {kwd}: {e}\n{traceback.format_exc()}")
            return None


    def measure_good_comb_orders(self, chip='green', 
                                       intensity_thresh=40**2, 
                                       min_lines=100, 
                                       divisions_per_order=8):
        """
        This method uses the find_peaks algorithm to measure the number of 
        emission lines above an intensity threshold. Additionally, it checks
        that each order has at least one peak in each of the 
        `divisions_per_order` subregions.
    
        Args:
            chip (str):               CCD name ('green' or 'red')
            intensity_thresh (float): minimum line amplitude to be considered good
            min_lines (int):          minimum number of lines in a spectral 
                                      order for it to be considered good
            divisions_per_order (int): number of contiguous subregions each order 
                                       must have at least one peak in
    
        Returns:
            SCI_fl, CAL_fl, SKY_fl where, e.g., SCI_fl = [first_good_order, last_good_order]
        """
        
        chip = chip.lower()
        data = np.array(self.L1[chip.upper() + '_CAL_WAVE'].data, dtype='d')
        orderlets = ['SCI_FLUX1', 'SCI_FLUX2', 'SCI_FLUX3', 'CAL_FLUX', 'SKY_FLUX']
        
        norder = data.shape[0]
        norderlet = len(orderlets)
        # lines[o, oo] will hold the final "count" for each (order, orderlet)
        lines = np.zeros((norder, norderlet), dtype=int)
    
        def find_first_last_true(arr):
            """
            Find the first and last elements of each column that are True.
            The last element is determined first.
            The first element is then determined by scanning from the last 
            good element downward.
            """
            first_true = np.full(arr.shape[1], None)  
            last_true  = np.full(arr.shape[1], None)
            for col in range(arr.shape[1]):  # Iterate over each column
                true_indices = np.where(arr[:, col])[0]  # Indices of True values
                if true_indices.size > 0:
                    last_true[col]  = true_indices[-1]
                    first_true[col] = last_true[col]
                    i = last_true[col]
                    while i >= 0 and arr[i, col]:
                        first_true[col] = i
                        i -= 1           
            return first_true, last_true
    
        for oo, oo_str in enumerate(orderlets):
            for o in range(norder):
                # Extract flux for this order / orderlet
                flux = np.array(self.L1[chip.upper() + '_' + oo_str].data, dtype='d')[o, :].flatten()
                
                # Find peaks above intensity_thresh
                peaks, properties = scipy.signal.find_peaks(flux, height=intensity_thresh, prominence=intensity_thresh)
    
                # Now we check if each of the divisions_per_order subregions 
                # has at least one peak
                flux_len = len(flux)
                region_size = flux_len // divisions_per_order
                
                # Track how many regions actually have >= 1 peak
                num_regions_with_peaks = 0
                
                for d in range(divisions_per_order):
                    start = d * region_size
                    # Make sure we capture any 'leftover' indices in the final region
                    end = (d+1) * region_size if d < divisions_per_order - 1 else flux_len
                    
                    # Check if at least one peak is within [start, end)
                    if np.any((peaks >= start) & (peaks < end)):
                        num_regions_with_peaks += 1
                
                # If all subregions contained at least one peak,
                # we keep the actual count of peaks; otherwise 0.
                if num_regions_with_peaks == divisions_per_order:
                    lines[o, oo] = len(peaks)
                else:
                    lines[o, oo] = 0
    
        # Determine which orders are 'good' (i.e., above min_lines)
        lines_above_threshold = lines > min_lines
    
        # Find the first and last "good" orders in each of the orderlet columns
        first_indices, last_indices = find_first_last_true(lines_above_threshold)
    
        # SCI Fluxes combine the first three columns
        if None in first_indices[0:3]:
            SCI_f = None
        else:
            SCI_f = max(first_indices[0], first_indices[1], first_indices[2])
        if None in last_indices[0:3]:
            SCI_l = None
        else:
            SCI_l = min(last_indices[0], last_indices[1], last_indices[2])
        SCI_fl = [SCI_f, SCI_l]
        # CAL Flux is the 4th column
        if first_indices[3] == None:
            CAL_f = None
        else:
            CAL_f = first_indices[3]
        if last_indices[3] == None:
            CAL_l = None
        else:
            CAL_l = last_indices[3]
        CAL_fl = [CAL_f, CAL_l]
        # SKY Flux is the 5th column
        if first_indices[4] == None:
            SKY_f = None
        else:
            SKY_f = first_indices[4]
        if last_indices[4] == None:
            SKY_l = None
        else:
            SKY_l = last_indices[4]
        SKY_fl = [SKY_f, SKY_l]
    
        return (SCI_fl, CAL_fl, SKY_fl)


    def measure_L1_snr(self, snr_percentile=95, counts_percentile=95):
        """
        Compute the signal-to-noise ratio (SNR) for each spectral order and
        orderlet in an L1 spectrum from KPF.
        SNR is defined as signal / sqrt(abs(variance)) and can be negative.
        Also, compute the 

        Args:
            snr_percentile: percentile in the SNR distribution for each 
                combination of order and orderlet
            counts_percentile: percentile in the counts distribution for each 
                combination of order and orderlet

        Attributes:
            GREEN_SNR - Two-dimensional array of SNR values for the Green CCD.
                The first array index specifies the spectral order
                (0-34 = green, 0-31 = red).  The second array index
                specifies the orderlet:
                0=CAL, 1=SCI1, 2=SCI2, 3=SCI3, 4=SKY, 5=SCI1+SCI2+SCI3.
                For example, GREEN_SNR[1,2] is the SNR for order=1 and the
                SCI2 orderlet.
            RED_SNR - Similar to GREEN_SNR, but for the Red CCD.
            GREEN_PEAK_FLUX - Similar to GREEN_SNR, but it is an array of top-
                percentile counts instead of SNR.
            RED_PEAK_FLUX - Similar to GREEN_PEAK_FLUX, but for the Red CCD.
            GREEN_SNR_WAV - One-dimensional array of the wavelength of the
                middle of the spectral orders on the green CCD.
            RED_SNR_WAV - Similar to GREEN_SNR, but for the Red CCD.

        Returns:
            None
        """
        L1 = self.L1
        self.snr_percentile = snr_percentile
        self.counts_percentile = counts_percentile

        # Determine the number of orders
        norders_green = (L1['GREEN_SKY_WAVE']).shape[0]
        norders_red   = (L1['RED_SKY_WAVE']).shape[0]
        orderlets = {'CAL','SCI1','SCI2','SCI3','SKY'}
        norderlets = len(orderlets)

        # Define SNR arrays (needed for operations below where VAR = 0)
        GREEN_SCI_SNR1 = 0 * L1['GREEN_SCI_VAR1']
        GREEN_SCI_SNR2 = 0 * L1['GREEN_SCI_VAR2']
        GREEN_SCI_SNR3 = 0 * L1['GREEN_SCI_VAR3']
        GREEN_CAL_SNR  = 0 * L1['GREEN_CAL_VAR']
        GREEN_SKY_SNR  = 0 * L1['GREEN_SKY_VAR']
        GREEN_SCI_SNR  = 0 * L1['GREEN_SCI_VAR1']
        RED_SCI_SNR1   = 0 * L1['RED_SCI_VAR1']
        RED_SCI_SNR2   = 0 * L1['RED_SCI_VAR2']
        RED_SCI_SNR3   = 0 * L1['RED_SCI_VAR3']
        RED_CAL_SNR    = 0 * L1['RED_CAL_VAR']
        RED_SKY_SNR    = 0 * L1['RED_SKY_VAR']
        RED_SCI_SNR    = 0 * L1['RED_SCI_VAR1']

        # Create Arrays
        GREEN_SNR       = np.zeros((norders_green, norderlets+1))
        RED_SNR         = np.zeros((norders_red, norderlets+1))
        GREEN_PEAK_FLUX = np.zeros((norders_green, norderlets+1))
        RED_PEAK_FLUX   = np.zeros((norders_red, norderlets+1))
        GREEN_SNR_WAV   = np.zeros(norders_green)
        RED_SNR_WAV     = np.zeros(norders_red)

        # Compute SNR arrays for each of the orders, orderlets, and CCDs.
        GREEN_SCI_SNR1 = np.divide(L1['GREEN_SCI_FLUX1'],
                                   np.sqrt(abs(L1['GREEN_SCI_VAR1'])),
                                   where=(L1['GREEN_SCI_VAR1']!=0))
        GREEN_SCI_SNR2 = np.divide(L1['GREEN_SCI_FLUX2'],
                                   np.sqrt(abs(L1['GREEN_SCI_VAR2'])),
                                   where=(L1['GREEN_SCI_VAR2']!=0))
        GREEN_SCI_SNR3 = np.divide(L1['GREEN_SCI_FLUX3'],
                                   np.sqrt(abs(L1['GREEN_SCI_VAR3'])),
                                   where=(L1['GREEN_SCI_VAR3']!=0))
        GREEN_SCI_SNR  = np.divide(L1['GREEN_SCI_FLUX1']+L1['GREEN_SCI_FLUX3']+L1['GREEN_SCI_FLUX3'],
                                   np.sqrt(abs(L1['GREEN_SCI_VAR1'])+abs(L1['GREEN_SCI_VAR2'])+abs(L1['GREEN_SCI_VAR3'])),
                                   where=(L1['GREEN_SCI_VAR1']+L1['GREEN_SCI_VAR2']+L1['GREEN_SCI_VAR3']!=0))
        GREEN_CAL_SNR  = np.divide(L1['GREEN_CAL_FLUX'],
                                   np.sqrt(abs(L1['GREEN_CAL_VAR'])),
                                   where=(L1['GREEN_CAL_VAR']!=0))
        GREEN_SKY_SNR  = np.divide(L1['GREEN_SKY_FLUX'],
                                   np.sqrt(abs(L1['GREEN_SKY_VAR'])),
                                   where=(L1['GREEN_SKY_VAR']!=0))
        RED_SCI_SNR1   = np.divide(L1['RED_SCI_FLUX1'],
                                   np.sqrt(abs(L1['RED_SCI_VAR1'])),
                                   where=(L1['RED_SCI_VAR1']!=0))
        RED_SCI_SNR2   = np.divide(L1['RED_SCI_FLUX2'],
                                   np.sqrt(abs(L1['RED_SCI_VAR2'])),
                                   where=(L1['RED_SCI_VAR2']!=0))
        RED_SCI_SNR3   = np.divide(L1['RED_SCI_FLUX3'],
                                   np.sqrt(abs(L1['RED_SCI_VAR3'])),
                                   where=(L1['RED_SCI_VAR3']!=0))
        RED_SCI_SNR    = np.divide(L1['RED_SCI_FLUX1']+L1['RED_SCI_FLUX3']+L1['RED_SCI_FLUX3'],
                                   np.sqrt(abs(L1['RED_SCI_VAR1'])+abs(L1['RED_SCI_VAR2'])+abs(L1['RED_SCI_VAR3'])),
                                   where=(L1['RED_SCI_VAR1']+L1['RED_SCI_VAR2']+L1['RED_SCI_VAR3']!=0))
        RED_CAL_SNR    = np.divide(L1['RED_CAL_FLUX'],
                                   np.sqrt(abs(L1['RED_CAL_VAR'])),
                                   where=(L1['RED_CAL_VAR']!=0))
        RED_SKY_SNR    = np.divide(L1['RED_SKY_FLUX'],
                                   np.sqrt(abs(L1['RED_SKY_VAR'])),
                                   where=(L1['RED_SKY_VAR']!=0))

        # Compute SNR per order and per orderlet
        for o in range(norders_green):
            GREEN_SNR_WAV[o] = L1['GREEN_SCI_WAVE1'][o,2040]
            GREEN_SNR[o,0] = np.nanpercentile(GREEN_CAL_SNR[o], snr_percentile)
            GREEN_SNR[o,1] = np.nanpercentile(GREEN_SCI_SNR1[o], snr_percentile)
            GREEN_SNR[o,2] = np.nanpercentile(GREEN_SCI_SNR2[o], snr_percentile)
            GREEN_SNR[o,3] = np.nanpercentile(GREEN_SCI_SNR3[o], snr_percentile)
            GREEN_SNR[o,4] = np.nanpercentile(GREEN_SKY_SNR[o], snr_percentile)
            GREEN_SNR[o,5] = np.nanpercentile(GREEN_SCI_SNR[o], snr_percentile)
            GREEN_PEAK_FLUX[o,0] = np.nanpercentile(L1['GREEN_CAL_FLUX'][o], counts_percentile)
            GREEN_PEAK_FLUX[o,1] = np.nanpercentile(L1['GREEN_SCI_FLUX1'][o], counts_percentile)
            GREEN_PEAK_FLUX[o,2] = np.nanpercentile(L1['GREEN_SCI_FLUX2'][o], counts_percentile)
            GREEN_PEAK_FLUX[o,3] = np.nanpercentile(L1['GREEN_SCI_FLUX3'][o], counts_percentile)
            GREEN_PEAK_FLUX[o,4] = np.nanpercentile(L1['GREEN_SKY_FLUX'][o], counts_percentile)
            GREEN_PEAK_FLUX[o,5] = np.nanpercentile(L1['GREEN_SCI_FLUX1'][o]+L1['GREEN_SCI_FLUX3'][o]+L1['GREEN_SCI_FLUX3'][o], counts_percentile)
        for o in range(norders_red):
            RED_SNR_WAV[o] = L1['RED_SCI_WAVE1'][o,2040]
            RED_SNR[o,0] = np.nanpercentile(RED_CAL_SNR[o], snr_percentile)
            RED_SNR[o,1] = np.nanpercentile(RED_SCI_SNR1[o], snr_percentile)
            RED_SNR[o,2] = np.nanpercentile(RED_SCI_SNR2[o], snr_percentile)
            RED_SNR[o,3] = np.nanpercentile(RED_SCI_SNR3[o], snr_percentile)
            RED_SNR[o,4] = np.nanpercentile(RED_SKY_SNR[o], snr_percentile)
            RED_SNR[o,5] = np.nanpercentile(RED_SCI_SNR[o], snr_percentile)
            RED_PEAK_FLUX[o,0] = np.nanpercentile(L1['RED_CAL_FLUX'][o], counts_percentile)
            RED_PEAK_FLUX[o,1] = np.nanpercentile(L1['RED_SCI_FLUX1'][o], counts_percentile)
            RED_PEAK_FLUX[o,2] = np.nanpercentile(L1['RED_SCI_FLUX2'][o], counts_percentile)
            RED_PEAK_FLUX[o,3] = np.nanpercentile(L1['RED_SCI_FLUX3'][o], counts_percentile)
            RED_PEAK_FLUX[o,4] = np.nanpercentile(L1['RED_SKY_FLUX'][o], counts_percentile)
            RED_PEAK_FLUX[o,5] = np.nanpercentile(L1['RED_SCI_FLUX1'][o]+L1['RED_SCI_FLUX2'][o]+L1['RED_SCI_FLUX3'][o], counts_percentile)

        # Save SNR and COUNTS arrays to the object
        self.GREEN_SNR       = GREEN_SNR
        self.RED_SNR         = RED_SNR
        self.GREEN_PEAK_FLUX = GREEN_PEAK_FLUX
        self.RED_PEAK_FLUX   = RED_PEAK_FLUX
        self.GREEN_SNR_WAV   = GREEN_SNR_WAV
        self.RED_SNR_WAV     = RED_SNR_WAV


    def plot_L1_snr(self, fig_path=None, show_plot=False):
        """
        Generate a plot of SNR per order as compuated using the compute_l1_snr
        function.

        Args:
            fig_path (string) - set to the path for a SNR vs. wavelength file
                to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it the current environment
            (e.g., in a Jupyter Notebook).
        """

        # Make 3-panel plot. First, create the figure and subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10,10), tight_layout=True)

        # Plot the data on each subplot
        ax1.scatter(self.GREEN_SNR_WAV, self.GREEN_SNR[:,5], marker="8", color='darkgreen', label='SCI1+SCI2+SCI3')
        ax1.scatter(self.GREEN_SNR_WAV, self.GREEN_SNR[:,1], marker=">", color='darkgreen', label='SCI1')
        ax1.scatter(self.GREEN_SNR_WAV, self.GREEN_SNR[:,2], marker="s", color='darkgreen', label='SCI2')
        ax1.scatter(self.GREEN_SNR_WAV, self.GREEN_SNR[:,3], marker="<", color='darkgreen', label='SCI3')
        ax1.scatter(self.RED_SNR_WAV,   self.RED_SNR[:,5],   marker="8", color='r', label='SCI1+SCI2+SCI3')
        ax1.scatter(self.RED_SNR_WAV,   self.RED_SNR[:,1],   marker=">", color='r', label='SCI1')
        ax1.scatter(self.RED_SNR_WAV,   self.RED_SNR[:,2],   marker="s", color='r', label='SCI2')
        ax1.scatter(self.RED_SNR_WAV,   self.RED_SNR[:,3],   marker="<", color='r', label='SCI3')
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=12))
        ax1.grid()
        ax2.scatter(self.GREEN_SNR_WAV[1:], self.GREEN_SNR[1:,4], marker="D", color='darkgreen', label='SKY')
        ax2.scatter(self.RED_SNR_WAV,       self.RED_SNR[:,4],   marker="D", color='r', label='SKY')
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=12))
        ax2.grid()
        ax3.scatter(self.GREEN_SNR_WAV, self.GREEN_SNR[:,0], marker="D", color='darkgreen', label='CAL')
        ax3.scatter(self.RED_SNR_WAV,   self.RED_SNR[:,0],   marker="D", color='r', label='CAL')
        ax3.yaxis.set_major_locator(MaxNLocator(nbins=12))
        ax3.grid()
        ax3.set_xlim(4450,8700)

        # Add legend
        ax1.legend(["SCI1+SCI2+SCI3","SCI1","SCI2","SCI3"], ncol=4)

        # Set titles and labels for each subplot
        ax1.set_title(self.ObsID + ' - ' + self.name + ': ' + r'$\mathrm{SNR}_{'+str(self.snr_percentile)+'}$ = '+str(self.snr_percentile)+'th percentile (Signal / $\sqrt{\mathrm{Variance}}$)', fontsize=16)
        ax3.set_xlabel('Wavelength [Ang]', fontsize=14)
        ax1.set_ylabel(r'$\mathrm{SNR}_{'+str(self.snr_percentile)+'}$ - SCI', fontsize=14)
        ax2.set_ylabel(r'$\mathrm{SNR}_{'+str(self.snr_percentile)+'}$ - SKY', fontsize=14)
        ax3.set_ylabel(r'$\mathrm{SNR}_{'+str(self.snr_percentile)+'}$ - CAL', fontsize=14)
        ax3.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        ax2.yaxis.set_tick_params(labelsize=14)
        ax3.yaxis.set_tick_params(labelsize=14)
        ymin, ymax = ax1.get_ylim()
        if ymin > 0:
            ax1.set_ylim(bottom=0)
        ymin, ymax = ax2.get_ylim()
        if ymin > 0:
            ax2.set_ylim(bottom=0)
        ymin, ymax = ax3.get_ylim()
        if ymin > 0:
            ax3.set_ylim(bottom=0)

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0)
        plt.tight_layout()

        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        ax3.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    xytext=(0, -40), textcoords='offset points')

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=300, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def plot_L1_peak_flux(self, fig_path=None, show_plot=False):
        """
        Generate a plot of peak_counts per order as compuated using the compute_l1_snr
        function.

        Args:
            fig_path (string) - set to the path for a peak counts vs. wavelength file
                to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it the current environment
            (e.g., in a Jupyter Notebook).
        """

        # Make 3-panel plot. First, create the figure and subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10,10), tight_layout=True)

        # Plot the data on each subplot
        ax1.scatter(self.GREEN_SNR_WAV, self.GREEN_PEAK_FLUX[:,5], marker="8", color='darkgreen', label='SCI1+SCI2+SCI3')
        ax1.scatter(self.GREEN_SNR_WAV, self.GREEN_PEAK_FLUX[:,1], marker=">", color='darkgreen', label='SCI1')
        ax1.scatter(self.GREEN_SNR_WAV, self.GREEN_PEAK_FLUX[:,2], marker="s", color='darkgreen', label='SCI2')
        ax1.scatter(self.GREEN_SNR_WAV, self.GREEN_PEAK_FLUX[:,3], marker="<", color='darkgreen', label='SCI3')
        ax1.scatter(self.RED_SNR_WAV,   self.RED_PEAK_FLUX[:,5],   marker="8", color='r', label='SCI1+SCI2+SCI3')
        ax1.scatter(self.RED_SNR_WAV,   self.RED_PEAK_FLUX[:,1],   marker=">", color='r', label='SCI1')
        ax1.scatter(self.RED_SNR_WAV,   self.RED_PEAK_FLUX[:,2],   marker="s", color='r', label='SCI2')
        ax1.scatter(self.RED_SNR_WAV,   self.RED_PEAK_FLUX[:,3],   marker="<", color='r', label='SCI3')
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=12))
        ax1.grid()
        ax2.scatter(self.GREEN_SNR_WAV[1:], self.GREEN_PEAK_FLUX[1:,4], marker="D", color='darkgreen', label='SKY')
        ax2.scatter(self.RED_SNR_WAV,       self.RED_PEAK_FLUX[:,4],   marker="D", color='r', label='SKY')
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=12))
        ax2.grid()
        ax3.scatter(self.GREEN_SNR_WAV, self.GREEN_PEAK_FLUX[:,0], marker="D", color='darkgreen', label='CAL')
        ax3.scatter(self.RED_SNR_WAV,   self.RED_PEAK_FLUX[:,0],   marker="D", color='r', label='CAL')
        ax3.yaxis.set_major_locator(MaxNLocator(nbins=12))
        ax3.grid()
        ax3.set_xlim(4450,8700)

        # Add legend
        ax1.legend(["SCI1+SCI2+SCI3","SCI1","SCI2","SCI3"], ncol=4)

        # Set titles and labels for each subplot
        ax1.set_title(self.ObsID + ' - ' + self.name + ': ' + r'$\mathrm{FLUX}_{'+str(self.snr_percentile)+'}$ = '+str(self.snr_percentile)+'th percentile (Signal)', fontsize=16)
        ax3.set_xlabel('Wavelength [Ang]', fontsize=14)
        ax1.set_ylabel(r'$\mathrm{FLUX}_{'+str(self.snr_percentile)+'}$ - SCI', fontsize=14)
        ax2.set_ylabel(r'$\mathrm{FLUX}_{'+str(self.snr_percentile)+'}$ - SKY', fontsize=14)
        ax3.set_ylabel(r'$\mathrm{FLUX}_{'+str(self.snr_percentile)+'}$ - CAL', fontsize=14)
        ax3.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        ax2.yaxis.set_tick_params(labelsize=14)
        ax3.yaxis.set_tick_params(labelsize=14)
        ymin, ymax = ax1.get_ylim()
        if ymin > 0:
            ax1.set_ylim(bottom=0)
        ymin, ymax = ax2.get_ylim()
        if ymin > 0:
            ax2.set_ylim(bottom=0)
        ymin, ymax = ax3.get_ylim()
        if ymin > 0:
            ax3.set_ylim(bottom=0)

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0)
        plt.tight_layout()

        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    xytext=(0, -50), textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)     

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=300, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def plot_L1_spectrum(self, variance=False, data_over_sqrt_variance=False, 
                         orderlet=None, fig_path=None, show_plot=False):
        """
        Generate a rainbow-colored plot L1 spectrum.  One must select an orderlet.

        Args:
            variance - plot variance (VAR extensions) instead of signal (CCD extensions)
            data_over_sqrt_variance - plot data divided by sqrt(variance), an approximate SNR spectrum
            orderlet (string) - "CAL", "SCI1", "SCI2", "SCI3", "SKY"
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment
            (e.g., in a Jupyter Notebook).
        """
        
        # Parameters
        n_orders_per_panel = 8 # int(self.config['L1']['n_per_row']) #number of orders per panel

        # Define wavelength and flux arrays
        if orderlet.lower() == 'sci1':
            wav_green  = np.array(self.L1['GREEN_SCI_WAVE1'].data,'d')
            wav_red    = np.array(self.L1['RED_SCI_WAVE1'].data,'d')
            if variance:
                flux_green = np.array(self.L1['GREEN_SCI_VAR1'].data,'d')
                flux_red   = np.array(self.L1['RED_SCI_VAR1'].data,'d')
            elif data_over_sqrt_variance:
                flux_green = np.divide(np.array(self.L1['GREEN_SCI_FLUX1'].data,'d'), 
                                       np.sqrt(np.abs(np.array(self.L1['GREEN_SCI_VAR1'].data,'d'))), 
                                       out=np.zeros_like(np.array(self.L1['GREEN_SCI_FLUX1'].data,'d'), dtype=float), 
                                       where=np.sqrt(np.abs(np.array(self.L1['GREEN_SCI_VAR1'].data,'d'))) != 0)
                flux_red   = np.divide(np.array(self.L1['RED_SCI_FLUX1'].data,'d'), 
                                       np.sqrt(np.abs(np.array(self.L1['RED_SCI_VAR1'].data,'d'))), 
                                       out=np.zeros_like(np.array(self.L1['RED_SCI_FLUX1'].data,'d'), dtype=float), 
                                       where=np.sqrt(np.abs(np.array(self.L1['RED_SCI_VAR1'].data,'d'))) != 0)
                #flux_green = np.array(self.L1['GREEN_SCI_FLUX1'].data,'d') / np.sqrt(np.abs(np.array(self.L1['GREEN_SCI_VAR1'].data,'d')))
                #flux_red   = np.array(self.L1['RED_SCI_FLUX1'].data,'d')   / np.sqrt(np.abs(np.array(self.L1['RED_SCI_VAR1'].data,'d')))
            else:
                flux_green = np.array(self.L1['GREEN_SCI_FLUX1'].data,'d')
                flux_red   = np.array(self.L1['RED_SCI_FLUX1'].data,'d')

        elif orderlet.lower() == 'sci2':
            wav_green  = np.array(self.L1['GREEN_SCI_WAVE2'].data,'d')
            wav_red    = np.array(self.L1['RED_SCI_WAVE2'].data,'d')
            if variance:
                flux_green = np.array(self.L1['GREEN_SCI_VAR2'].data,'d')
                flux_red   = np.array(self.L1['RED_SCI_VAR2'].data,'d')
            elif data_over_sqrt_variance:
                flux_green = np.divide(np.array(self.L1['GREEN_SCI_FLUX2'].data,'d'), 
                                       np.sqrt(np.abs(np.array(self.L1['GREEN_SCI_VAR2'].data,'d'))), 
                                       out=np.zeros_like(np.array(self.L1['GREEN_SCI_FLUX2'].data,'d'), dtype=float), 
                                       where=np.sqrt(np.abs(np.array(self.L1['GREEN_SCI_VAR2'].data,'d'))) != 0)
                flux_red   = np.divide(np.array(self.L1['RED_SCI_FLUX2'].data,'d'), 
                                       np.sqrt(np.abs(np.array(self.L1['RED_SCI_VAR2'].data,'d'))), 
                                       out=np.zeros_like(np.array(self.L1['RED_SCI_FLUX2'].data,'d'), dtype=float), 
                                       where=np.sqrt(np.abs(np.array(self.L1['RED_SCI_VAR2'].data,'d'))) != 0)
                #flux_green = np.array(self.L1['GREEN_SCI_FLUX2'].data,'d') / np.sqrt(np.abs(np.array(self.L1['GREEN_SCI_VAR2'].data,'d')))
                #flux_red   = np.array(self.L1['RED_SCI_FLUX2'].data,'d')   / np.sqrt(np.abs(np.array(self.L1['RED_SCI_VAR2'].data,'d')))
            else:
                flux_green = np.array(self.L1['GREEN_SCI_FLUX1'].data,'d')
                flux_red   = np.array(self.L1['RED_SCI_FLUX1'].data,'d')
        elif orderlet.lower() == 'sci3':
            wav_green  = np.array(self.L1['GREEN_SCI_WAVE3'].data,'d')
            wav_red    = np.array(self.L1['RED_SCI_WAVE3'].data,'d')
            if variance:
                flux_green = np.array(self.L1['GREEN_SCI_VAR3'].data,'d')
                flux_red   = np.array(self.L1['RED_SCI_VAR3'].data,'d')
            elif data_over_sqrt_variance:
                flux_green = np.divide(np.array(self.L1['GREEN_SCI_FLUX3'].data,'d'), 
                                       np.sqrt(np.abs(np.array(self.L1['GREEN_SCI_VAR3'].data,'d'))), 
                                       out=np.zeros_like(np.array(self.L1['GREEN_SCI_FLUX3'].data,'d'), dtype=float), 
                                       where=np.sqrt(np.abs(np.array(self.L1['GREEN_SCI_VAR3'].data,'d'))) != 0)
                flux_red   = np.divide(np.array(self.L1['RED_SCI_FLUX3'].data,'d'), 
                                       np.sqrt(np.abs(np.array(self.L1['RED_SCI_VAR3'].data,'d'))), 
                                       out=np.zeros_like(np.array(self.L1['RED_SCI_FLUX3'].data,'d'), dtype=float), 
                                       where=np.sqrt(np.abs(np.array(self.L1['RED_SCI_VAR3'].data,'d'))) != 0)
                #flux_green = np.array(self.L1['GREEN_SCI_FLUX3'].data,'d') / np.sqrt(np.abs(np.array(self.L1['GREEN_SCI_VAR3'].data,'d')))
                #flux_red   = np.array(self.L1['RED_SCI_FLUX3'].data,'d')   / np.sqrt(np.abs(np.array(self.L1['RED_SCI_VAR3'].data,'d')))
            else:
                flux_green = np.array(self.L1['GREEN_SCI_FLUX3'].data,'d')
                flux_red   = np.array(self.L1['RED_SCI_FLUX3'].data,'d')
        elif orderlet.lower() == 'sky':
            wav_green  = np.array(self.L1['GREEN_SKY_WAVE'].data,'d')
            wav_red    = np.array(self.L1['RED_SKY_WAVE'].data,'d')
            if variance:
                flux_green = np.array(self.L1['GREEN_SKY_VAR'].data,'d')
                flux_red   = np.array(self.L1['RED_SKY_VAR'].data,'d')
            elif data_over_sqrt_variance:
                flux_green = np.divide(np.array(self.L1['GREEN_SKY_FLUX'].data,'d'), 
                                       np.sqrt(np.abs(np.array(self.L1['GREEN_SKY_VAR'].data,'d'))), 
                                       out=np.zeros_like(np.array(self.L1['GREEN_SKY_FLUX'].data,'d'), dtype=float), 
                                       where=np.sqrt(np.abs(np.array(self.L1['GREEN_SKY_VAR'].data,'d'))) != 0)
                flux_red   = np.divide(np.array(self.L1['RED_SKY_FLUX'].data,'d'), 
                                       np.sqrt(np.abs(np.array(self.L1['RED_SKY_VAR'].data,'d'))), 
                                       out=np.zeros_like(np.array(self.L1['RED_SKY_FLUX'].data,'d'), dtype=float), 
                                       where=np.sqrt(np.abs(np.array(self.L1['RED_SKY_VAR'].data,'d'))) != 0)
                #flux_green = np.array(self.L1['GREEN_SKY_FLUX'].data,'d') / np.sqrt(np.abs(np.array(self.L1['GREEN_SKY_VAR'].data,'d')))
                #flux_red   = np.array(self.L1['RED_SKY_FLUX'].data,'d')   / np.sqrt(np.abs(np.array(self.L1['RED_SKY_VAR'].data,'d')))
            else:
                flux_green = np.array(self.L1['GREEN_SKY_FLUX'].data,'d')
                flux_red   = np.array(self.L1['RED_SKY_FLUX'].data,'d')
        elif orderlet.lower() == 'cal':
            wav_green  = np.array(self.L1['GREEN_CAL_WAVE'].data,'d')
            wav_red    = np.array(self.L1['RED_CAL_WAVE'].data,'d')
            if variance:
                flux_green = np.array(self.L1['GREEN_CAL_VAR'].data,'d')
                flux_red   = np.array(self.L1['RED_CAL_VAR'].data,'d')
            elif data_over_sqrt_variance:
                flux_green = np.divide(np.array(self.L1['GREEN_CAL_FLUX'].data,'d'), 
                                       np.sqrt(np.abs(np.array(self.L1['GREEN_CAL_VAR'].data,'d'))), 
                                       out=np.zeros_like(np.array(self.L1['GREEN_CAL_FLUX'].data,'d'), dtype=float), 
                                       where=np.sqrt(np.abs(np.array(self.L1['GREEN_CAL_VAR'].data,'d'))) != 0)
                flux_red   = np.divide(np.array(self.L1['RED_CAL_FLUX'].data,'d'), 
                                       np.sqrt(np.abs(np.array(self.L1['RED_CAL_VAR'].data,'d'))), 
                                       out=np.zeros_like(np.array(self.L1['RED_CAL_FLUX'].data,'d'), dtype=float), 
                                       where=np.sqrt(np.abs(np.array(self.L1['RED_CAL_VAR'].data,'d'))) != 0)
                #flux_green = np.array(self.L1['GREEN_CAL_FLUX'].data,'d') / np.sqrt(np.abs(np.array(self.L1['GREEN_CAL_VAR'].data,'d')))
                #flux_red   = np.array(self.L1['RED_CAL_FLUX'].data,'d')   / np.sqrt(np.abs(np.array(self.L1['RED_CAL_VAR'].data,'d')))
            else:
                flux_green = np.array(self.L1['GREEN_CAL_FLUX'].data,'d')
                flux_red   = np.array(self.L1['RED_CAL_FLUX'].data,'d')
        else:
            self.logger.error('plot_1D_spectrum: orderlet not specified properly.')
        if np.shape(flux_green)==(0,):flux_green = wav_green*0. # placeholder when there is no data
        if np.shape(flux_red)==(0,):  flux_red   = wav_red  *0. # placeholder when there is no data
        wav = np.concatenate((wav_green,wav_red), axis = 0)
        flux = np.concatenate((flux_green,flux_red), axis = 0)

        # Set up figure
        cm = plt.cm.get_cmap('rainbow')
        gs = gridspec.GridSpec(n_orders_per_panel, 1 , height_ratios=np.ones(n_orders_per_panel))
        fig, ax = plt.subplots(int(np.shape(wav)[0]/n_orders_per_panel)+1,1, sharey=False, 
                               figsize=(20,16), tight_layout=True)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, hspace=0.0) 

        # Iterate over spectral orders
        for i in range(np.shape(wav)[0]):
            if wav[i,0] == 0: continue
            low, high = np.nanpercentile(flux[i,:],[0.1,99.9])
            flux[i,:][(flux[i,:]>high) | (flux[i,:]<low)] = np.nan
            j = int(i/n_orders_per_panel)
            rgba = cm((i % n_orders_per_panel)/n_orders_per_panel*1.)
            ax[j].plot(wav[i,:], flux[i,:], linewidth = 0.3, color = rgba)
            left  = min((wav[j*n_orders_per_panel:(j+1)*n_orders_per_panel,:]).flatten())
            right = max((wav[j*n_orders_per_panel:(j+1)*n_orders_per_panel,:]).flatten())
            low, high = np.nanpercentile(flux[j*n_orders_per_panel:(j+1)*n_orders_per_panel,:],[0.1,99.9])
            ax[j].set_xlim(left, right)
            ax[j].set_ylim(np.nanmin(flux[j*n_orders_per_panel:(j+1)*n_orders_per_panel,:])-high*0.05, high*1.15)
            ax[j].xaxis.set_tick_params(labelsize=16)
            ax[j].yaxis.set_tick_params(labelsize=16)
            ax[j].axhline(0, color='gray', linestyle='dotted', linewidth = 0.5)
            ax[j].grid(False)
            
        for j in range(int(np.shape(flux)[0]/n_orders_per_panel)):
            left  = min((wav[j*n_orders_per_panel:(j+1)*n_orders_per_panel,:]).flatten())
            right = max((wav[j*n_orders_per_panel:(j+1)*n_orders_per_panel,:]).flatten())
            low, high = np.nanpercentile(flux[j*n_orders_per_panel:(j+1)*n_orders_per_panel,:],[0.1,99.9])
            ax[j].set_xlim(left, right)
            ax[j].set_ylim(np.nanmin(flux[j*n_orders_per_panel:(j+1)*n_orders_per_panel,:])-high*0.05, high*1.15)
            ax[j].xaxis.set_tick_params(labelsize=16)
            ax[j].yaxis.set_tick_params(labelsize=16)
            ax[j].axhline(0, color='gray', linestyle='dotted', linewidth = 0.5)

        # Add axis labels

        if variance:
            title = 'L1 Variance Spectrum of ' + orderlet.upper() + ': ' + str(self.ObsID) + ' - ' + self.name
            ylabel = 'Variance (e-) in ' + orderlet.upper()
        elif data_over_sqrt_variance:
            title = 'L1 SNR Spectrum of ' + orderlet.upper() + ': ' + str(self.ObsID) + ' - ' + self.name
            ylabel = r'SNR (Counts / Variance$^{1/2}$) in ' + orderlet.upper()
        else:
            title = 'L1 Spectrum of ' + orderlet.upper() + ': ' + str(self.ObsID) + ' - ' + self.name
            ylabel = 'Counts (e-) in ' + orderlet.upper()

        low, high = np.nanpercentile(flux,[0.1,99.9])
        ax[int(np.shape(wav)[0]/n_orders_per_panel/2)].set_ylabel(ylabel,fontsize = 28)
        plt.xlabel('Wavelength (Ang)',fontsize = 28)

        # Add overall title to array of plots
        ax = fig.add_subplot(111, frame_on=False)
        ax.grid(False)
        ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        ax.set_title(title, fontsize=28)
        plt.tight_layout()

        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=16, color="darkgray", ha="right", va="bottom",
                    xytext=(0, -50), textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)     

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=288, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')
        

    def plot_1D_spectrum_single_order(self, chip=None, order=11, ylog=False, 
                                            orderlet=['SCI1', 'SCI2', 'SCI3'], 
                                            fig_path=None, show_plot=False):
        """
        Generate a plot of a single order of the L1 spectrum showing all orderlets.

        Args:
            chip (string) - "green" or "red"
            order (int) - spectral order to plot; if SCI, then SCI1+SCI2+SCI3
            fig_path (string) - set to the path for the file to be generated.
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
        else:
            self.logger.debug('chip not supplied.  Exiting plot_1D_spectrum_single_order')
            print('chip not supplied.  Exiting plot_1D_spectrum_single_order')
            return
        orderlet_lowercase = [o.lower() for o in orderlet]
        if len(orderlet) == 1:
            orderlet_label = orderlet[0].upper()
        else: 
            orderlet_uppercase = [o.upper() for o in orderlet]
            orderlet_label = '/'.join(orderlet_uppercase)

        # Define wavelength and flux arrays
        wav_sci1  = np.array(self.L1[CHIP + '_SCI_WAVE1'].data,'d')[order,:].flatten()
        flux_sci1 = np.array(self.L1[CHIP + '_SCI_FLUX1'].data,'d')[order,:].flatten()
        wav_sci2  = np.array(self.L1[CHIP + '_SCI_WAVE2'].data,'d')[order,:].flatten()
        flux_sci2 = np.array(self.L1[CHIP + '_SCI_FLUX2'].data,'d')[order,:].flatten()
        wav_sci3  = np.array(self.L1[CHIP + '_SCI_WAVE3'].data,'d')[order,:].flatten()
        flux_sci3 = np.array(self.L1[CHIP + '_SCI_FLUX3'].data,'d')[order,:].flatten()
        wav_sky   = np.array(self.L1[CHIP + '_SKY_WAVE'].data,'d')[order,:].flatten()
        flux_sky  = np.array(self.L1[CHIP + '_SKY_FLUX'].data,'d')[order,:].flatten()
        wav_cal   = np.array(self.L1[CHIP + '_CAL_WAVE'].data,'d')[order,:].flatten()
        flux_cal  = np.array(self.L1[CHIP + '_CAL_FLUX'].data,'d')[order,:].flatten()
        wav_sci   = wav_sci2
        flux_sci  = (np.array(self.L1[CHIP + '_SCI_FLUX1'].data,'d')[order,:]+
                     np.array(self.L1[CHIP + '_SCI_FLUX2'].data,'d')[order,:]+
                     np.array(self.L1[CHIP + '_SCI_FLUX3'].data,'d')[order,:]).flatten()

        plt.figure(figsize=(12, 4), tight_layout=True)
        if 'sci1' in orderlet_lowercase:
            plt.plot(wav_sci1, flux_sci1, linewidth=0.75, label='SCI1')
        if 'sci2' in orderlet_lowercase:
            plt.plot(wav_sci2, flux_sci2, linewidth=0.75, label='SCI2')
        if 'sci3' in orderlet_lowercase:
            plt.plot(wav_sci3, flux_sci3, linewidth=0.75, label='SCI3')
        if 'sky' in orderlet_lowercase:
            plt.plot(wav_sci3, flux_sky,  linewidth=0.75, label='SKY')
        if 'cal' in orderlet_lowercase:
            plt.plot(wav_sci3, flux_cal,  linewidth=0.75, label='CAL')
        if 'sci' in orderlet_lowercase:
            plt.plot(wav_sci, flux_sci, linewidth=0.75, label='SCI')
        plt.xlim(min(wav_sci1), max(wav_sci1))
        plt.title('L1 (' + orderlet_label + ') - ' + chip_title + ' CCD: ' + str(self.ObsID) + ' - ' + self.name, fontsize=18)
        plt.xlabel('Wavelength (Ang)', fontsize=18)
        plt.tick_params(axis='both', labelsize=14)  # Setting x-axis label size
        plt.ylabel('Counts (e-)', fontsize=18)
        if ylog: plt.yscale('log')
        plt.grid(True)
        plt.legend()

        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    xytext=(0, -30), textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=400, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def my_1d_interp(self, wav, flux, newwav):
        """
        1D interpolation function that uses B-splines unless the input 
        wavelengths are non-monotonic, in which case it uses cubic splines.  
        This function is used in measure_orderlet_flux_ratio().
        """
        
        # B-spline is not compatabile with non-monotonic WLS (which we should eliminate anyway)
        if np.any(wav[1:] <= wav[:-1]):
            monotonic = False 
        else:
            monotonic = True
        
        # Also check for duplicate x values, which B-spline doesn't like
        unique_elements, counts = np.unique(wav, return_counts=True)
        if np.any(counts > 1):
            monotonic = False
        
        if monotonic == True:
            try:
                interpolator = make_interp_spline(wav, flux, k=3)
                newflux = interpolator(newwav)
            except Exception as e:
                self.logger.info(f'Error: {e}')
                self.logger.info('Using cubic-spline interpolation instead of B-splines.')
                interpolator = interp1d(wav, flux, kind='cubic', fill_value='extrapolate')
                newflux = interpolator(newwav)   
        else:
            try:
                interpolator = interp1d(wav, flux, kind='cubic', fill_value='extrapolate')
                newflux = interpolator(newwav)   
            except Exception as e:
                self.logger.info(f'Error: {e}')
                self.logger.info(f'No interpolation applied.')
                newflux = flux  
        return newflux


    def measure_orderlet_flux_ratios(self):
        """
        Extracts the wavelengths and fluxes for each order.
        Computes the flux ratios of SCI2/SCI1, SCI3/SCI1, CAL/SCI1, SKY/SCI1.

        Args:
            None

        Returns:
            None
        """

        # Define wavelength and flux arrays
        self.w_g_sci1 = np.array(self.L1['GREEN_SCI_WAVE1'].data,'d')
        self.w_r_sci1 = np.array(self.L1['RED_SCI_WAVE1'].data,'d')
        self.f_g_sci1 = np.array(self.L1['GREEN_SCI_FLUX1'].data,'d')
        self.f_r_sci1 = np.array(self.L1['RED_SCI_FLUX1'].data,'d')
        self.w_g_sci2 = np.array(self.L1['GREEN_SCI_WAVE2'].data,'d')
        self.w_r_sci2 = np.array(self.L1['RED_SCI_WAVE2'].data,'d')
        self.f_g_sci2 = np.array(self.L1['GREEN_SCI_FLUX2'].data,'d')
        self.f_r_sci2 = np.array(self.L1['RED_SCI_FLUX2'].data,'d')
        self.w_g_sci3 = np.array(self.L1['GREEN_SCI_WAVE3'].data,'d')
        self.w_r_sci3 = np.array(self.L1['RED_SCI_WAVE3'].data,'d')
        self.f_g_sci3 = np.array(self.L1['GREEN_SCI_FLUX3'].data,'d')
        self.f_r_sci3 = np.array(self.L1['RED_SCI_FLUX3'].data,'d')
        self.w_g_sky  = np.array(self.L1['GREEN_SKY_WAVE'].data,'d')
        self.w_r_sky  = np.array(self.L1['RED_SKY_WAVE'].data,'d')
        self.f_g_sky  = np.array(self.L1['GREEN_SKY_FLUX'].data,'d')
        self.f_r_sky  = np.array(self.L1['RED_SKY_FLUX'].data,'d')
        self.w_g_cal  = np.array(self.L1['GREEN_CAL_WAVE'].data,'d')
        self.w_r_cal  = np.array(self.L1['RED_CAL_WAVE'].data,'d')
        self.f_g_cal  = np.array(self.L1['GREEN_CAL_FLUX'].data,'d')
        self.f_r_cal  = np.array(self.L1['RED_CAL_FLUX'].data,'d')
        
        # Interpolate flux arrays onto SCI2 wavelength scale
        self.f_g_sci1_int = self.f_g_sci2*0
        self.f_g_sci3_int = self.f_g_sci2*0
        self.f_g_sky_int  = self.f_g_sci2*0
        self.f_g_cal_int  = self.f_g_sci2*0
        self.f_r_sci1_int = self.f_r_sci2*0
        self.f_r_sci3_int = self.f_r_sci2*0
        self.f_r_sky_int  = self.f_r_sci2*0
        self.f_r_cal_int  = self.f_r_sci2*0
        
        for o in np.arange(35):
            if sum(self.w_g_sky[o,:]) ==0: self.w_g_sky[o,:] = self.w_g_sci2[o,:] # hack to fix bad sky data
            self.f_g_sci1_int[o,:] = self.my_1d_interp(self.w_g_sci1[o,:], self.f_g_sci1[o,:], self.w_g_sci2[o,:])
            self.f_g_sci3_int[o,:] = self.my_1d_interp(self.w_g_sci3[o,:], self.f_g_sci3[o,:], self.w_g_sci2[o,:])
            self.f_g_sky_int[o,:]  = self.my_1d_interp(self.w_g_sky[o,:],  self.f_g_sky[o,:],  self.w_g_sci2[o,:])
            self.f_g_cal_int[o,:]  = self.my_1d_interp(self.w_g_cal[o,:],  self.f_g_cal[o,:],  self.w_g_sci2[o,:])
        for o in np.arange(32):
            if sum(self.w_r_sky[o,:]) ==0: self.w_r_sky[o,:] = self.w_r_sci2[o,:] # hack to fix bad sky data
            self.f_r_sci1_int[o,:] = self.my_1d_interp(self.w_r_sci1[o,:], self.f_r_sci1[o,:], self.w_r_sci2[o,:])
            self.f_r_sci3_int[o,:] = self.my_1d_interp(self.w_r_sci3[o,:], self.f_r_sci3[o,:], self.w_r_sci2[o,:])
            self.f_r_sky_int[o,:]  = self.my_1d_interp(self.w_r_sky[o,:],  self.f_r_sky[o,:],  self.w_r_sci2[o,:])
            self.f_r_cal_int[o,:]  = self.my_1d_interp(self.w_r_cal[o,:],  self.f_r_cal[o,:],  self.w_r_sci2[o,:])
        
        # Define ratios for each order
        self.ratio_g_sci1_sci2 = np.zeros(35) # for each order median(f_g_sci1(intp on sci2 wav) / f_g_sci2)
        self.ratio_g_sci3_sci2 = np.zeros(35) # "
        self.ratio_g_sci1_sci3 = np.zeros(35) 
        self.ratio_g_sky_sci2  = np.zeros(35)
        self.ratio_g_cal_sci2  = np.zeros(35)
        self.ratio_r_sci1_sci2 = np.zeros(32)
        self.ratio_r_sci3_sci2 = np.zeros(32)
        self.ratio_r_sci1_sci3 = np.zeros(32)
        self.ratio_r_sky_sci2  = np.zeros(32)
        self.ratio_r_cal_sci2  = np.zeros(32)
        
        # Define orderlet-to-orderlet ratios over all orders
        self.f_sci1_flat = np.hstack((self.f_g_sci1.flatten(), self.f_r_sci1.flatten()))
        self.f_sci2_flat = np.hstack((self.f_g_sci2.flatten(), self.f_r_sci2.flatten()))
        self.f_sci3_flat = np.hstack((self.f_g_sci3.flatten(), self.f_r_sci3.flatten()))
        self.f_sky_flat  = np.hstack((self.f_g_sky.flatten(),  self.f_r_sky.flatten()))
        self.f_cal_flat  = np.hstack((self.f_g_cal.flatten(),  self.f_r_cal.flatten()))
        self.f_sci2_flat_ind = self.f_sci2_flat != 0
        self.ratio_sci1_sci2 = np.nanmedian(np.divide(self.f_sci1_flat[self.f_sci2_flat_ind], self.f_sci2_flat[self.f_sci2_flat_ind], where=(self.f_sci2_flat[self.f_sci2_flat_ind]!=0)))
        self.ratio_sci3_sci2 = np.nanmedian(np.divide(self.f_sci3_flat[self.f_sci2_flat_ind],self.f_sci2_flat[self.f_sci2_flat_ind], where=(self.f_sci2_flat[self.f_sci2_flat_ind]!=0)))
        self.ratio_sci1_sci3 = np.nanmedian(np.divide(self.f_sci1_flat[self.f_sci2_flat_ind],self.f_sci3_flat[self.f_sci2_flat_ind], where=(self.f_sci3_flat[self.f_sci2_flat_ind]!=0)))
        self.ratio_sky_sci2  = np.nanmedian(np.divide(self.f_sky_flat[self.f_sci2_flat_ind], self.f_sci2_flat[self.f_sci2_flat_ind], where=(self.f_sci2_flat[self.f_sci2_flat_ind]!=0)))
        self.ratio_cal_sci2  = np.nanmedian(np.divide(self.f_cal_flat[self.f_sci2_flat_ind], self.f_sci2_flat[self.f_sci2_flat_ind], where=(self.f_sci2_flat[self.f_sci2_flat_ind]!=0)))
        
        # Compute ratios
        for o in np.arange(35):
            ind = (self.f_g_sci2[o,:] != 0) 
            self.ratio_g_sci1_sci2[o] = np.nanmedian(np.divide(self.f_g_sci1_int[o,ind], self.f_g_sci2[o,ind],     where=(self.f_g_sci2[o,ind]!=0)))
            self.ratio_g_sci3_sci2[o] = np.nanmedian(np.divide(self.f_g_sci3_int[o,ind], self.f_g_sci2[o,ind],     where=(self.f_g_sci2[o,ind]!=0)))
            self.ratio_g_sci1_sci3[o] = np.nanmedian(np.divide(self.f_g_sci1_int[o,ind], self.f_g_sci3_int[o,ind], where=(self.f_g_sci3_int[o,ind]!=0)))
            self.ratio_g_sky_sci2[o]  = np.nanmedian(np.divide(self.f_g_sky_int[o,ind],  self.f_g_sci2[o,ind],     where=(self.f_g_sci2[o,ind]!=0)))
            self.ratio_g_cal_sci2[o]  = np.nanmedian(np.divide(self.f_g_cal_int[o,ind],  self.f_g_sci2[o,ind],     where=(self.f_g_sci2[o,ind]!=0)))
        for o in np.arange(32):
            ind = (self.f_r_sci2[o,:] != 0) 
            self.ratio_r_sci1_sci2[o] = np.nanmedian(np.divide(self.f_r_sci1_int[o,ind], self.f_r_sci2[o,ind],     where=(self.f_g_sci2[o,ind]!=0)))
            self.ratio_r_sci3_sci2[o] = np.nanmedian(np.divide(self.f_r_sci3_int[o,ind], self.f_r_sci2[o,ind],     where=(self.f_g_sci2[o,ind]!=0)))
            self.ratio_r_sci1_sci3[o] = np.nanmedian(np.divide(self.f_r_sci1_int[o,ind], self.f_r_sci3_int[o,ind], where=(self.f_r_sci3_int[o,ind]!=0)))
            self.ratio_r_sky_sci2[o]  = np.nanmedian(np.divide(self.f_r_sky_int[o,ind],  self.f_r_sci2[o,ind],     where=(self.f_g_sci2[o,ind]!=0)))
            self.ratio_r_cal_sci2[o]  = np.nanmedian(np.divide(self.f_r_cal_int[o,ind],  self.f_r_sci2[o,ind],     where=(self.f_g_sci2[o,ind]!=0)))        

        # Define central wavelengths per order
        self.w_g_order = np.zeros(35) 
        self.w_r_order = np.zeros(32) 
        for o in np.arange(35): self.w_g_order[o] = np.nanmedian(self.w_g_sci2[o,:])
        for o in np.arange(32): self.w_r_order[o] = np.nanmedian(self.w_r_sci2[o,:])


    def plot_orderlet_flux_ratios(self, fig_path=None, show_plot=False):
        """
        Generate a plot of a orderlet flux ratio as a function of spectral orders.

        Args:
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment
            (e.g., in a Jupyter Notebook).
        """

        fig, axs = plt.subplots(5, 1, figsize=(18, 15), sharex=True, tight_layout=True)
        axs[0].set_title('L1 Orderlet Flux Ratios: ' + str(self.ObsID) + ' - ' + self.name, fontsize=18)
        
        # SCI1 / SCI2
        axs[0].scatter(self.w_g_order, self.ratio_g_sci1_sci2, s=100, facecolors='green', edgecolors='black', zorder=2)
        axs[0].plot(   self.w_g_order, self.ratio_g_sci1_sci2, 'k-', zorder=1) 
        axs[0].scatter(self.w_r_order, self.ratio_r_sci1_sci2, s=100, marker='D', facecolors='darkred', edgecolors='black', zorder=2)
        axs[0].plot(   self.w_r_order, self.ratio_r_sci1_sci2, 'k-', zorder=1) 
        axs[0].set_ylabel('SCI1 / SCI2', fontsize=18)
        axs[0].set_xlim(min(self.w_g_order)*0.99, max(self.w_r_order)*1.01)
        axs[0].axhline(self.ratio_sci1_sci2, color='gray', linestyle='--', label=r'median(SCI1$_\mathrm{interp}$(WAV2) / SCI2(WAV2) = %.5f)' % self.ratio_sci1_sci2)
        axs[0].legend(fontsize=16, loc='upper right')
        
        # SCI3 / SCI2
        axs[1].scatter(self.w_g_order, self.ratio_g_sci3_sci2, s=100, facecolors='green', edgecolors='black', zorder=2)
        axs[1].plot(   self.w_g_order, self.ratio_g_sci3_sci2, 'k-', zorder=1) 
        axs[1].scatter(self.w_r_order, self.ratio_r_sci3_sci2, s=100, marker='D', facecolors='darkred', edgecolors='black', zorder=2)
        axs[1].plot(   self.w_r_order, self.ratio_r_sci3_sci2, 'k-', zorder=1) 
        axs[1].set_ylabel('SCI3 / SCI2', fontsize=18)
        axs[1].axhline(self.ratio_sci3_sci2, color='gray', linestyle='--', label=r'median(SCI3$_\mathrm{interp}$(WAV2) / SCI2(WAV2) = %.5f)' % self.ratio_sci3_sci2)
        axs[1].legend(fontsize=16, loc='upper right')
        
        # SCI1 / SCI3
        axs[2].scatter(self.w_g_order, self.ratio_g_sci1_sci3, s=100, facecolors='green', edgecolors='black', zorder=2)
        axs[2].plot(   self.w_g_order, self.ratio_g_sci1_sci3, 'k-', zorder=1) 
        axs[2].scatter(self.w_r_order, self.ratio_r_sci1_sci3, s=100, marker='D', facecolors='darkred', edgecolors='black', zorder=2)
        axs[2].plot(   self.w_r_order, self.ratio_r_sci1_sci3, 'k-', zorder=1) 
        axs[2].set_ylabel('SCI1 / SCI3', fontsize=18)
        axs[2].set_xlim(min(self.w_g_order)*0.99, max(self.w_r_order)*1.01)
        axs[2].axhline(self.ratio_sci1_sci3, color='gray', linestyle='--', label=r'median(SCI1$_\mathrm{interp}$(WAV2) / SCI3$_\mathrm{interp}$(WAV2) = %.5f)' % self.ratio_sci1_sci3)
        axs[2].legend(fontsize=16, loc='upper right')

        # SKY / SCI2
        ind_g = (self.ratio_g_sky_sci2 != 0)
        ind_r = (self.ratio_r_sky_sci2 != 0)
        axs[3].scatter(self.w_g_order[ind_g], self.ratio_g_sky_sci2[ind_g], s=100, facecolors='green', edgecolors='black', zorder=2)
        axs[3].plot(   self.w_g_order[ind_g], self.ratio_g_sky_sci2[ind_g], 'k-', zorder=1) 
        axs[3].scatter(self.w_r_order[ind_r], self.ratio_r_sky_sci2[ind_r], s=100, marker='D', facecolors='darkred', edgecolors='black', zorder=2)
        axs[3].plot(   self.w_r_order[ind_r], self.ratio_r_sky_sci2[ind_r], 'k-', zorder=1) 
        axs[3].set_ylabel('SKY / SCI2', fontsize=18)
        axs[3].axhline(self.ratio_sky_sci2, color='gray', linestyle='--', label=r'median(SKY$_\mathrm{interp}$(WAV2) / SCI2(WAV2) = %.5f)' % self.ratio_sky_sci2)
        axs[3].legend(fontsize=16, loc='upper right')
        
        # CAL / SCI2
        axs[4].scatter(self.w_g_order, self.ratio_g_cal_sci2, s=100, facecolors='green', edgecolors='black', zorder=2)
        axs[4].plot(   self.w_g_order, self.ratio_g_cal_sci2, 'k-', zorder=1) 
        axs[4].scatter(self.w_r_order, self.ratio_r_cal_sci2, s=100, marker='D', facecolors='darkred', edgecolors='black', zorder=2)
        axs[4].plot(   self.w_r_order, self.ratio_r_cal_sci2, 'k-', zorder=1) 
        axs[4].set_ylabel('CAL / SCI2', fontsize=18)
        axs[4].axhline(self.ratio_cal_sci2, color='gray', linestyle='--', label=r'median(CAL$_\mathrm{interp}$(WAV2) / SCI2(WAV2) = %.5f)' % self.ratio_cal_sci2)
        axs[4].legend(fontsize=16, loc='upper right')
        axs[4].set_xlabel('Wavelength (Ang)', fontsize=18)

        for ax in axs:
            ax.tick_params(axis='both', which='major', labelsize=14)

        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    xytext=(0, -30), textcoords='offset points')

        plt.subplots_adjust(bottom=0.1)
        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=200, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def plot_orderlet_flux_ratios_grid(self, orders=[10,20,30], ind_range=[1040,3040], chip=None, fig_path=None, show_plot=False):
        """
        Generate a plot of a orderlet flux ratio as a function of spectral orders.

        Args:
            fig_path (string) - set to the path for the file to be generated.
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
                w_sci2     = self.w_g_sci2
                f_sci2     = self.f_g_sci2
                f_sci1_int = self.f_g_sci1_int
                f_sci3_int = self.f_g_sci3_int
                f_sky_int  = self.f_g_sky_int
                f_cal_int  = self.f_g_cal_int
            if chip == 'red':
                CHIP = 'RED'
                chip_title = 'Red'
                w_sci2     = self.w_r_sci2
                f_sci2     = self.f_r_sci2
                f_sci1_int = self.f_r_sci1_int
                f_sci3_int = self.f_r_sci3_int
                f_sky_int  = self.f_r_sky_int
                f_cal_int  = self.f_r_cal_int
        else:
            self.logger.debug('chip not supplied.  Exiting plot_1D_spectrum_single_order')
            print('chip not supplied.  Exiting plot_1D_spectrum_single_order')
            return

        # Create a 5x3 array of subplots with no vertical space between cells
        fig, axs = plt.subplots(5, 3, sharex='col', sharey='row', figsize=(18, 15))
        for i in range(5):
            for j in range(3):
                axs[i, j].tick_params(axis='both', which='major', labelsize=14)
        
        # orders and pixel ranges to plot (consider making this user configurable)
        o1 = orders[0]
        o2 = orders[1]
        o3 = orders[2]
        imin1 = ind_range[0]; imax1 = ind_range[1]
        imin2 = ind_range[0]; imax2 = ind_range[1]
        imin3 = ind_range[0]; imax3 = ind_range[1]
        
        sigmas = [50-34.1, 50, 50+34.1]
        # Row 0
        o=o1; imin = imin1; imax = imax1
        med = np.median(f_sci1_int[o,imin:imax] / f_sci2[o,imin:imax])
        med_unc = uncertainty_median(f_sci1_int[o,imin:imax] / f_sci2[o,imin:imax])
        axs[0,0].plot(w_sci2[o,imin:imax], f_sci1_int[o,imin:imax] / f_sci2[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='teal') 
        axs[0,0].legend(loc='upper right')
        axs[0,0].set_ylabel('SCI1 / SCI2', fontsize=18)
        axs[0,0].set_title('Order = ' + str(o) + ' (' + str(imax-imin) + ' pixels)', fontsize=14)
        axs[0,0].grid()
        o=o2; imin = imin2; imax = imax2
        med = np.median(f_sci1_int[o,imin:imax] / f_sci2[o,imin:imax])
        med_unc = uncertainty_median(f_sci1_int[o,imin:imax] / f_sci2[o,imin:imax])
        axs[0,1].plot(w_sci2[o,imin:imax], f_sci1_int[o,imin:imax] / f_sci2[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='teal') 
        axs[0,1].legend(loc='upper right')
        axs[0,1].set_title('Order = ' + str(o) + ' (' + str(imax-imin) + ' pixels)', fontsize=14)
        axs[0,1].grid()
        o=o3; imin = imin3; imax = imax3
        med = np.median(f_sci1_int[o,imin:imax] / f_sci2[o,imin:imax])
        med_unc = uncertainty_median(f_sci1_int[o,imin:imax] / f_sci2[o,imin:imax])
        axs[0,2].plot(w_sci2[o,imin:imax], f_sci1_int[o,imin:imax] / f_sci2[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='teal') 
        axs[0,2].legend(loc='upper right')
        axs[0,2].set_title('Order = ' + str(o) + ' (' + str(imax-imin) + ' pixels)', fontsize=14)
        axs[0,2].grid()

        # Row 1
        o=o1; imin = imin1; imax = imax1
        med = np.median(f_sci3_int[o,imin:imax] / f_sci2[o,imin:imax])
        med_unc = uncertainty_median(f_sci3_int[o,imin:imax] / f_sci2[o,imin:imax])
        axs[1,0].plot(w_sci2[o,imin:imax], f_sci3_int[o,imin:imax] / f_sci2[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='tomato') 
        axs[1,0].legend(loc='upper right')
        axs[1,0].set_ylabel('SCI3 / SCI2', fontsize=18)
        axs[1,0].grid()
        o=o2; imin = imin2; imax = imax2
        med = np.median(f_sci3_int[o,imin:imax] / f_sci2[o,imin:imax])
        med_unc = uncertainty_median(f_sci3_int[o,imin:imax] / f_sci2[o,imin:imax])
        axs[1,1].plot(w_sci2[o,imin:imax], f_sci3_int[o,imin:imax] / f_sci2[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='tomato') 
        axs[1,1].legend(loc='upper right')
        axs[1,1].grid()
        o=o3; imin = imin3; imax = imax3
        med = np.median(f_sci3_int[o,imin:imax] / f_sci2[o,imin:imax])
        med_unc = uncertainty_median(f_sci3_int[o,imin:imax] / f_sci2[o,imin:imax])
        axs[1,2].plot(w_sci2[o,imin:imax], f_sci3_int[o,imin:imax] / f_sci2[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='tomato') 
        axs[1,2].legend(loc='upper right')
        axs[1,2].grid()

        # Row 2
        o=o1; imin = imin1; imax = imax1
        med = np.median(f_sci1_int[o,imin:imax] / f_sci3_int[o,imin:imax])
        med_unc = uncertainty_median(f_sci1_int[o,imin:imax] / f_sci3_int[o,imin:imax])
        axs[2,0].plot(w_sci2[o,imin:imax], f_sci1_int[o,imin:imax] / f_sci3_int[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='cornflowerblue') 
        axs[2,0].legend(loc='upper right')
        axs[2,0].set_ylabel('SCI1 / SCI3', fontsize=18)
        axs[2,0].grid()
        o=o2; imin = imin2; imax = imax2
        med = np.median(f_sci1_int[o,imin:imax] / f_sci3_int[o,imin:imax])
        med_unc = uncertainty_median(f_sci1_int[o,imin:imax] / f_sci3_int[o,imin:imax])
        axs[2,1].plot(w_sci2[o,imin:imax], f_sci1_int[o,imin:imax] / f_sci3_int[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='cornflowerblue') 
        axs[2,1].legend(loc='upper right')
        axs[2,1].grid()
        o=o3; imin = imin3; imax = imax3
        med = np.median(f_sci1_int[o,imin:imax] / f_sci3_int[o,imin:imax])
        med_unc = uncertainty_median(f_sci1_int[o,imin:imax] / f_sci3_int[o,imin:imax])
        axs[2,2].plot(w_sci2[o,imin:imax], f_sci1_int[o,imin:imax] / f_sci3_int[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='cornflowerblue') 
        axs[2,2].legend(loc='upper right')
        axs[2,2].grid()

        # Row 3
        o=o1; imin = imin1; imax = imax1
        med = np.median(f_sky_int[o,imin:imax] / f_sci2[o,imin:imax])
        med_unc = uncertainty_median(f_sky_int[o,imin:imax] / f_sci2[o,imin:imax])
        axs[3,0].plot(w_sci2[o,imin:imax], f_sky_int[o,imin:imax] / f_sci2[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='orchid') 
        axs[3,0].legend(loc='upper right')
        axs[3,0].set_ylabel('SKY / SCI2', fontsize=18)
        axs[3,0].grid()
        o=o2; imin = imin2; imax = imax2
        med = np.median(f_sky_int[o,imin:imax] / f_sci2[o,imin:imax])
        med_unc = uncertainty_median(f_sky_int[o,imin:imax] / f_sci2[o,imin:imax])
        axs[3,1].plot(w_sci2[o,imin:imax], f_sky_int[o,imin:imax] / f_sci2[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='orchid') 
        axs[3,1].legend(loc='upper right')
        axs[3,1].grid()
        o=o3; imin = imin3; imax = imax3
        axs[3,2].plot(w_sci2[o,imin:imax], f_sky_int[o,imin:imax] / f_sci2[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='orchid') 
        med = np.median(f_sky_int[o,imin:imax] / f_sci2[o,imin:imax])
        med_unc = uncertainty_median(f_sky_int[o,imin:imax] / f_sci2[o,imin:imax])
        axs[3,2].legend(loc='upper right')
        axs[3,2].grid()
        
        # Row 4
        o=o1; imin = imin1; imax = imax1
        med = np.median(f_cal_int[o,imin:imax] / f_sci2[o,imin:imax])
        med_unc = uncertainty_median(f_cal_int[o,imin:imax] / f_sci2[o,imin:imax])
        axs[4,0].plot(w_sci2[o,imin:imax], f_cal_int[o,imin:imax] / f_sci2[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='turquoise') 
        axs[4,0].legend(loc='upper right')
        axs[4,0].set_ylabel('CAL / SCI2', fontsize=18)
        axs[4,0].set_xlabel('Wavelength (Ang)', fontsize=18)
        axs[4,0].grid()
        o=o2; imin = imin2; imax = imax2
        med = np.median(f_cal_int[o,imin:imax] / f_sci2[o,imin:imax])
        med_unc = uncertainty_median(f_cal_int[o,imin:imax] / f_sci2[o,imin:imax])
        axs[4,1].plot(w_sci2[o,imin:imax], f_cal_int[o,imin:imax] / f_sci2[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='turquoise') 
        axs[4,1].legend(loc='upper right')
        axs[4,1].set_xlabel('Wavelength (Ang)', fontsize=18)
        axs[4,1].grid()
        o=o3; imin = imin3; imax = imax3
        med = np.median(f_cal_int[o,imin:imax] / f_sci2[o,imin:imax])
        med_unc = uncertainty_median(f_cal_int[o,imin:imax] / f_sci2[o,imin:imax])
        axs[4,2].plot(w_sci2[o,imin:imax], f_cal_int[o,imin:imax] / f_sci2[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='turquoise') 
        axs[4,2].legend(loc='upper right')
        axs[4,2].set_xlabel('Wavelength (Ang)', fontsize=18)
        axs[4,2].grid()

        plt.subplots_adjust(hspace=0,wspace=0) # Adjust layout to remove vertical space between subplots

        # Add overall title to array of plots
        ax = fig.add_subplot(111, frame_on=False)
        ax.grid(False)
        ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        ax.set_title('L1 Orderlet Flux Ratios - ' + chip_title + ' CCD: ' + str(self.ObsID) + ' - ' + self.name+ '\n', fontsize=24)
        plt.tight_layout()

        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    xytext=(0, -30), textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=400, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def compare_wave_to_reference(self, reference_file='auto'):
        '''

        This method compares the WAVE arrays of the L1 object to a WAVE arrays
        of a reference L1.  The comparisons are: 1) the median difference in 
        wavelength or pixels between L1 and L1_ref per order and per orderlet, 
        2) the stddev of the difference in wavelength and pixel, 3) 
        the difference evaluate and the first, middle, or last pixel.
        The reference can be from a file whose name is given or is automatically 
        set using GetCalibrations.  The method does note return anything, but it 
        sets a set of attributes (below).

        Arguments:
            reference_file - filename of reference wavelength solution
                             for default value of "auto", the reference is 
                             equal to the rough_wls from GetCalibrations
    
        Attributes set:
            self.wave_diff_green   - (L1 - L1_ref diff) eval at p; 
                                     indices: [order, orderlet, p]; 
                                         orderlet indices: 0=SCI1, 1=SCI2, 2=SCI3, 3=SKY, 4=CAL
                                         p = 0th pixel, middle pixel, last pixel
            self.wave_diff_red     - (L1 - L1_ref diff) eval at p; 
                                     indices: [order, orderlet, p]
            self.wave_median_green - median(L1 - L1_ref); indices: [order, orderlet]
            self.wave_median_red   - median(L1 - L1_ref); indices: [order, orderlet]
            self.wave_stddev_green - stddev(L1 - L1_ref); indices: [order, orderlet]
            self.wave_stddev_red   - stddev(L1 - L1_ref); indices: [order, orderlet]
            self.wave_mid_green    - wavelength of middle of order; indices: [order]
            self.wave_mid_red      - wavelength of middle of order; indices: [order]
            self.pix_diff_green   - same as self.wave_diff_green but in pixels
            self.pix_diff_red     - same as self.wave_diff_red but in pixels   
            self.pix_median_green - same as self.wave_median_green but in pixels 
            self.pix_median_red   - same as self.wave_median_red but in pixels   
            self.pix_stddev_green - same as self.wave_stddev_green but in pixels 
            self.pix_stddev_red   - same as self.wave_stddev_red but in pixels   
            self.pix_mid_green    - same as self.wave_mid_green but in pixels
            self.pix_mid_red      - same as self.wave_mid_red but in pixels   
        '''
        
        # Load reference wavelength solution
        if reference_file == 'auto':
            dt = get_datetime_obsid(self.ObsID).strftime('%Y-%m-%dT%H:%M:%S.%f')
            default_config_path = '/code/KPF-Pipeline/modules/calibration_lookup/configs/default.cfg'
            GC = GetCalibrations(dt, default_config_path, use_db=False)
            wls_dict = GC.lookup(subset=['rough_wls'])
            self.reference_file = wls_dict['rough_wls']
        else:
            self.reference_file = reference_file
        L1_ref = KPF1.from_fits(self.reference_file)

        # definitions
        self.chips = ['GREEN', 'RED']
        self.orderlets = ['SCI_WAVE1', 'SCI_WAVE2', 'SCI_WAVE3', 'SKY_WAVE', 'CAL_WAVE']
        self.green_exts = [self.chips[0] + "_" + o for o in self.orderlets]
        self.red_exts   = [self.chips[1] + "_" + o for o in self.orderlets]
        self.exts = self.green_exts + self.red_exts
        self.norders_per_chip = [self.L1[self.chips[0]+'_'+self.orderlets[0]].shape[0], self.L1[self.chips[1]+'_'+self.orderlets[0]].shape[0]] # green, red
        #self.npix = self.L1[self.chips[0]+'_'+self.orderlets[0]].shape[1] #last pixel
        self.norderlets = len(self.orderlets)
        
        # Check that L1 and L1_ref have same shaped arrays
        self.consistent_array_shapes = True
        for ext in self.green_exts + self.red_exts:
            if not (self.L1[ext].shape == L1_ref[ext].shape):
                self.consistent_array_shapes = False
                self.logger.info(f'Different array sizes between L1 and L1_ref for {ext}.')
                return

        # arrays to store results:
        # difference at end/middle/end pixel between L1 and L1_ref - indices: [order, orderlet, pixel]
        self.wave_diff_green = np.zeros((int(self.norders_per_chip[0]), self.norderlets, 3)) # L1 - L1_ref diff between 0th pixel, middle pixel, last pixel
        self.wave_diff_red   = np.zeros((int(self.norders_per_chip[1]), self.norderlets, 3))
        self.pix_diff_green  = self.wave_diff_green 
        self.pix_diff_red    = self.wave_diff_red
        # median difference between L1 and L1_ref - indices: [order, orderlet]
        self.wave_median_green = np.zeros((int(self.norders_per_chip[0]), self.norderlets)) # stddev(L1 - L1_ref) per order and orderlet
        self.wave_median_red   = np.zeros((int(self.norders_per_chip[1]), self.norderlets)) 
        self.pix_median_green  = self.wave_median_green 
        self.pix_median_red    = self.wave_median_red
        # stddev of difference between L1 and L1_ref - indices: [order, orderlet]
        self.wave_stddev_green = np.zeros((int(self.norders_per_chip[0]), self.norderlets)) # stddev(L1 - L1_ref) per order and orderlet
        self.wave_stddev_red   = np.zeros((int(self.norders_per_chip[1]), self.norderlets)) 
        self.pix_stddev_green  = self.wave_stddev_green 
        self.pix_stddev_red    = self.wave_stddev_red
        # indices: [order]
        self.wave_mid_green = np.zeros(int(self.norders_per_chip[0])) # wavelength of middle of the order
        self.wave_mid_red   = np.zeros(int(self.norders_per_chip[1])) 
        
        # Compute endpint wavelength differences and stddev between L1 and L1_ref
        for oo, ext in enumerate(self.green_exts): # oo = orderlet number - 'SCI1, SCI2, SCI3, SKY, CAL
            for o in np.arange(self.norders_per_chip[0]):
                self.wave_diff_green[o,oo,0] = self.L1[ext][o,0] - L1_ref[ext][o,0]  # 0th pixel
                self.wave_diff_green[o,oo,1] = self.L1[ext][o,2040] - L1_ref[ext][o,2040] # middle pixel
                self.wave_diff_green[o,oo,2] = self.L1[ext][o,-1] - L1_ref[ext][o,-1] # last pixel
                self.wave_median_green[o,oo] = np.nanmedian(self.L1[ext][o,:] - L1_ref[ext][o,:])
                self.wave_stddev_green[o,oo] = np.nanstd(self.L1[ext][o,:] - L1_ref[ext][o,:])
                npix_deriv = 1000
                dwavdpix = (self.L1[ext][o,2040+npix_deriv] - self.L1[ext][o,2040-npix_deriv]) / (2*npix_deriv)
                self.pix_diff_green[o,oo,0] = self.wave_diff_green[o,oo,0] / dwavdpix
                self.pix_diff_green[o,oo,1] = self.wave_diff_green[o,oo,1] / dwavdpix
                self.pix_diff_green[o,oo,2] = self.wave_diff_green[o,oo,2] / dwavdpix
                self.pix_median_green[o,oo] = self.wave_median_green[o,oo] / dwavdpix
                self.pix_stddev_green[o,oo] = self.wave_stddev_green[o,oo] / dwavdpix
        for oo, ext in enumerate(self.red_exts):
            for o in np.arange(self.norders_per_chip[1]):
                self.wave_diff_red[o,oo,0] = self.L1[ext][o,0] - L1_ref[ext][o,0]  # 0th pixel
                self.wave_diff_red[o,oo,1] = self.L1[ext][o,2040] - L1_ref[ext][o,2040] # middle pixel
                self.wave_diff_red[o,oo,2] = self.L1[ext][o,-1] - L1_ref[ext][o,-1] # last pixel
                self.wave_median_red[o,oo] = np.nanmedian(self.L1[ext][o,:] - L1_ref[ext][o,:])
                self.wave_stddev_red[o,oo] = np.nanstd(self.L1[ext][o,:] - L1_ref[ext][o,:])
                npix_deriv = 1000
                dwavdpix = (self.L1[ext][o,2040+npix_deriv] - self.L1[ext][o,2040-npix_deriv]) / (2*npix_deriv)
                self.pix_diff_red[o,oo,0] = self.wave_diff_red[o,oo,0] / dwavdpix
                self.pix_diff_red[o,oo,1] = self.wave_diff_red[o,oo,1] / dwavdpix
                self.pix_diff_red[o,oo,2] = self.wave_diff_red[o,oo,2] / dwavdpix
                self.pix_median_red[o,oo] = self.wave_median_red[o,oo] / dwavdpix
                self.pix_stddev_red[o,oo] = self.wave_stddev_red[o,oo] / dwavdpix
        for o in np.arange(self.norders_per_chip[0]):
            self.wave_mid_green[o] = L1_ref['GREEN_SCI_WAVE2'][o,2040]  # middle pixel
        for o in np.arange(self.norders_per_chip[1]):
            self.wave_mid_red[o] = L1_ref['RED_SCI_WAVE2'][o,2040]  # middle pixel


    def plot_L1_wave_comparison(self, reference_file='auto', 
                                      label_n_outliers=0, nsigma_threshold=4,
                                      fig_path=None, show_plot=False):
        """
        Generate a multi-panel plot comparing the L1 wavelength solution to a
        reference WLS.  There are 5 rows (corresponding to SCI1, SCI2, SCI3, SKY, CAL)
        by 5 columns (corresponding to the meidan(L1-L1_ref) per order, 
        standard deviation, value at pixel=0, value at pixel=2040, value at pixel=4079).

        Args:
            label_n_outliers (int, default=0) - number of outliers to annotate per panel
            nsigma_threshold (int, default=4) - threshold (in standard deviations) for labeling
                                                outliers; note that sigma is determined
                                                separately for red and green orders            
            fig_path (string) - set to the path for a SNR vs. wavelength file
                to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it the current environment
            (e.g., in a Jupyter Notebook).
        """

        self.compare_wave_to_reference(reference_file=reference_file)
        
        nrows=5
        ncols=5
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 12), tight_layout=True)
        xtick_positions = [4450, 5100, 5800, 6650, 7600, 8700]
        xtick_labels = ["4450", "5100", "5800", "6650", "7600", "8700"]
 
        for row in range(nrows):
            for col in range(ncols):
                oo = row # orderlet
                if oo == 0:
                    orderlet_label = 'SCI1'
                    orderlet_marker = ">"
                if oo == 1:
                    orderlet_label = 'SCI2'
                    orderlet_marker = "s"
                if oo == 2:
                    orderlet_label = 'SCI3'
                    orderlet_marker = "<"
                if oo == 3:
                    orderlet_label = 'SKY'
                    orderlet_marker = "D"
                if oo == 4:
                    orderlet_label = 'CAL'
                    orderlet_marker = "D"
                ax = axes[row, col]  # Access the specific Axes object
                if col == 0:
                    green_data = self.pix_median_green[:,oo]
                    red_data   = self.pix_median_red[:,oo]
                if col == 1:
                    green_data = self.pix_stddev_green[:,oo]
                    red_data   = self.pix_stddev_red[:,oo]
                if col == 2:
                    green_data = self.pix_diff_green[:,oo,0]
                    red_data   = self.pix_diff_red[:,oo,0] 
                if col == 3:
                    green_data = self.pix_diff_green[:,oo,1]
                    red_data   = self.pix_diff_red[:,oo,1] 
                if col == 4:
                    green_data = self.pix_diff_green[:,oo,2]
                    red_data   = self.pix_diff_red[:,oo,2] 
                ax.scatter(self.wave_mid_green, green_data, marker=orderlet_marker, c='green', label=orderlet_label)
                ax.scatter(self.wave_mid_red,   red_data,   marker=orderlet_marker, c='darkred')

                if label_n_outliers > 0:
                
                    def compute_median_and_sigma(data_valid):
                        """Return (median, sigma) using 16th and 84th percentiles of data_valid."""
                        p16 = np.percentile(data_valid, 16)
                        p50 = np.percentile(data_valid, 50)
                        p84 = np.percentile(data_valid, 84)
                        sigma = (p84 - p16)/2
                        return p50, sigma
                
                    # Compute median/sigma for green, ignoring zeros
                    mask_g = np.isfinite(green_data) & (green_data != 0.0)
                    g_valid = green_data[mask_g]
                    if len(g_valid) == 0:
                        p50_g, sigma_g = 0, 0
                    else:
                        p50_g, sigma_g = compute_median_and_sigma(g_valid)
                    
                    # Compute median/sigma for red, ignoring zeros
                    mask_r = np.isfinite(red_data) & (red_data != 0.0)
                    r_valid = red_data[mask_r]
                    if len(r_valid) == 0:
                        p50_r, sigma_r = 0, 0
                    else:
                        p50_r, sigma_r = compute_median_and_sigma(r_valid)
                
                    # Distance in sigma
                    dist_in_sigma_g = np.full_like(green_data, np.nan, dtype=float)
                    dist_in_sigma_r = np.full_like(red_data,   np.nan, dtype=float)
                    if sigma_g > 0:
                        dist_in_sigma_g[mask_g] = np.abs(green_data[mask_g] - p50_g) / sigma_g
                    if sigma_r > 0:
                        dist_in_sigma_r[mask_r] = np.abs(red_data[mask_r] - p50_r) / sigma_r
                    combined_dist_in_sigma = np.concatenate([dist_in_sigma_g, dist_in_sigma_r])
                    combined_wave = np.concatenate([self.wave_mid_green, self.wave_mid_red])
                    combined_y    = np.concatenate([green_data, red_data])
                    
                    # Outliers >= n-sigma
                    outlier_mask = (combined_dist_in_sigma >= nsigma_threshold)
                    if np.any(outlier_mask):
                        outlier_idx = np.where(outlier_mask)[0]
                        sort_desc_idx = outlier_idx[np.argsort(combined_dist_in_sigma[outlier_idx])[::-1]]
                        top_outliers = sort_desc_idx[:label_n_outliers]
                        
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()
                        x_range = xlim[1] - xlim[0]
                        y_range = ylim[1] - ylim[0]
                        
                        # We consider angles in 12 increments from 0..2pi
                        base_angles = np.linspace(0, 2*pi, 12, endpoint=False)
                        
                        # Decide the "preferred angle" based on which side of the midpoint
                        def preferred_angle(xf, yf):
                            """
                            Return an angle in radians that places the label
                            on the 'opposite' side from the outlier location:
                             - If outlier is top-right => angle ~ 225° (5*pi/4)
                             - top-left  => 315° (7*pi/4)
                             - bottom-right => 135° (3*pi/4)
                             - bottom-left  => 45° (pi/4)
                            """
                            xmid, ymid = 0.5, 0.5  # midpoint in fraction coords
                            # Determine quadrant
                            if xf >= xmid and yf >= ymid:
                                # top-right => place label bottom-left => ~225°
                                return 5*pi/4
                            elif xf < xmid and yf >= ymid:
                                # top-left => place label bottom-right => ~315°
                                return 7*pi/4
                            elif xf >= xmid and yf < ymid:
                                # bottom-right => place label top-left => ~135°
                                return 3*pi/4
                            else:
                                # bottom-left => place label top-right => ~45°
                                return pi/4
                
                        # Radial distance & boundary padding
                        radius_frac = 0.18
                        pad_frac = 0.04
                
                        for idx in top_outliers:
                            x_val = combined_wave[idx]
                            y_val = combined_y[idx]
                            
                            if idx < len(green_data):
                                point_color = 'green'
                                local_idx = idx
                            else:
                                point_color = 'darkred'
                                local_idx = idx - len(green_data)
                
                            label_text = f"o={local_idx}"
                            
                            # Convert outlier location to fraction of axes
                            x_frac0 = (x_val - xlim[0]) / x_range
                            y_frac0 = (y_val - ylim[0]) / y_range
                            
                            # Build candidate angles: put the "preferred angle" first
                            pref = preferred_angle(x_frac0, y_frac0)
                            # Then add the other base angles
                            # We can ensure we don't duplicate if 'pref' is close to one in base_angles
                            angles = np.concatenate(([pref], base_angles))
                            
                            annotation_placed = False
                            
                            # Try angles in order
                            for angle in angles:
                                x_frac = x_frac0 + radius_frac * cos(angle)
                                y_frac = y_frac0 + radius_frac * sin(angle)
                
                                # Clamp to keep box inside plot
                                x_frac = max(pad_frac, min(x_frac, 1.0 - pad_frac))
                                y_frac = max(pad_frac, min(y_frac, 1.0 - pad_frac))
                
                                # Check if that fraction is sufficiently far from the point
                                dist_to_point = np.hypot(x_frac - x_frac0, y_frac - y_frac0)
                                if dist_to_point > 0.02:
                                    # Place label here
                                    ax.annotate(
                                        label_text,
                                        xy=(x_val, y_val),
                                        xycoords='data',
                                        xytext=(x_frac, y_frac),
                                        textcoords=ax.transAxes,
                                        arrowprops=dict(
                                            arrowstyle="-",  # or "->"
                                            color=point_color,
                                            lw=1,
                                            shrinkA=4,
                                            shrinkB=4
                                        ),
                                        bbox=dict(
                                            boxstyle="round,pad=0.3",
                                            fc="white",
                                            ec=point_color,
                                            alpha=0.7
                                        ),
                                        color=point_color
                                    )
                                    annotation_placed = True
                                    break
                            
                            if not annotation_placed:
                                # Fallback if all angles fail
                                ax.annotate(
                                    label_text,
                                    xy=(x_val, y_val),
                                    xycoords='data',
                                    xytext=(x_frac0, y_frac0),
                                    textcoords=ax.transAxes,
                                    arrowprops=dict(arrowstyle="->", color=point_color),
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=point_color),
                                    color=point_color
                                )

                #ax.xaxis.set_tick_params(labelsize=12)
                ax.set_xscale('log')
                ax.set_xlim(4450, 8700)
                ax.get_xaxis().set_major_locator(plt.NullLocator())
                ax.get_xaxis().set_minor_locator(plt.NullLocator())
                ax.set_xticks(xtick_positions)               # Set positions of ticks
                ax.set_xticklabels(xtick_labels, rotation=0) # Set custom labels (optionally rotate)
                #ax.legend()
                ax.grid()
                ymin, ymax = ax.get_ylim()
                if ymin > 0:
                    ax.set_ylim(bottom=0)
                if row == 0:
                    if col == 0:
                        ax.set_title(r'median(L1 - L1$_\mathrm{ref}$)', fontsize=14)                    
                    if col == 1:
                        ax.set_title(r'stddev(L1 - L1$_\mathrm{ref}$)', fontsize=14)                    
                    if col == 2:
                        ax.set_title(r'L1 - L1$_\mathrm{ref}$'+f' (pixel=0)', fontsize=14)
                    if col == 3:
                        ax.set_title(r'L1 - L1$_\mathrm{ref}$'+f' (pixel=2040)', fontsize=14)
                    if col == 4:
                        ax.set_title(r'L1 - L1$_\mathrm{ref}$'+f' (pixel=4079)', fontsize=14)
                if row == 4:
                    ax.set_xlabel('Wavelength [Ang]', fontsize=14)
                if col == 0:
                    ax.set_ylabel(f'$\Delta$ pixels ' + f'({orderlet_label})\nper order', fontsize=14)

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0)
        plt.tight_layout()

        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        axes[4, 4].annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    xytext=(0, -50), textcoords='offset points')

        # Add overall title to array of plots
        title = f'WAVE Comparison: L1 = {self.ObsID}, ' + r'L1$_\mathrm{ref}$' + f' = {self.reference_file}\n'
        ax = fig.add_subplot(111, frame_on=False)
        ax.grid(False)
        ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        ax.set_title(title, fontsize=16)
        plt.tight_layout()

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=300, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


def uncertainty_median(input_data, n_bootstrap=1000):
    """
    Estimate the uncertainty of the median of a dataset.

    Args:
        input_data (array) - 1D array
        n_bootstrap (int) - number of bootstrap resamplings

    Returns:
        uncertainty of median of input_data
    """

    n = len(input_data)
    indices = np.random.randint(0, n, (n_bootstrap, n))
    bootstrapped_medians = np.median(input_data[indices], axis=1)
    median_uncertainty = np.std(bootstrapped_medians)
    return median_uncertainty

