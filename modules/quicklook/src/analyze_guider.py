import time
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, ScalarFormatter
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
from astropy.table import Table
from astropy.time import Time
from modules.Utils.utils import DummyLogger
from modules.Utils.kpf_parse import HeaderParse
from modules.Utils.utils import get_moon_sep, get_sun_alt


class AnalyzeGuider:

    """
    Description:
        This class contains functions to analyze the Guider extensions in 
        L0 or 2D images (storing them as attributes) and functions to plot the results.

    Arguments:
        L0 - an L0 object (or a 2D object -- to be tested)

    Attributes:
        x_rms   - x-coordinate RMS guiding error in milliarcsec (mas)
        y_rms   - y-coordinate RMS guiding error in milliarcsec (mas)
        r_rms   - r-coordinate RMS guiding error in milliarcsec (mas)
        x_bias  - x-coordinate bias guiding error in milliarcsec (mas)
        y_bias  - y-coordinate bias guiding error in milliarcsec (mas)
        t       - time array of guider frames
        x_mas   - x-coordinate array of measured guider errors
        y_mas   - y-coordinate array of measured guider errors
        r_mas   - radial coordinate array of measured guider errors
        nframes - number of frames in guider time series
        nframes_uniq_mas - number of frames in guider time series with detected objects
        fwhm_mas_median - median of FWHM (mas)
        fwhm_mas_std - standard deviation of FWHM (mas)
        flux_median - median integrated flux
        flux_std - standard deviation of integrated flux
        peak_flux_median - median of peak flux (per pixel)
        peak_flux_std - standard deviation of peak flux (per pixel)
        frac_frames_saturated - fraction of frames with 1 or more pixels within 90% of saturation
    """

    def __init__(self, L0, logger=None):
        self.logger = logger if logger is not None else DummyLogger()
        self.L0 = copy.deepcopy(L0)
        self.pixel_scale = 0.056 # arcsec per pixel for the CRED-2 imager on the KPF FIU

        header_primary_obj = HeaderParse(L0, 'PRIMARY')
        if hasattr(L0, 'guider_avg'):
            header_guider_obj = HeaderParse(L0, 'guider_avg')
            self.guider_avg = L0['guider_avg']
        elif hasattr(L0, 'GUIDER_AVG'):
            header_guider_obj = HeaderParse(L0, 'GUIDER_AVG')
            self.guider_avg = L0['GUIDER_AVG']
        else:
            print("Guider image not in file.")
        self.guider_header = header_guider_obj.header
        self.header = header_primary_obj.header
        self.name = header_primary_obj.get_name()
        self.ObsID = header_primary_obj.get_obsid()
        self.date_mid = Time(self.header['DATE-MID'])
        self.ra  = self.header['RA'] # string
        self.dec = self.header['DEC'] # string
        self.el = self.header['EL'] # string
        self.airmass = self.header['AIRMASS'] # string
        self.gmag = self.header['GAIAMAG'] # Gaia G magnitude
        self.jmag = self.header['2MASSMAG'] # 2MASS J magnitude
        try:
            self.gmag = float(self.gmag)
        except:
            print('The keyword GAIAMAG is not a float.')
        try:
            self.jmag = float(self.jmag)
        except:
            print('The keyword 2MASSMAG is not a float.')
        self.gcfps = self.header['GCFPS'] # frames per second for guide camera
        self.gcgain = self.header['GCGAIN'] # detector gain setting 
        if hasattr(L0, 'guider_cube_origins'):
            self.df_GUIDER = self.L0['guider_cube_origins']
        elif hasattr(L0, 'GUIDER_CUBE_ORIGINS'):
            self.df_GUIDER = self.L0['GUIDER_CUBE_ORIGINS']
        else:
            print("Guider table not in file.")
        # only drop bogus rows if the array has non-zero values for flux (i.e., the guide camera was used)
        self.nframes = self.df_GUIDER.shape[0]
        if not (self.df_GUIDER['object1_flux'] == 0.0).all() and self.nframes > 1:
            self.df_GUIDER = self.df_GUIDER[self.df_GUIDER.timestamp != 0.0]    # remove bogus rows
            self.df_GUIDER = self.df_GUIDER[self.df_GUIDER.object1_flux != 0.0] # remove bogus rows
        self.good_fit = None
        
        # Define datasets
        self.t     =  self.df_GUIDER.timestamp-min(self.df_GUIDER.timestamp)
        self.x_mas = (self.df_GUIDER.target_x - self.df_GUIDER.object1_x) * self.pixel_scale*1000
        self.y_mas = (self.df_GUIDER.target_y - self.df_GUIDER.object1_y) * self.pixel_scale*1000
        self.r_mas = (self.x_mas**2+self.y_mas**2)**0.5
        self.nframes = self.df_GUIDER.shape[0]
        self.nframes_uniq_mas = min(np.unique(self.df_GUIDER.object1_x).size, np.unique(self.df_GUIDER.object1_y).size)

        self.df_GUIDER['timestamp'] = pd.to_numeric(self.df_GUIDER['timestamp'], errors='coerce')
        self.df_GUIDER['datetime'] = (pd.to_datetime(self.df_GUIDER['timestamp'], unit='s', origin='unix').dt.tz_localize('UTC'))
        self.df_GUIDER['timestep_ms'] = (self.df_GUIDER['datetime'].diff().dt.total_seconds() * 1000)
        
        self.saturation_threshold = 15830
        self.n_saturated_pixels = np.count_nonzero(self.guider_avg[255-50:255+50, 320-50:320+50] > self.saturation_threshold*0.9)

        
        # Measure FWHM, flux, peak flux, etc.
        if not (self.df_GUIDER['object1_flux'] == 0.0).all() and self.nframes > 2:
            self.fwhm_mas_median = np.median((self.df_GUIDER.object1_a**2 + self.df_GUIDER.object1_b**2)**0.5 / self.pixel_scale * (2*(2*np.log(2))**0.5))
            self.fwhm_mas_std = np.std((self.df_GUIDER.object1_a**2 + self.df_GUIDER.object1_b**2)**0.5 / self.pixel_scale * (2*(2*np.log(2))**0.5))
            self.flux_median = np.median(self.df_GUIDER.object1_flux)
            self.flux_std = np.std(self.df_GUIDER.object1_flux)
            self.peak_flux_median = np.median(self.df_GUIDER.object1_peak)
            self.peak_flux_std = np.std(self.df_GUIDER.object1_peak)
            self.frac_frames_saturated = (self.df_GUIDER['object1_peak'] > self.saturation_threshold*0.9).sum() / self.df_GUIDER.shape[0]
        else:
            self.fwhm_mas_median = -1
            self.flux_median = -1
            self.peak_flux_median = -1
            self.frac_frames_saturated = -1

        # Measure guiding statistics
        self.measure_guider_errors()


    def measure_guider_errors(self):

        """
        Compute the guiding error RMS for X/Y/R and the bias (offset) for X/Y. 

        Args:
            None

        Attributes:
            x_rms  - x-coordinate RMS guiding error in milliarcsec (mas)
            y_rms  - y-coordinate RMS guiding error in milliarcsec (mas)
            r_rms  - r-coordinate RMS guiding error in milliarcsec (mas)
            x_bias - x-coordinate bias guiding error in milliarcsec (mas)
            y_bias - y-coordinate bias guiding error in milliarcsec (mas)

        Returns:
            None
        """
        if self.nframes_uniq_mas > 10:
            try:
                self.x_rms = (np.nanmean(self.x_mas**2))**0.5
                self.y_rms = (np.nanmean(self.y_mas**2))**0.5
                self.r_rms = (np.nanmean(self.r_mas**2))**0.5
                self.x_bias = np.nanmean(self.x_mas)
                self.y_bias = np.nanmean(self.y_mas)
            except:
                self.logger.info('Error computing guiding errors')
                self.x_rms = None
                self.y_rms = None
                self.r_rms = None
                self.x_bias = None
                self.y_bias = None
        else:
            self.logger.info('Error computing guiding errors.  Number of unique guiding errors = ' + str(self.nframes_uniq_mas) + '.')
            self.x_rms = None
            self.y_rms = None
            self.r_rms = None
            self.x_bias = None
            self.y_bias = None


    def measure_seeing(self, debug=False):

        """
        Compute the seeing from a stacked guider image.

        Args:
            None

        Attributes:
            guider_image - stacked guider image
            image_fit - image of the Moffat function fit of the guider image; 
                        this has the same dimensions as guider_image
            amplitude - amplitude of the fitted Moffat function
            seeing_550nm - seeing value scaled with wavelength^(1/5)
            seeing - seeing value at the guide camera wavelengths (950-1200 nm);
                     equal to the alpha parameter of the fitted Moffat function
            beta - beta parameter of the fitted Moffat function
            x0 - x-coordinate of the centroid of the fitted Moffat function 
            y0 - x-coordinate of the centroid of the fitted Moffat function 

        Returns:
            None
        """
        
        self.guider_image = self.L0['GUIDER_AVG'].data - np.median(self.L0['GUIDER_AVG'].data)
        
        # Experiment with smoothed image to remove hot pixels
        #self.smoothed_image = median_filter(self.guider_image, size=2)
        #self.guider_image = self.smoothed_image

        def moffat_2D(xy, amplitude, x0, y0, alpha, beta):
            x, y = xy
            return amplitude * (1 + ((x - x0) ** 2 + (y - y0) ** 2) / alpha ** 2) ** -beta
        
        x = np.arange(self.guider_image.shape[1])
        y = np.arange(self.guider_image.shape[0])
        X, Y = np.meshgrid(x, y)
        x_flat = X.flatten()
        y_flat = Y.flatten()
        image_data_flat = self.guider_image.flatten()
        
        # Define the range of initial guesses for the fourth parameter
        alpha_initial_guesses = [0.4/0.056, 1.0/0.056, 2.5/0.056]
        
        best_fit = None
        smallest_residuals = np.inf
        
        for alpha_guess in alpha_initial_guesses:
            try:
                if debug:
                    self.logger.info(f"Trying to fit Guider image with alpha guess = {alpha_guess*0.056} arcsec.")
                # Update the initial guess with the current alpha value
                p0 = [1, 343.1, 264.7, alpha_guess, 2.5]
                if 'GCCRPIX1' in self.header:
                    p0 = [1, self.header['GCCRPIX1'], self.header['GCCRPIX2'], alpha_guess, 2.5]
                elif 'CRPIX1' in self.guider_header:
                    p0 = [1, self.guider_header['CRPIX1'], self.guider_header['CRPIX2'], alpha_guess, 2.5]
        
                # Attempt to fit the model
                popt, pcov = curve_fit(moffat_2D, (x_flat, y_flat), image_data_flat, p0=p0) #, maxfev=10000)
        
                # Calculate the residuals
                residuals = image_data_flat - moffat_2D((x_flat, y_flat), *popt)
                sum_of_squares = np.sum(residuals**2)
        
                # Check if this fit is better than previous ones
                if sum_of_squares < smallest_residuals:
                    smallest_residuals = sum_of_squares
                    best_fit = popt
        
            except Exception as e:
                print(f"Fit failed for alpha guess = {alpha_guess*0.056} arcsec. Error: {e}")
        
        if best_fit is not None:
            amplitude_fit, x0_fit, y0_fit, alpha_fit, beta_fit = best_fit
            alpha_fit = abs(alpha_fit)
            self.image_fit = moffat_2D((X, Y), amplitude_fit, x0_fit, y0_fit, alpha_fit, beta_fit)
            self.amplitude = amplitude_fit
            self.seeing = alpha_fit
            self.seeing_550nm = self.seeing * (((1200+950)/2)/550)**0.2
            self.beta = beta_fit
            self.x0 = x0_fit
            self.y0 = y0_fit
            self.good_fit = True
            if debug:
                self.logger.info(f"Best fit found with alpha = {alpha_fit*0.056} arcsec.")
                self.logger.ingo("Fitted positions: " + str(self.x0*self.pixel_scale) + ", " + str(self.y0*self.pixel_scale))
        else:
            self.good_fit = False
            print("No successful fit found.")


    def plot_guider_image(self, fig_path=None, show_plot=False):

        """
        Generate a three-panel plot of the guider image (full view), the zoomed-in 
        guider image, and residuals of the guider image with a fitted Moffat function. 
        Note that the function AnalyzeGuider.measure_seeing must be run first.

        Args:
            fig_path (string) - set to the path for a SNR vs. wavelength file 
                to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it the current environment 
            (e.g., in a Jupyter Notebook).
        """

        # Plot the original image and residuals
        guider_im_zoom = self.guider_image[255-50:255+50, 320-50:320+50]
        if self.good_fit:
            resid_im = self.guider_image - self.image_fit
            resid_im_zoom = resid_im[255-50:255+50, 320-50:320+50]
        else:
            resid_im_zoom = 0*self.guider_image[255-50:255+50, 320-50:320+50]

        fig, axs = plt.subplots(1, 3, figsize=(16, 5), tight_layout=True)

        # Left panel - full image
        im1 = axs[0].imshow(self.guider_image, cmap='viridis', origin='lower', vmin=0, vmax=np.nanpercentile(guider_im_zoom,99.9))
        axs[1].set_aspect(640/512)
        image_size_pixels = self.guider_image.shape
        image_size_arcsec = (image_size_pixels[1] * self.pixel_scale, image_size_pixels[0] * self.pixel_scale)
        x_tick_locator = ticker.MultipleLocator(5/(self.pixel_scale-0.001))  # Set tick every 5 arcsec
        axs[0].xaxis.set_major_locator(x_tick_locator)
        y_tick_locator = ticker.MultipleLocator(5/(self.pixel_scale-0.001))  # Set tick every 5 arcsec
        axs[0].yaxis.set_major_locator(y_tick_locator)
        xticks = axs[0].get_xticks()
        yticks = axs[0].get_yticks()
        #axs[0].set_xticks(xticks)  # Set the x-ticks explicitly
        #axs[0].set_yticks(yticks)  # Set the y-ticks explicitly
        # The following line (and others) produces this warning: UserWarning: FixedFormatter should only be used together with FixedLocator
        axs[0].set_xticklabels([f'{int(x * self.pixel_scale)}' for x in xticks])
        axs[0].set_yticklabels([f'{int(y * self.pixel_scale)}' for y in yticks])
        axs[0].set_xlabel('Arcseconds', fontsize=12)
        axs[0].set_ylabel('Arcseconds', fontsize=12)
        title = str(self.ObsID)+' - ' + self.name 
        if isinstance(self.jmag, float):
            title = title + "\nJ = " + f"{self.jmag:.2f}"
        else:
            title = title + "\nJ = " + self.jmag
        if isinstance(self.gmag, float):
            title = title + ", G = " + f"{self.gmag:.2f}"
        else:
            title = title + ", G = " + self.gmag
        title = title + ', ' + str(int(self.gcfps)) + ' fps, ' + str(self.gcgain) + ' gain'
        axs[0].set_title(title, fontsize=12)
        axs[0].grid(True, linestyle='solid', linewidth=0.5, alpha=0.5)
        #cmap = plt.get_cmap('viridis')
        #axs[0].set_facecolor(cmap(0))
        #cbar1 = plt.colorbar(im1, ax=axs[0], shrink=0.5)

        # Middle panel - zoomed image
        im2 = axs[1].imshow(guider_im_zoom, cmap='viridis', origin='lower', vmin=0, vmax=np.nanpercentile(guider_im_zoom,99.9))
        axs[1].contour(guider_im_zoom, 
                       levels=[0.33*np.nanpercentile(guider_im_zoom,99.9),
                               0.66*np.nanpercentile(guider_im_zoom,99.9)], 
                       colors='gray', 
                       linewidths = [0.5, 0.5],
                       extent=[0, guider_im_zoom.shape[0], 0, guider_im_zoom.shape[1]])
        axs[1].set_aspect('equal')
        image_size_pixels = guider_im_zoom.shape
        image_size_arcsec = (image_size_pixels[1] * self.pixel_scale, image_size_pixels[0] * self.pixel_scale)
        x_tick_locator = ticker.MultipleLocator(0.5/(self.pixel_scale-0.001))  # Set tick every 0.5 arcsec
        axs[1].xaxis.set_major_locator(x_tick_locator)
        y_tick_locator = ticker.MultipleLocator(0.5/(self.pixel_scale-0.001))  # Set tick every 0.5 arcsec
        axs[1].yaxis.set_major_locator(y_tick_locator)
        xticks = axs[1].get_xticks()
        yticks = axs[1].get_yticks()
        #axs[1].set_xticks(xticks)  # Set the x-ticks explicitly
        #axs[1].set_yticks(yticks)  # Set the y-ticks explicitly
        axs[1].set_xticklabels([f'{int(x * self.pixel_scale*10)/10}' for x in xticks])
        axs[1].set_yticklabels([f'{int(y * self.pixel_scale*10)/10}' for y in yticks])
        axs[1].set_xlabel('Arcseconds', fontsize=12)
        axs[1].set_ylabel('Arcseconds', fontsize=12)
        title = 'Guider Image (zoomed in)'
        if self.good_fit:
            title = title + "\n seeing: " + f"{(self.seeing*self.pixel_scale):.2f}" + '" (z+J)'+ r' $\rightarrow$ ' +f"{(self.seeing_550nm*self.pixel_scale):.2f}" + '" (V, scaled)'
        axs[1].set_title(title, fontsize=12)
        axs[1].grid(True, linestyle='solid', linewidth=0.5, alpha=0.5)
        cbar2 = plt.colorbar(im2, ax=axs[1], shrink=0.7)
        cbar2.set_label('Intensity', fontsize=10)

        # Right panel - zoomed image of residuals to model
        scaled_resid_im_zoom = resid_im_zoom/np.nanpercentile(guider_im_zoom,99.9)
        #minmax = max([-np.nanpercentile(scaled_resid_im_zoom, 0.1), np.nanpercentile(scaled_resid_im_zoom,99.9)])
        minmax = 0.1 # use fixed range
        im2 = axs[2].imshow(scaled_resid_im_zoom, 
                            cmap='twilight',#'viridis', 
                            origin='lower', 
                            vmin=-minmax,
                            vmax= minmax,
                            extent=[0, scaled_resid_im_zoom.shape[0], 0, scaled_resid_im_zoom.shape[1]])
        axs[2].contour(guider_im_zoom, 
                       levels=[0.33*np.nanpercentile(guider_im_zoom,99.9),
                               0.66*np.nanpercentile(guider_im_zoom,99.9)], 
                       colors='gray', 
                       linewidths = [1, 1],
                       extent=[0, scaled_resid_im_zoom.shape[0], 0, scaled_resid_im_zoom.shape[1]])
        image_size_pixels = guider_im_zoom.shape
        image_size_arcsec = (image_size_pixels[1] * self.pixel_scale, image_size_pixels[0] * self.pixel_scale)
        x_tick_locator = ticker.MultipleLocator(0.5/(self.pixel_scale-0.001))  # Set tick every 0.5 arcsec
        axs[2].xaxis.set_major_locator(x_tick_locator)
        y_tick_locator = ticker.MultipleLocator(0.5/(self.pixel_scale-0.001))  # Set tick every 0.5 arcsec
        axs[2].yaxis.set_major_locator(y_tick_locator)
        xticks = axs[2].get_xticks()
        yticks = axs[2].get_yticks()
        #axs[2].set_xticks(xticks)  # Set the x-ticks explicitly
        #axs[2].set_yticks(yticks)  # Set the y-ticks explicitly
        axs[2].set_xticklabels([f'{x * self.pixel_scale:.1f}' for x in xticks])
        axs[2].set_yticklabels([f'{y * self.pixel_scale:.1f}' for y in yticks])
        axs[2].set_xlabel('Arcseconds', fontsize=12)
        axs[2].set_ylabel('Arcseconds', fontsize=12)
        if self.good_fit:
            title = 'Residuals to Moffat Function Model'
            axs[2].set_title(title, fontsize=10)
        else:
            title = 'Unsuccessful fit with Moffat Function'
            axs[2].set_title(title, fontsize=12, color='r')  # red to indicate unusual image
        axs[2].set_title(title, fontsize=12)
        axs[2].grid(True, linestyle='solid', linewidth=0.5, alpha=0.5)
        cbar3 = plt.colorbar(im2, ax=axs[2], shrink=0.7)
        cbar3.set_label('Fractional Residuals (scaled to peak of guider image)', fontsize=10)
            
        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time} UT"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    xytext=(100, -40), textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)     

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=288, facecolor='w')
            if self.logger:
                self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def plot_guider_error_time_series_simple(self, fig_path=None, show_plot=False):

        """
        Generate a two-panel plot of the guider time series errors as 1) a time series 
        and 2) as a 2-D histogram.

        Args:
            fig_path (string) - set to the path for a SNR vs. wavelength file 
                to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it the current environment 
            (e.g., in a Jupyter Notebook)
        """
        
        if np.sqrt(self.df_GUIDER.shape[0]) < 60:
            hist_bins = 25
        else:
            hist_bins = 40
        
        # Create the figure and subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 4), gridspec_kw={'width_ratios': [2, 1]}, tight_layout=True)
        sns.set_theme(style='whitegrid')

        # Plot the data
        if self.nframes_uniq_mas > 10:
            im1 = axes[1].hist2d(self.x_mas, self.y_mas, bins=hist_bins, cmap='viridis')
        cmap = plt.get_cmap('viridis')
        axes[1].set_facecolor(cmap(0))
        xylim = round(np.nanpercentile(self.r_mas, 99)*1.4 / 5.0) * 5
        axes[1].set_xlim(-xylim, xylim) # set symmetric limits for x and y
        axes[1].set_ylim(-xylim, xylim)
        axes[1].set_aspect('equal')
        axes[1].set_title('r: ' + f'{int(self.r_rms*10)/10}' + ' mas (RMS)', fontsize=14)
        axes[1].set_xlabel('Guiding Error - x (mas)', fontsize=14, alpha=0.5)
        axes[1].set_ylabel('Guiding Error - y (mas)', fontsize=14, alpha=0.5)
        axes[1].grid(True, linestyle='solid', linewidth=0.5, alpha=0.5)
        cbar = plt.colorbar(im1[3])
        cbar.set_label('Samples', fontsize=12)
        
        if self.nframes_uniq_mas > 1:
            axes[0].plot(self.t, self.x_mas, color='royalblue', alpha=0.5)
            axes[0].plot(self.t, self.y_mas, color='orange', alpha=0.5)
        axes[0].set_title("Guiding Error Time Series: " + str(self.ObsID)+' - ' + self.name, fontsize=14)
        axes[0].set_xlabel("Time (sec)", fontsize=14)
        axes[0].set_ylabel("Guiding Error (mas)", fontsize=14)
        axes[0].legend(['Guiding error - x: ' + f'{int(self.x_rms*10)/10}' + ' mas (RMS),' + f'{int(self.x_bias*100)/100}' + ' mas (bias)', 
                        'Guiding error - y: ' + f'{int(self.y_rms*10)/10}' + ' mas (RMS),' + f'{int(self.y_bias*100)/100}' + ' mas (bias)'], 
                       fontsize=12, 
                       loc='best') 

        # Set the font size of tick mark labels
        axes[0].tick_params(axis='both', which='major', labelsize=14)
        axes[1].tick_params(axis='both', which='major', labelsize=14)

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=144, facecolor='w')
            if self.logger:
                self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def plot_guider_error_time_series(self, fig_path=None, show_plot=False):

        """
        Generate a two-panel plot of the guider time series errors as 1) a time series 
        and 2) as a 2-D histogram.

        Args:
            fig_path (string) - set to the path for a SNR vs. wavelength file 
                to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it the current environment 
            (e.g., in a Jupyter Notebook).
        """
        
        # Create the figure and subplots
        fig, axes = plt.subplots(4, 2, figsize=(16, 15), gridspec_kw={'width_ratios': [2, 1]}, tight_layout=True)
        sns.set_theme(style='whitegrid')

        # Count number of stars
        #nstars = []
        #for index, row in self.df_GUIDER.iterrows():
        #    star_count = 0
        #    if row['object1_x'] < -998: star_count += 1
        #    if row['object2_x'] < -998: star_count += 1
        #    if row['object3_x'] < -998: star_count += 1
        #    nstars.append(star_count)
        #nstars = np.array(nstars, dtype=int)
        #nframes_0stars = len(np.where(nstars == 0)[0])
        #nframes_1stars = len(np.where(nstars == 1)[0])
        #nframes_2stars = len(np.where(nstars == 2)[0])
        #nframes_3stars = len(np.where(nstars == 3)[0])
        #try:
        #    median_nstars = int(np.median(nstars))
        #except:
        #    median_nstars = 0
        #w_extra_detections = np.where(nstars > median_nstars)[0]
        #nframes_extra_detections = len(w_extra_detections)
        #w_fewer_detections = np.where(nstars < median_nstars)[0]
        #nframes_fewer_detections = len(w_fewer_detections)
        
        # Set the number of histogram bins
        if np.sqrt(self.nframes) < 60:
            hist_bins = 25
        else:
            hist_bins = 40
        if max(max(self.x_mas), abs(min(self.x_mas)), max(self.y_mas), abs(min(self.y_mas))) / hist_bins > 3:  # if a few really big errors dominate
            hist_bins = 2 * int(max(max(self.x_mas), abs(min(self.x_mas)), max(self.y_mas), abs(min(self.y_mas))) / 3)

        # Histogram of guider errors
        if self.nframes_uniq_mas > 1:
            cbar_ax = fig.add_axes([0.95, 0.775, 0.01, 0.20])  # Adjust these values to properly position your colorbar
            hist = axes[0,1].hist2d(self.x_mas, self.y_mas, bins=hist_bins, cmap='viridis')
            fig.colorbar(hist[3], cax=cbar_ax, label='Samples', shrink=0.6)#, fontsize=12)
            cmap = plt.get_cmap('viridis')
            axes[0,1].set_facecolor(cmap(0))
            xylim = round(np.nanpercentile(self.r_mas, 95)*1.4 / 5.0) * 5
            axes[0,1].set_xlim(-xylim, xylim) # set symmetric limits for x and y
            axes[0,1].set_ylim(-xylim, xylim)
            axes[0,1].set_title('r: ' + f'{int(self.r_rms*10)/10}' + ' mas (RMS)', fontsize=14)
        axes[0,1].set_aspect('equal')
        axes[0,1].set_xlabel('Guiding Error - x (mas)', fontsize=14)
        axes[0,1].set_ylabel('Guiding Error - y (mas)', fontsize=14)
        axes[0,1].grid(True, linestyle='solid', linewidth=0.5, alpha=0.5)

        # Time series plot of guider errors
        if self.nframes_uniq_mas > 1:
            axes[0,0].plot(self.t, self.x_mas, color='royalblue', alpha=0.5)
            axes[0,0].plot(self.t, self.y_mas, color='orange',    alpha=0.5)
        axes[0,0].set_title("Guiding Error Time Series: " + str(self.ObsID)+' - ' + self.name, fontsize=14)
        axes[0,0].set_xlabel("Time (sec)", fontsize=14)
        axes[0,0].set_ylabel("Guiding Error (mas)", fontsize=14)
        axes[0,0].set_xlim(0, max(self.t)) 
        axes[0,0].legend(['Guiding error - x: ' + f'{int(self.x_rms*10)/10}' + ' mas (RMS), ' + f'{int(self.x_bias*10)/10}' + ' mas (bias)', 
                          'Guiding error - y: ' + f'{int(self.y_rms*10)/10}' + ' mas (RMS), ' + f'{int(self.y_bias*10)/10}' + ' mas (bias)'], 
                         fontsize=12, 
                         loc='best') 

        # Power spectral density plot
        fps = self.gcfps # my_Guider.guider_header['FPS']  # Sample rate in Hz
        if self.nframes > 10 and self.nframes_uniq_mas > 100:
            Pxx, freqs = mlab.psd(self.x_mas/1000, Fs=fps)
            Pyy, freqs = mlab.psd(self.y_mas/1000, Fs=fps)
            axes[1,0].step(freqs, Pxx*1e6, where='mid', color='royalblue', label='X - Guiding errors', lw=2, alpha=0.5)
            axes[1,0].step(freqs, Pyy*1e6, where='mid', color='orange',    label='Y - Guiding errors', lw=2, alpha=0.5)
            axes[1,0].grid(True, linestyle='dashed', linewidth=1, alpha=0.5)
            axes[1,0].set_xlim(min(freqs),max(freqs))
            axes[1,0].legend(fontsize=12)
        else:
            pass
        axes[1,0].set_xlabel('Frequency [Hz]', fontsize=14)
        axes[1,0].set_ylabel('Guiding Error\n' + r'Power Spectral Density (mas$^2$/Hz)', fontsize=14)
        axes[1,0].set_yscale('log')

        # Blank - plot to the right of power spectral density
        strings = []
        mag_string = ''
        if isinstance(self.jmag, float):
            mag_string = mag_string + "J = " + f"{self.jmag:.2f}"
        else:
            mag_string = mag_string + "\nJ = " + self.jmag
        if isinstance(self.gmag, float):
            mag_string = mag_string + ", G = " + f"{self.gmag:.2f}"
        else:
            mag_string = mag_string + ", G = " + self.gmag
        strings.append(mag_string)
        strings.append("Elevation = " + f"{self.el:.1f}" + " deg   Airmass = " + f"{self.airmass:.2f}")
        strings.append(str(int(self.gcfps)) + ' fps, ' + str(self.gcgain) + ' gain')
        strings.append('\n')
        strings.append(str(self.df_GUIDER.shape[0]) + ' guider frames.  Fraction with:')
        strings.append('    saturated pixels: ' + "%.2f" % (100*(self.df_GUIDER['object1_peak'] > 15800).sum() / self.df_GUIDER.shape[0]) + '%')
        strings.append('    pixels at >90% saturation: ' + "%.2f" % (100*(self.df_GUIDER['object1_peak'] > 15830*0.9).sum() / self.df_GUIDER.shape[0]) + '%')
        strings.append('\n')
        strings.append("Sun's altitude below horizon = " + str(int(-get_sun_alt(self.date_mid))) + " deg")
        strings.append("Lunar separation = " + str(int(get_moon_sep(self.date_mid, self.ra, self.dec))) + " deg")
        #strings.append('\n')
        #strings.append('Nframes = ' + str(nframes))
        #strings.append('   ' + str(nframes_0stars) + ' with 0 stars detected')
        #strings.append('   ' + str(nframes_1stars) + ' with 1 star detected')
        #strings.append('   ' + str(nframes_2stars) + ' with 2 stars detected')
        #strings.append('   ' + str(nframes_3stars) + ' with 3 stars detected')
        axes[1,1].axis('off')
        axes[1,1].text(0.01, 1.0, '\n'.join(strings), fontsize=14, ha='left', va='top')

        # Guider FWHM time series plot
        fwhm = (self.df_GUIDER.object1_a**2 + self.df_GUIDER.object1_b**2)**0.5 / self.pixel_scale * (2*(2*np.log(2))**0.5)
        if self.nframes > 0:
            axes[2,0].plot(self.t-min(self.t), fwhm, color='royalblue', alpha=0.5)
            axes[2,0].set_xlim(min(self.t-min(self.t)), max(self.t-min(self.t)))
        else:
            axes[2,0].plot([0.], [0.], color='royalblue', alpha=0.5)        
        axes[2,0].grid(True, linestyle='dashed', linewidth=1, alpha=0.5)
#        axes[2,0].set_title('Nframes = ' + str(nframes) + '; median number of detected stars/frame=' + str(median_nstars) + ', ' + str(nframes_fewer_detections) + ' w/fewer, ' + str(nframes_extra_detections) + ' w/more', fontsize=14) 
        axes[2,0].set_xlabel("Time (sec)", fontsize=14)
        axes[2,0].set_ylabel("Guider FWHM (mas)", fontsize=14)
        axes[2,0].legend([r'Guider FWHM ($\neq$ seeing)'], fontsize=12, loc='best') 

        # Histogram of guider FWHM time series plot
        if self.nframes > 0:
            axes[2,1].hist(fwhm, bins=30, color='royalblue', alpha=0.5)
            axes[2,1].grid(True, linestyle='dashed', linewidth=1, alpha=0.5)
        axes[2,1].set_xlabel("Guider FWHM (mas)", fontsize=14)
        axes[2,1].set_ylabel("Samples", fontsize=14)

        # Guider flux time series plot
        flux      = self.df_GUIDER.object1_flux # /np.nanpercentile(self.df_GUIDER.object1_flux, 95)
        peak_flux = self.df_GUIDER.object1_peak
        if self.nframes > 0:
            axes[3,0].plot(self.t-min(self.t), flux, color='royalblue', alpha=0.5)
        else:
            axes[3,0].plot([0.], [0.],      color='royalblue', alpha=0.5)
        axesb = axes[3,0].twinx()
        axesb.set_ylabel(r'Peak Flux (DN pixel$^{-1}$)', color='orange', fontsize=14)
        if self.nframes > 0:
            axesb.plot(self.t-min(self.t), peak_flux, color='orange', alpha=0.5)
            axes[3,0].set_xlim(min(self.t-min(self.t)), max(self.t-min(self.t)))
        else:
            axesb.plot([0.], [0.], color='orange',    alpha=0.5)
        axesb.grid(False)
        axes[3,0].grid(True, linestyle='dashed', linewidth=1, alpha=0.5)
        axes[3,0].set_xlabel("Time (sec)", fontsize=14)
        axes[3,0].set_ylabel("Integrated Flux (DN)", fontsize=14, color='royalblue')
        #axes[3,0].legend(['Guider Flux', 'Peak Guider Flux'], fontsize=12, loc='best') 

        # Histogram of guider flux time series plot
        axesc = axes[3,1].twiny()
        if self.nframes > 0:
            axes[3,1].hist(flux, bins=30, color='royalblue', alpha=0.5)
            axes[3,1].grid(True, linestyle='dashed', linewidth=1, alpha=0.5)
            axesc.hist(peak_flux, bins=30, color='orange', alpha=0.5)
        axesc.set_xlabel(r'Peak Flux (DN pixel$^{-1}$; saturation = 15,830)', color='orange', fontsize=14)
        axesc.grid(False)
        axes[3,1].set_xlabel("Integrated Flux (DN)", fontsize=14, color='royalblue')
        axes[3,1].set_ylabel("Samples", fontsize=14)

        # Set the font size of tick mark labels
        axes[0,0].tick_params(axis='both', which='major', labelsize=14)
        axes[0,1].tick_params(axis='both', which='major', labelsize=10)
        axes[1,0].tick_params(axis='both', which='major', labelsize=14)
        axes[2,0].tick_params(axis='both', which='major', labelsize=14)
        axes[2,1].tick_params(axis='both', which='major', labelsize=10)
        axes[3,0].tick_params(axis='both', which='major', labelsize=14)
        axes[3,1].tick_params(axis='both', which='major', labelsize=10)
        axes[3,1].tick_params(axis='x', labelcolor='royalblue')
        axesc.tick_params(axis='x', labelcolor='orange')
        axes[3,0].tick_params(axis='y', labelcolor='royalblue')
        axesb.tick_params(axis='y', labelcolor='orange')
        axesb.tick_params(axis='both', which='major', labelsize=14)
        axesc.tick_params(axis='both', which='major', labelsize=10)

        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time} UT"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    xytext=(0, -50), textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)     

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=144, facecolor='w')
            if self.logger:
                self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def plot_guider_flux_time_series(self, fig_path=None, show_plot=False):

        """
        Generate a plot of the guider flux time series errors.
        
        To-do: compute the flux in a octagonal aperture centered on the guiding origin

        Args:
            fig_path (string) - set to the path for a SNR vs. wavelength file 
                to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it the current environment 
            (e.g., in a Jupyter Notebook).
        """

        # Construct plots
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(8, 4), tight_layout=True)
        if self.nframes > 0:
            plt.plot(self.t-min(self.t), self.df_GUIDER.object1_flux/np.nanpercentile(self.df_GUIDER.object1_flux, 95), color='royalblue')
            plt.xlim(min(self.t-min(self.t)), max(self.t-min(self.t)))
        else:
            plt.plot([0.], [0.], color='royalblue')        
        plt.title("Guiding Flux Time Series: " + str(self.ObsID)+' - ' + self.name, fontsize=14)
        plt.xlabel("Seconds Start of Exposure", fontsize=14)
        plt.ylabel("Flux (fractional)", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(['Guider Flux', 'Exposure Meter Flux'], fontsize=12, loc='best') 

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=144, facecolor='w')
            if self.logger:
                self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def plot_guider_fwhm_time_series(self, fig_path=None, show_plot=False):

        """
        Generate a plot of the guider FWHM time series errors.

        Args:
            fig_path (string) - set to the path for a SNR vs. wavelength file 
                to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it the current environment 
            (e.g., in a Jupyter Notebook).

        """

        # Make FWHM time series
        fwhm = (self.df_GUIDER.object1_a**2 + self.df_GUIDER.object1_b**2)**0.5 / self.pixel_scale * (2*(2*np.log(2))**0.5)

        # Construct plots
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(8, 4), tight_layout=True)
        if self.nframes > 0:
            plt.plot(self.t-min(self.t), fwhm, color='royalblue')
            plt.xlim(min(self.t-min(self.t)), max(self.t-min(self.t)))
        else:
            plt.plot([0.], [0.], color='royalblue')        
        plt.title("Guider FWHM Time Series: " + str(self.ObsID)+' - ' + self.name, fontsize=14)
        plt.xlabel("Seconds Start of Exposure", fontsize=14)
        plt.ylabel("FWHM (mas)", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend([r'Guider FWHM ($\neq$ seeing)'], fontsize=12, loc='best') 

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=144, facecolor='w')
            if self.logger:
                self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def plot_guider_delta_time_time_series(self, fig_path=None, show_plot=False):
        """
        Plot the time between guider exposures and a histogram of those intervals.
    
        Panels
        ------
        • Left  (2/3 width) : Δt between frames as a function of elapsed time
        • Right (1/3 width) : Histogram of Δt values (log-scaled counts axis)
    
        Args
        ----
        fig_path : str | None
            Path to save the PNG.  If None, nothing is written.
        show_plot : bool
            If True, display the figure (useful in Jupyter).
    
        Returns
        -------
        None
        """
        sns.set_theme(style="whitegrid")
    
        # ── Prepare data ────────────────────────────────────────────────────────
        if getattr(self, "nframes", 0) > 0:
            x_sec = (
                self.df_GUIDER["datetime"] - self.df_GUIDER["datetime"].iloc[0]
            ).dt.total_seconds()
            y_ms = self.df_GUIDER["timestep_ms"]
        else:
            x_sec = pd.Series([0.0])
            y_ms  = pd.Series([0.0])
    
        ref_ms = 1000.0 / self.gcfps   # expected spacing (ms per frame)
    
        # ── Figure with a 2.5 : 1 column split ────────────────────────────────
        fig, (ax_ts, ax_hist) = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(11.5, 4),
            gridspec_kw={"width_ratios": [2.5, 1], "wspace": 0.05},
            sharey=True,
            tight_layout=True,
        )
    
        # ── Centered title spanning both panels ────────────────────────────────
        fig.suptitle(
            f"Time Between Guider Timestamps: {self.ObsID} – {self.name}",
            fontsize=14
        )
    
        # ── Left panel: time-series plot ───────────────────────────────────────
        ax_ts.plot(
            x_sec,
            y_ms,
            color="royalblue",
            linewidth=1.2,
            label=r"$\Delta t$ (ms)"
        )
        ax_ts.axhline(
            ref_ms,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"1 / fps = {ref_ms:.1f} ms"
        )
        ax_ts.set_xlabel("Seconds Since Start of Observation", fontsize=14)
        ax_ts.set_ylabel(r"$\Delta t$ Between Timestamps (ms)", fontsize=14)
        ax_ts.tick_params(labelsize=12)
        ax_ts.legend(fontsize=10, loc="best")
    
        # ── Right panel: histogram (log counts) ────────────────────────────────
        ax_hist.hist(
            y_ms.dropna(),
            bins="auto",
            orientation="horizontal",
            facecolor="royalblue",
            edgecolor="royalblue",
            alpha=0.70,
        )
        ax_hist.set_xscale("log")
        ax_hist.xaxis.set_major_locator(LogLocator(base=10.0))
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        ax_hist.xaxis.set_major_formatter(fmt)
        ax_hist.set_xlabel(r"Number of $\Delta t$ Values", fontsize=13)
        ax_hist.tick_params(labelsize=12)
        ax_hist.axhline(ref_ms, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    
        sns.despine(ax=ax_hist, left=True)
        for spine in ax_hist.spines.values():
            spine.set_visible(True)
        plt.setp(ax_hist.get_yticklabels(), visible=False)
    
        # ── Save / show / close ────────────────────────────────────────────────
        if fig_path is not None:
            t0 = time.process_time()
            fig.savefig(fig_path, dpi=144, facecolor="w")
            if getattr(self, "logger", None):
                self.logger.info(f"savefig took {(time.process_time() - t0):.1f} s")
    
        if show_plot:
            plt.show()
    
        plt.close(fig)








