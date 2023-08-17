import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from astropy.table import Table
from astropy.time import Time
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
        None so far
    """

    def __init__(self, L0, logger=None):
        if logger:
            self.logger = logger
            self.logger.debug('AnalyzeGuider class constructor')
        else:
            self.logger = None
        self.logger.info('Initiating AnanalyzeGuider object.')
        self.L0 = L0
        self.pixel_scale = 0.056 # arcsec per pixel for the CRED-2 imager on the KPF FIU

        header_primary_obj = HeaderParse(L0, 'PRIMARY')
        header_guider_obj  = HeaderParse(L0, 'GUIDER_AVG')
        #header_guider_obj  = HeaderParse(L0, 'guider_avg')
        self.guider_header = header_guider_obj.header
        self.header = header_primary_obj.header
        self.name = header_primary_obj.get_name()
        self.ObsID = header_primary_obj.get_obsid()
        self.date_mid = Time(self.header['DATE-MID'])
        self.ra  = self.header['RA'] # string
        self.dec = self.header['DEC'] # string
        self.gmag = float(self.header['GAIAMAG']) # Gaia G magnitude
        self.jmag = float(self.header['2MASSMAG']) # J magnitude
        self.gcfps = self.header['GCFPS'] # frames per second for guide camera
        self.gcgain = self.header['GCGAIN'] # detector gain setting 
        # to-do: set up logic to determine if L0 is a KPF object or a .fits file
        #self.df_GUIDER = Table.read(self.L0, format='fits',hdu='guider_cube_origins').to_pandas()
        #self.df_GUIDER = Table.read(self.L0, hdu='guider_cube_origins').to_pandas()
        #self.df_GUIDER = self.L0['guider_cube_origins']
        self.df_GUIDER = self.L0['GUIDER_CUBE_ORIGINS']
        self.good_fit = None


    def measure_seeing(self):

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

        def moffat_2D(xy, amplitude, x0, y0, alpha, beta):
            x, y = xy
            return amplitude * (1 + ((x - x0) ** 2 + (y - y0) ** 2) / alpha ** 2) ** -beta

        x = np.arange(self.guider_image.shape[1])
        y = np.arange(self.guider_image.shape[0])
        X, Y = np.meshgrid(x, y)
        x_flat = X.flatten()
        y_flat = Y.flatten()
        image_data_flat = self.guider_image.flatten()
        p0 = [1, self.guider_header['CRPIX1'], self.guider_header['CRPIX2'], 5/0.056, 2.5]  # Initial guess for the parameters
        #p0 = [1, self.guider_image.shape[1] / 2, self.guider_image.shape[0] / 2, 2/0.056, 2]  # Initial guess for the parameters
        
        try:
            popt, pcov = curve_fit(moffat_2D, (x_flat, y_flat), image_data_flat, p0=p0)
            amplitude_fit, x0_fit, y0_fit, alpha_fit, beta_fit = popt
            alpha_fit = abs(alpha_fit)
            self.good_fit = True
            self.image_fit = moffat_2D((X, Y), amplitude_fit, x0_fit, y0_fit, alpha_fit, beta_fit)
            self.amplitude = amplitude_fit
            self.seeing = alpha_fit
            self.seeing_550nm = self.seeing*(((1200+950)/2)/550)**0.2  # seeing scales with wavelength^0.2
            self.beta = beta_fit
            self.x0 = x0_fit
            self.y0 = y0_fit
        except:
            self.good_fit = False
        

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
        #guider_im_zoom = self.guider_image[255-38:255+38, 320-38:320+38] # 4 arcsec x 4 arcsec
        guider_im_zoom = self.guider_image[255-50:255+50, 320-50:320+50]
        if self.good_fit:
            resid_im = self.guider_image - self.image_fit
            resid_im_zoom = resid_im[255-38:255+38, 320-38:320+38]
        else:
            resid_im_zoom = 0*self.guider_image[255-38:255+38, 320-38:320+38]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)

        # Left panel - full image
        im1 = axs[0].imshow(self.guider_image, cmap='viridis', origin='lower', vmin=0, vmax=np.percentile(guider_im_zoom,99.9))
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
        title = title + "\nJ = " + f"{self.jmag:.2f}" + ", G = " + f"{self.gmag:.2f}" + ', ' + str(int(self.gcfps)) + ' fps, ' + str(self.gcgain) + ' gain'
        axs[0].set_title(title, fontsize=12)
        axs[0].grid(True, linestyle='solid', linewidth=0.5, alpha=0.5)
        #cmap = plt.get_cmap('viridis')
        #axs[0].set_facecolor(cmap(0))
        #cbar1 = plt.colorbar(im1, ax=axs[0], shrink=0.5)

        # Middle panel - zoomed image
        im2 = axs[1].imshow(guider_im_zoom, cmap='viridis', origin='lower', vmin=0, vmax=np.percentile(guider_im_zoom,99.9))
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
        #axs[1].set_facecolor(cmap(0))

        # Right panel - zoomed image of residuals to model
        im2 = axs[2].imshow(resid_im_zoom, cmap='viridis', origin='lower', vmin=0, vmax=np.percentile(guider_im_zoom,99.9))
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
        axs[2].set_xticklabels([f'{int(x * self.pixel_scale*10)/10}' for x in xticks])
        axs[2].set_yticklabels([f'{int(y * self.pixel_scale*10)/10}' for y in yticks])
        axs[2].set_xlabel('Arcseconds', fontsize=12)
        axs[2].set_ylabel('Arcseconds', fontsize=12)
        if self.good_fit:
            title = 'Residuals to Moffat Function Model'
            axs[2].set_title(title, fontsize=12)
        else:
            title = 'Unsuccessful fit to Moffat Function Model'
            axs[2].set_title(title, fontsize=12, color='r')  # red to indicate unusual image
        axs[2].set_title(title, fontsize=12)
        axs[2].grid(True, linestyle='solid', linewidth=0.5, alpha=0.5)
        cbar3 = plt.colorbar(im2, ax=axs[2], shrink=0.7)
        #axs[0].set_facecolor(cmap(0))
            
        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=288, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def measure_guider_errors(self):

        """
        Compute the guiding error RMS for X/Y/R errors. -- TO BE WRITTEN!

        Args:
            None

        Attributes:
            TBD

        Returns:
            None
        """


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
            (e.g., in a Jupyter Notebook).

        """
        
        if np.sqrt(self.df_GUIDER.shape[0]) < 60:
            hist_bins = 25
        else:
            hist_bins = 40
        
        # Create the figure and subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 4), gridspec_kw={'width_ratios': [2, 1]}, tight_layout=True)
        plt.style.use('seaborn-whitegrid')

        # Define datasets
        t     =  self.df_GUIDER.timestamp-min(self.df_GUIDER.timestamp)
        x_mas = (self.df_GUIDER.target_x - self.df_GUIDER.object1_x) * self.pixel_scale*1000
        y_mas = (self.df_GUIDER.target_y - self.df_GUIDER.object1_y) * self.pixel_scale*1000
        r_mas = (x_mas**2+y_mas**2)**0.5
        
        x_rms = (np.nanmean(x_mas**2))**0.5
        y_rms = (np.nanmean(y_mas**2))**0.5
        r_rms = (np.nanmean(r_mas**2))**0.5
        x_bias = np.nanmean(x_mas)
        y_bias = np.nanmean(y_mas)

        # Plot the data
        im1 = axes[1].hist2d(x_mas, y_mas, bins=hist_bins, cmap='viridis')
        cmap = plt.get_cmap('viridis')
        axes[1].set_facecolor(cmap(0))
        xylim = round(np.nanpercentile(r_mas, 99)*1.4 / 5.0) * 5
        axes[1].set_xlim(-xylim, xylim) # set symmetric limits for x and y
        axes[1].set_ylim(-xylim, xylim)
        axes[1].set_aspect('equal')
        axes[1].set_title('r: ' + f'{int(r_rms*10)/10}' + ' mas (RMS)', fontsize=14)
        axes[1].set_xlabel('Guiding Error - x (mas)', fontsize=14)
        axes[1].set_ylabel('Guiding Error - y (mas)', fontsize=14)
        axes[1].grid(True, linestyle='solid', linewidth=0.5, alpha=0.5)
        cbar = plt.colorbar(im1[3])
        cbar.set_label('Samples', fontsize=12)

        axes[0].plot(t, x_mas, color='royalblue')
        axes[0].plot(t, y_mas, color='orange')
        axes[0].set_title("Guiding Error Time Series: " + str(self.ObsID)+' - ' + self.name, fontsize=14)
        axes[0].set_xlabel("Time (sec)", fontsize=14)
        axes[0].set_ylabel("Guiding Error (mas)", fontsize=14)
        axes[0].legend(['Guiding error - x: ' + f'{int(x_rms*10)/10}' + ' mas (RMS),' + f'{int(x_bias*100)/100}' + ' mas (bias)', 
                        'Guiding error - y: ' + f'{int(y_rms*10)/10}' + ' mas (RMS),' + f'{int(y_bias*100)/100}' + ' mas (bias)'], 
                       fontsize=12, 
                       loc='best') 

        # Set the font size of tick mark labels
        axes[0].tick_params(axis='both', which='major', labelsize=14)
        axes[1].tick_params(axis='both', which='major', labelsize=14)

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=144, facecolor='w')
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
        fig, axes = plt.subplots(4, 2, figsize=(16, 16), gridspec_kw={'width_ratios': [2, 1]}, tight_layout=True)
        plt.style.use('seaborn-whitegrid')

        # Count number of stars
        nstars = []
        for index, row in self.df_GUIDER.iterrows():
            star_count = 0
            if row['object1_x'] < -998: star_count += 1
            if row['object2_x'] < -998: star_count += 1
            if row['object3_x'] < -998: star_count += 1
            nstars.append(star_count)
        nstars = np.array(nstars, dtype=int)
        nframes_0stars = len(np.where(nstars == 0)[0])
        nframes_1stars = len(np.where(nstars == 1)[0])
        nframes_2stars = len(np.where(nstars == 2)[0])
        nframes_3stars = len(np.where(nstars == 3)[0])
        median_nstars = int(np.median(nstars))
        w_extra_detections = np.where(nstars > median_nstars)[0]
        nframes_extra_detections = len(w_extra_detections)
        w_fewer_detections = np.where(nstars < median_nstars)[0]
        nframes_fewer_detections = len(w_fewer_detections)

        # Define datasets and statistics
        nframes = self.df_GUIDER.shape[0]
        t     =  self.df_GUIDER.timestamp-min(self.df_GUIDER.timestamp)
        x_mas = (self.df_GUIDER.target_x - self.df_GUIDER.object1_x) * self.pixel_scale*1000
        y_mas = (self.df_GUIDER.target_y - self.df_GUIDER.object1_y) * self.pixel_scale*1000
        r_mas = (x_mas**2+y_mas**2)**0.5
        x_rms = (np.nanmean(x_mas**2))**0.5
        y_rms = (np.nanmean(y_mas**2))**0.5
        r_rms = (np.nanmean(r_mas**2))**0.5
        x_bias = np.nanmean(x_mas)
        y_bias = np.nanmean(y_mas)
        
        # Set the number of histogram bins
        if np.sqrt(self.df_GUIDER.shape[0]) < 60:
            hist_bins = 25
        else:
            hist_bins = 40
        if max(max(x_mas), abs(min(x_mas)), max(y_mas), abs(min(y_mas))) / hist_bins > 3:  # if a few really big errors dominate
            hist_bins = 2 * int(max(max(x_mas), abs(min(x_mas)), max(y_mas), abs(min(y_mas))) / 3)

        # Histogram of guider errors
        hist = axes[0,1].hist2d(x_mas, y_mas, bins=hist_bins, cmap='viridis')
        cbar_ax = fig.add_axes([0.95, 0.775, 0.01, 0.20])  # Adjust these values to properly position your colorbar
        fig.colorbar(hist[3], cax=cbar_ax, label='Samples')#, fontsize=12)
        #cbar = plt.colorbar(hist[3])
        #cbar_ax.set_label('Samples', fontsize=12)
        cmap = plt.get_cmap('viridis')
        axes[0,1].set_facecolor(cmap(0))
        xylim = round(np.nanpercentile(r_mas, 95)*1.4 / 5.0) * 5
        axes[0,1].set_xlim(-xylim, xylim) # set symmetric limits for x and y
        axes[0,1].set_ylim(-xylim, xylim)
        axes[0,1].set_aspect('equal')
        axes[0,1].set_title('r: ' + f'{int(r_rms*10)/10}' + ' mas (RMS)', fontsize=14)
        axes[0,1].set_xlabel('Guiding Error - x (mas)', fontsize=14)
        axes[0,1].set_ylabel('Guiding Error - y (mas)', fontsize=14)
        axes[0,1].grid(True, linestyle='solid', linewidth=0.5, alpha=0.5)

        # Time series plot of guider errors
        axes[0,0].plot(t, x_mas, color='royalblue')
        axes[0,0].plot(t, y_mas, color='orange')
        axes[0,0].set_title("Guiding Error Time Series: " + str(self.ObsID)+' - ' + self.name, fontsize=14)
        axes[0,0].set_xlabel("Time (sec)", fontsize=14)
        axes[0,0].set_ylabel("Guiding Error (mas)", fontsize=14)
        axes[0,0].set_xlim(0, max(t)) 
        axes[0,0].legend(['Guiding error - x: ' + f'{int(x_rms*10)/10}' + ' mas (RMS), ' + f'{int(x_bias*10)/10}' + ' mas (bias)', 
                          'Guiding error - y: ' + f'{int(y_rms*10)/10}' + ' mas (RMS), ' + f'{int(y_bias*10)/10}' + ' mas (bias)'], 
                         fontsize=12, 
                         loc='best') 

        # Power spectral density plot
        fps = self.gcfps # my_Guider.guider_header['FPS']  # Sample rate in Hz
        Pxx, freqs = mlab.psd(x_mas/1000, Fs=fps)
        Pyy, freqs = mlab.psd(y_mas/1000, Fs=fps)
        Prr, freqs = mlab.psd(r_mas/1000, Fs=fps)
#        axes[1,0].step(freqs, Prr*1e6, where='mid', color='b', alpha=0.8, label='R - Guiding Errors')
        axes[1,0].step(freqs, Pxx*1e6, where='mid', color='royalblue', label='X - Guiding errors', lw=2)
        axes[1,0].step(freqs, Pyy*1e6, where='mid', color='orange',    label='Y - Guiding errors', lw=2)
        axes[1,0].grid(True, linestyle='dashed', linewidth=1, alpha=0.5)
        axes[1,0].set_xlabel('Frequency [Hz]', fontsize=14)
        axes[1,0].set_ylabel('Guiding Error\n' + r'Power Spectral Density (mas$^2$/Hz)', fontsize=14)
        axes[1,0].set_xlim(min(freqs),max(freqs))
        axes[1,0].set_yscale('log')
        axes[1,0].legend(fontsize=12)

        # Blank - plot to the right of power spectral density
        strings = ["Sun's altitude below horizon = " + str(int(-get_sun_alt(self.date_mid))) + " deg"]
        strings.append("Lunar separation = " + str(int(get_moon_sep(self.date_mid, self.ra, self.dec))) + " deg")
        strings.append('\n')
        strings.append('Nframes = ' + str(nframes))
        strings.append('   ' + str(nframes_0stars) + ' with 0 stars detected')
        strings.append('   ' + str(nframes_1stars) + ' with 1 star detected')
        strings.append('   ' + str(nframes_2stars) + ' with 2 stars detected')
        strings.append('   ' + str(nframes_3stars) + ' with 3 stars detected')
        axes[1,1].axis('off')
        axes[1,1].text(0.03, 0.9, '\n'.join(strings), fontsize=14, ha='left', va='top')

        # Guider FWHM time series plot
        fwhm = (self.df_GUIDER.object1_a**2 + self.df_GUIDER.object1_b**2)**0.5 / self.pixel_scale * (2*(2*np.log(2))**0.5)
        axes[2,0].plot(self.df_GUIDER.timestamp-min(self.df_GUIDER.timestamp), fwhm, color='royalblue')
        axes[2,0].grid(True, linestyle='dashed', linewidth=1, alpha=0.5)
#        axes[2,0].set_title('Nframes = ' + str(nframes) + '; median number of detected stars/frame=' + str(median_nstars) + ', ' + str(nframes_fewer_detections) + ' w/fewer, ' + str(nframes_extra_detections) + ' w/more', fontsize=14) 
        axes[2,0].set_xlabel("Time (sec)", fontsize=14)
        axes[2,0].set_ylabel("Guider FWHM (mas)", fontsize=14)
        axes[2,0].set_xlim(min(self.df_GUIDER.timestamp-min(self.df_GUIDER.timestamp)), max(self.df_GUIDER.timestamp-min(self.df_GUIDER.timestamp)))
        axes[2,0].legend([r'Guider FWHM ($\neq$ seeing)'], fontsize=12, loc='best') 

        # Histogram of guider FWHM time series plot
        axes[2,1].hist(fwhm, bins=30, color='royalblue', edgecolor='k')
        axes[2,1].set_xlabel("Guider FWHM (mas)", fontsize=14)
        axes[2,1].set_ylabel("Frequency", fontsize=14)

        # Guider flux time series plot
        flux = self.df_GUIDER.object1_flux/np.nanpercentile(self.df_GUIDER.object1_flux, 95)
        axes[3,0].plot(self.df_GUIDER.timestamp-min(self.df_GUIDER.timestamp), flux, color='royalblue')
        axes[3,0].grid(True, linestyle='dashed', linewidth=1, alpha=0.5)
        axes[3,0].set_xlabel("Time (sec)", fontsize=14)
        axes[3,0].set_ylabel("Guider Flux (fractional)", fontsize=14)
        axes[3,0].set_xlim(min(self.df_GUIDER.timestamp-min(self.df_GUIDER.timestamp)), max(self.df_GUIDER.timestamp-min(self.df_GUIDER.timestamp)))
        axes[3,0].legend([r'Guider Flux (fractional, normalized by 95th percentile)'], fontsize=12, loc='best') 

        # Histogram of guider flux time series plot
        axes[3,1].hist(flux, bins=30, color='royalblue', edgecolor='k')
        axes[3,1].set_xlabel("Flux (fractional)", fontsize=14)
        axes[3,1].set_ylabel("Frequency", fontsize=14)

        # Set the font size of tick mark labels
        axes[0,0].tick_params(axis='both', which='major', labelsize=14)
        axes[0,1].tick_params(axis='both', which='major', labelsize=14)
        axes[1,0].tick_params(axis='both', which='major', labelsize=14)
        axes[2,0].tick_params(axis='both', which='major', labelsize=14)
        axes[3,0].tick_params(axis='both', which='major', labelsize=14)

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=144, facecolor='w')
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
        plt.style.use('seaborn-whitegrid')
        plt.figure(figsize=(8, 4), tight_layout=True)
        plt.plot(self.df_GUIDER.timestamp-min(self.df_GUIDER.timestamp), self.df_GUIDER.object1_flux/np.nanpercentile(self.df_GUIDER.object1_flux, 95), color='royalblue')
        #plt.plot(time, int_SCI_flux / ((847+4.8/2)-(450.1-0.4/2)) / tdur_sec / max(int_SCI_flux / ((847+4.8/2)-(450.1-0.4/2)) / tdur_sec), marker='o', color='k')
        plt.title("Guiding Flux Time Series: " + str(self.ObsID)+' - ' + self.name, fontsize=14)
        plt.xlabel("Seconds since " + str(self.guider_header['DATE-BEG']), fontsize=14)
        plt.ylabel("Flux (fractional)", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(min(self.df_GUIDER.timestamp-min(self.df_GUIDER.timestamp)), max(self.df_GUIDER.timestamp-min(self.df_GUIDER.timestamp)))
        plt.legend(['Guider Flux', 'Exposure Meter Flux'], fontsize=12, loc='best') 

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=144, facecolor='w')
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
        plt.style.use('seaborn-whitegrid')
        plt.figure(figsize=(8, 4), tight_layout=True)
        plt.plot(self.df_GUIDER.timestamp-min(self.df_GUIDER.timestamp), fwhm, color='royalblue')
        plt.title("Guider FWHM Time Series: " + str(self.ObsID)+' - ' + self.name, fontsize=14)
        plt.xlabel("Seconds since " + str(self.guider_header['DATE-BEG']), fontsize=14)
        plt.ylabel("FWHM (mas)", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(min(self.df_GUIDER.timestamp-min(self.df_GUIDER.timestamp)), max(self.df_GUIDER.timestamp-min(self.df_GUIDER.timestamp)))
        plt.legend([r'Guider FWHM ($\neq$ seeing)'], fontsize=12, loc='best') 

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=144, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')
