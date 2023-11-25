import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from modules.Utils.kpf_parse import HeaderParse
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline

class AnalyzeL1:
    """
    Description:
        This class contains functions to analyze L1 spectra (storing them
        as attributes) and functions to plot the results.

    Arguments:
        L1 - an L1 object

    Attributes:
        GREEN_SNR (from compute_l1_snr): Two-dimensional array of
            SNR values for the Green CCD.  The first array index specifies
            the spectral order (0-34 = green, 0-31 = red).
            The second array index specifies the orderlet:
            0=CAL, 1=SCI1, 2=SCI2, 3=SCI3, 4=SKY, 5=SCI1+SCI2+SCI3
        RED_SNR (from compute_l1_snr): Similar to GREEN_SNR,
            but for the Red CCD.
        GREEN_SNR_WAV (from compute_l1_snr): One-dimensional array
            of the wavelength of the middle of the spectral orders on
            the green CCD.
        RED_SNR_WAV (from compute_l1_snr): Similar to GREEN_SNR,
            but for the Red CCD.
    """

    def __init__(self, L1, logger=None):
        if logger:
            self.logger = logger
            self.logger.debug('Initializing AnalyzeL1 object')
        else:
            self.logger = None
        self.L1 = L1
        #self.header = L1['PRIMARY'].header
        primary_header = HeaderParse(L1, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        self.ObsID = primary_header.get_obsid()
        

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
        #GREEN_SCI_SNR1 = np.zeros(L1['GREEN_SCI_VAR1'])
        GREEN_SCI_SNR1 = 0 * L1['GREEN_SCI_VAR2']
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
            RE_PEAKD_FLUX[o,5] = np.nanpercentile(L1['RED_SCI_FLUX1'][o]+L1['RED_SCI_FLUX2'][o]+L1['RED_SCI_FLUX3'][o], counts_percentile)

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
        ax2.scatter(self.RED_SNR_WAV[1:],   self.RED_SNR[1:,4],   marker="D", color='r', label='SKY')
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
        ax1.set_ylabel('SNR - SCI', fontsize=14)
        ax2.set_ylabel('SNR - SKY', fontsize=14)
        ax3.set_ylabel('SNR - CAL', fontsize=14)
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
        ax2.scatter(self.RED_SNR_WAV[1:],   self.RED_PEAK_FLUX[1:,4],   marker="D", color='r', label='SKY')
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
        ax1.set_title(self.ObsID + ' - ' + self.name, fontsize=16)
        ax3.set_xlabel('Wavelength [Ang]', fontsize=14)
        ax1.set_ylabel(str(self.counts_percentile) + 'th %ile Flux [e-] - SCI', fontsize=14)
        ax2.set_ylabel(str(self.counts_percentile) + 'th %ile Flux [e-] - SKY', fontsize=14)
        ax3.set_ylabel(str(self.counts_percentile) + 'th %ile Flux [e-] - CAL', fontsize=14)
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

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=300, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def plot_L1_spectrum(self, orderlet=None, fig_path=None, show_plot=False):
        """
        Generate a rainbow-colored plot L1 spectrum.  One must select an orderlet.

        Args:
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
            flux_green = np.array(self.L1['GREEN_SCI_FLUX1'].data,'d')
            flux_red   = np.array(self.L1['RED_SCI_FLUX1'].data,'d')
        elif orderlet.lower() == 'sci2':
            wav_green  = np.array(self.L1['GREEN_SCI_WAVE2'].data,'d')
            wav_red    = np.array(self.L1['RED_SCI_WAVE2'].data,'d')
            flux_green = np.array(self.L1['GREEN_SCI_FLUX2'].data,'d')
            flux_red   = np.array(self.L1['RED_SCI_FLUX2'].data,'d')
        elif orderlet.lower() == 'sci3':
            wav_green  = np.array(self.L1['GREEN_SCI_WAVE3'].data,'d')
            wav_red    = np.array(self.L1['RED_SCI_WAVE3'].data,'d')
            flux_green = np.array(self.L1['GREEN_SCI_FLUX3'].data,'d')
            flux_red   = np.array(self.L1['RED_SCI_FLUX3'].data,'d')
        elif orderlet.lower() == 'sky':
            wav_green  = np.array(self.L1['GREEN_SKY_WAVE'].data,'d')
            wav_red    = np.array(self.L1['RED_SKY_WAVE'].data,'d')
            flux_green = np.array(self.L1['GREEN_SKY_FLUX'].data,'d')
            flux_red   = np.array(self.L1['RED_SKY_FLUX'].data,'d')
        elif orderlet.lower() == 'cal':
            wav_green  = np.array(self.L1['GREEN_CAL_WAVE'].data,'d')
            wav_red    = np.array(self.L1['RED_CAL_WAVE'].data,'d')
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
        low, high = np.nanpercentile(flux,[0.1,99.9])
        ax[int(np.shape(wav)[0]/n_orders_per_panel/2)].set_ylabel('Counts (e-) in ' + orderlet.upper(),fontsize = 28)
        plt.xlabel('Wavelength (Ang)',fontsize = 28)

        # Add overall title to array of plots
        ax = fig.add_subplot(111, frame_on=False)
        ax.grid(False)
        ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        ax.set_title('L1 Spectrum of ' + orderlet.upper() + ': ' + str(self.ObsID) + ' - ' + self.name, fontsize=28)
        plt.tight_layout()

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
            order (int) - spectral order to plot
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

        plt.figure(figsize=(12, 4), tight_layout=True)
        if 'sci1' in orderlet_lowercase:
            plt.plot(wav_sci1, flux_sci1, linewidth=0.5, label='SCI1')
        if 'sci2' in orderlet_lowercase:
            plt.plot(wav_sci2, flux_sci2, linewidth=0.5, label='SCI2')
        if 'sci3' in orderlet_lowercase:
            plt.plot(wav_sci3, flux_sci3, linewidth=0.5, label='SCI3')
        if 'sky' in orderlet_lowercase:
            plt.plot(wav_sci3, flux_sky,  linewidth=0.5, label='SKY')
        if 'cal' in orderlet_lowercase:
            plt.plot(wav_sci3, flux_cal,  linewidth=0.5, label='CAL')
        plt.xlim(min(wav_sci1), max(wav_sci1))
        plt.title('L1 (' + orderlet_label + ') - ' + chip_title + ' CCD: ' + str(self.ObsID) + ' - ' + self.name, fontsize=14)
        plt.xlabel('Wavelength (Ang)', fontsize=14)
        plt.ylabel('Counts (e-)', fontsize=14)
        if ylog: plt.yscale('log')
        plt.grid(True)
        plt.legend()

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
        1D interpolation function that uses Bsplines unless the input wavelengths are non-monotonic, 
        in which case it uses cubic splines.  This function is used in measure_orderlet_flux_ratio().
        """
        
        if np.any(wav[1:] < wav[:-1]):
            monotonic = False #B-spline is not compatabile with non-monotonic WLS (which we should eliminate anyway)
        else:
            monotonic = True
        if monotonic == True:
            interpolator = make_interp_spline(wav, flux, k=3)
            newflux = interpolator(newwav)
        else:
             interpolator = interp1d(wav, flux, kind='cubic', fill_value='extrapolate')
             newflux = interpolator(newwav)   
        return newflux

    def measure_orderlet_flux_ratios(self):
        """
        Compute the flux ratios of SCI2/SCI1, SCI3/SCI1, CAL/SCI1, SKY/SCI1.

        Args:
            None

        Attributes:
            To be added.

Use the same wavelengths and orders as the SNR calculation:
        SNRSC452 - SNR of L1 SCI spectrum (SCI1+SCI2+SCI3) near 452 nm (second bluest order); on Green CCD
        SNRSC548 - SNR of L1 SCI spectrum (SCI1+SCI2+SCI3) near 548 nm; on Green CCD
        SNRSC661 - SNR of L1 SCI spectrum (SCI1+SCI2+SCI3) near 661 nm; on Red CCD
        SNRSC747 - SNR of L1 SCI spectrum (SCI1+SCI2+SCI3) near 747 nm; on Red CCD
        SNRSC865 - SNR of L1 SCI (SCI1+SCI2+SCI3) near 865 nm (second reddest order); on Red CCD

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
        self.ratio_g_sky_sci2  = np.zeros(35)
        self.ratio_g_cal_sci2  = np.zeros(35)
        self.ratio_r_sci1_sci2 = np.zeros(32)
        self.ratio_r_sci3_sci2 = np.zeros(32)
        self.ratio_r_sky_sci2  = np.zeros(32)
        self.ratio_r_cal_sci2  = np.zeros(32)
        
        # Define orderlet-to-orderlet ratios over all orders
        self.f_sci1_flat = np.hstack((self.f_g_sci1.flatten(), self.f_r_sci1.flatten()))
        self.f_sci2_flat = np.hstack((self.f_g_sci2.flatten(), self.f_r_sci2.flatten()))
        self.f_sci3_flat = np.hstack((self.f_g_sci3.flatten(), self.f_r_sci3.flatten()))
        self.f_sky_flat  = np.hstack((self.f_g_sky.flatten(),  self.f_r_sky.flatten()))
        self.f_cal_flat  = np.hstack((self.f_g_cal.flatten(),  self.f_r_cal.flatten()))
        self.f_sci2_flat_ind = self.f_sci2_flat != 0
        self.ratio_sci1_sci2 = np.nanmedian(self.f_sci1_flat[self.f_sci2_flat_ind]/self.f_sci2_flat[self.f_sci2_flat_ind])
        self.ratio_sci3_sci2 = np.nanmedian(self.f_sci3_flat[self.f_sci2_flat_ind]/self.f_sci2_flat[self.f_sci2_flat_ind])
        self.ratio_sky_sci2  = np.nanmedian(self.f_sky_flat[self.f_sci2_flat_ind] /self.f_sci2_flat[self.f_sci2_flat_ind])
        self.ratio_cal_sci2  = np.nanmedian(self.f_cal_flat[self.f_sci2_flat_ind] /self.f_sci2_flat[self.f_sci2_flat_ind])
        
        # Compute ratios
        for o in np.arange(35):
            ind = (self.f_g_sci2[o,:] != 0) 
            self.ratio_g_sci1_sci2[o] = np.nanmedian(self.f_g_sci1_int[o,ind]/self.f_g_sci2[o,ind])
            self.ratio_g_sci3_sci2[o] = np.nanmedian(self.f_g_sci3_int[o,ind]/self.f_g_sci2[o,ind])
            self.ratio_g_sky_sci2[o]  = np.nanmedian(self.f_g_sky_int[o,ind] /self.f_g_sci2[o,ind])
            self.ratio_g_cal_sci2[o]  = np.nanmedian(self.f_g_cal_int[o,ind] /self.f_g_sci2[o,ind])
        for o in np.arange(32):
            ind = (self.f_r_sci2[o,:] != 0) 
            self.ratio_r_sci1_sci2[o] = np.nanmedian(self.f_r_sci1_int[o,ind]/self.f_r_sci2[o,ind])
            self.ratio_r_sci3_sci2[o] = np.nanmedian(self.f_r_sci3_int[o,ind]/self.f_r_sci2[o,ind])
            self.ratio_r_sky_sci2[o]  = np.nanmedian(self.f_r_sky_int[o,ind] /self.f_r_sci2[o,ind])
            self.ratio_r_cal_sci2[o]  = np.nanmedian(self.f_r_cal_int[o,ind] /self.f_r_sci2[o,ind])
        
        # Compute median flux ratio per order
        

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

        fig, axs = plt.subplots(4, 1, figsize=(18, 12), sharex=True, tight_layout=True)
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
        
        # SKY / SCI2
        ind_g = (self.ratio_g_sky_sci2 != 0)
        ind_r = (self.ratio_r_sky_sci2 != 0)
        axs[2].scatter(self.w_g_order[ind_g], self.ratio_g_sky_sci2[ind_g], s=100, facecolors='green', edgecolors='black', zorder=2)
        axs[2].plot(   self.w_g_order[ind_g], self.ratio_g_sky_sci2[ind_g], 'k-', zorder=1) 
        axs[2].scatter(self.w_r_order[ind_r], self.ratio_r_sky_sci2[ind_r], s=100, marker='D', facecolors='darkred', edgecolors='black', zorder=2)
        axs[2].plot(   self.w_r_order[ind_r], self.ratio_r_sky_sci2[ind_r], 'k-', zorder=1) 
        axs[2].set_ylabel('SKY / SCI2', fontsize=18)
        axs[2].axhline(self.ratio_sky_sci2, color='gray', linestyle='--', label=r'median(SKY$_\mathrm{interp}$(WAV2) / SCI2(WAV2) = %.5f)' % self.ratio_sky_sci2)
        axs[2].legend(fontsize=16, loc='upper right')
        
        # CAL / SCI2
        axs[3].scatter(self.w_g_order, self.ratio_g_cal_sci2, s=100, facecolors='green', edgecolors='black', zorder=2)
        axs[3].plot(   self.w_g_order, self.ratio_g_cal_sci2, 'k-', zorder=1) 
        axs[3].scatter(self.w_r_order, self.ratio_r_cal_sci2, s=100, marker='D', facecolors='darkred', edgecolors='black', zorder=2)
        axs[3].plot(   self.w_r_order, self.ratio_r_cal_sci2, 'k-', zorder=1) 
        axs[3].set_ylabel('CAL / SCI2', fontsize=18)
        axs[3].axhline(self.ratio_cal_sci2, color='gray', linestyle='--', label=r'median(CAL$_\mathrm{interp}$(WAV2) / SCI2(WAV2) = %.5f)' % self.ratio_cal_sci2)
        axs[3].legend(fontsize=16, loc='upper right')
        axs[3].set_xlabel('Wavelength (Ang)', fontsize=18)

        for ax in axs:
            ax.tick_params(axis='both', which='major', labelsize=14)

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

        # Create a 4x4 array of subplots with no vertical space between cells
        fig, axs = plt.subplots(4, 3, sharex='col', sharey='row', figsize=(18, 12))
        for i in range(4):
            for j in range(3):
#                axs[i, j].set_xlabel('Wavelength (Ang)', fontsize=18)
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
        med = np.median(f_sky_int[o,imin:imax] / f_sci2[o,imin:imax])
        med_unc = uncertainty_median(f_sky_int[o,imin:imax] / f_sci2[o,imin:imax])
        axs[2,0].plot(w_sci2[o,imin:imax], f_sky_int[o,imin:imax] / f_sci2[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='orchid') 
        axs[2,0].legend(loc='upper right')
        axs[2,0].set_ylabel('SKY / SCI2', fontsize=18)
        axs[2,0].grid()
        o=o2; imin = imin2; imax = imax2
        med = np.median(f_sky_int[o,imin:imax] / f_sci2[o,imin:imax])
        med_unc = uncertainty_median(f_sky_int[o,imin:imax] / f_sci2[o,imin:imax])
        axs[2,1].plot(w_sci2[o,imin:imax], f_sky_int[o,imin:imax] / f_sci2[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='orchid') 
        axs[2,1].legend(loc='upper right')
        axs[2,1].grid()
        o=o3; imin = imin3; imax = imax3
        axs[2,2].plot(w_sci2[o,imin:imax], f_sky_int[o,imin:imax] / f_sci2[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='orchid') 
        med = np.median(f_sky_int[o,imin:imax] / f_sci2[o,imin:imax])
        med_unc = uncertainty_median(f_sky_int[o,imin:imax] / f_sci2[o,imin:imax])
        axs[0,0].legend(loc='upper right')
        axs[2,2].grid()
        
        # Row 3
        o=o1; imin = imin1; imax = imax1
        med = np.median(f_cal_int[o,imin:imax] / f_sci2[o,imin:imax])
        med_unc = uncertainty_median(f_cal_int[o,imin:imax] / f_sci2[o,imin:imax])
        axs[3,0].plot(w_sci2[o,imin:imax], f_cal_int[o,imin:imax] / f_sci2[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='turquoise') 
        axs[3,0].legend(loc='upper right')
        axs[3,0].set_ylabel('CAL / SCI2', fontsize=18)
        axs[3,0].set_xlabel('Wavelength (Ang)', fontsize=18)
        axs[3,0].grid()
        o=o2; imin = imin2; imax = imax2
        med = np.median(f_cal_int[o,imin:imax] / f_sci2[o,imin:imax])
        med_unc = uncertainty_median(f_cal_int[o,imin:imax] / f_sci2[o,imin:imax])
        axs[3,1].plot(w_sci2[o,imin:imax], f_cal_int[o,imin:imax] / f_sci2[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='turquoise') 
        axs[3,1].legend(loc='upper right')
        axs[3,1].set_xlabel('Wavelength (Ang)', fontsize=18)
        axs[3,1].grid()
        o=o3; imin = imin3; imax = imax3
        med = np.median(f_cal_int[o,imin:imax] / f_sci2[o,imin:imax])
        med_unc = uncertainty_median(f_cal_int[o,imin:imax] / f_sci2[o,imin:imax])
        axs[3,2].plot(w_sci2[o,imin:imax], f_cal_int[o,imin:imax] / f_sci2[o,imin:imax], 
                      label='median = ' + f'{med:07.5f}' + '$\pm$' + f'{med_unc:07.5f}', 
                      linewidth=0.3, color='turquoise') 
        axs[3,2].legend(loc='upper right')
        axs[3,2].set_xlabel('Wavelength (Ang)', fontsize=18)
        axs[3,2].grid()

        plt.subplots_adjust(hspace=0,wspace=0) # Adjust layout to remove vertical space between subplots

        # Add overall title to array of plots
        ax = fig.add_subplot(111, frame_on=False)
        ax.grid(False)
        ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        ax.set_title('L1 Orderlet Flux Ratios - ' + chip_title + ' CCD: ' + str(self.ObsID) + ' - ' + self.name+ '\n', fontsize=24)
        plt.tight_layout()

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=400, facecolor='w')
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

