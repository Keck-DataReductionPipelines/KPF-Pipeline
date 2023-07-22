import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from modules.Utils.kpf_parse import HeaderParse

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
        

    def measure_L1_snr(self, snr_percentile=95):

        """
        Compute the signal-to-noise ratio (SNR) for each spectral order and
        orderlet in an L1 spectrum from KPF.
        SNR is defined as signal / sqrt(abs(variance)) and can be negative.

        Args:
            snr_percentile: snr_percentile in the SNR distribution for each combination of
                order and orderlet

        Attributes:
            GREEN_SNR - Two-dimensional array of SNR values for the Green CCD.
                The first array index specifies the spectral order
                (0-34 = green, 0-31 = red).  The second array index
                specifies the orderlet:
                0=CAL, 1=SCI1, 2=SCI2, 3=SCI3, 4=SKY, 5=SCI1+SCI2+SCI3.
                For example, GREEN_SNR[1,2] is the SNR for order=1 and the
                SCI2 orderlet.
            RED_SNR - Similar to GREEN_SNR, but for the Red CCD.
            GREEN_SNR_WAV - One-dimensional array of the wavelength of the
                middle of the spectral orders on the green CCD.
            RED_SNR_WAV - Similar to GREEN_SNR, but for the Red CCD.

        Returns:
            None
        """
        L1 = self.L1
        self.snr_percentile = snr_percentile

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
        GREEN_SNR = np.zeros((norders_green, norderlets+1))
        RED_SNR   = np.zeros((norders_red, norderlets+1))
        GREEN_SNR_WAV = np.zeros(norders_green)
        RED_SNR_WAV   = np.zeros(norders_red)

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
            GREEN_SNR[o,0] = np.percentile(GREEN_CAL_SNR[o], snr_percentile)
            GREEN_SNR[o,1] = np.percentile(GREEN_SCI_SNR1[o], snr_percentile)
            GREEN_SNR[o,2] = np.percentile(GREEN_SCI_SNR2[o], snr_percentile)
            GREEN_SNR[o,3] = np.percentile(GREEN_SCI_SNR3[o], snr_percentile)
            GREEN_SNR[o,4] = np.percentile(GREEN_SKY_SNR[o], snr_percentile)
            GREEN_SNR[o,5] = np.percentile(GREEN_SCI_SNR[o], snr_percentile)
        for o in range(norders_red):
            RED_SNR_WAV[o] = L1['RED_SCI_WAVE1'][o,2040]
            RED_SNR[o,0] = np.percentile(RED_CAL_SNR[o], snr_percentile)
            RED_SNR[o,1] = np.percentile(RED_SCI_SNR1[o], snr_percentile)
            RED_SNR[o,2] = np.percentile(RED_SCI_SNR2[o], snr_percentile)
            RED_SNR[o,3] = np.percentile(RED_SCI_SNR3[o], snr_percentile)
            RED_SNR[o,4] = np.percentile(RED_SKY_SNR[o], snr_percentile)
            RED_SNR[o,5] = np.percentile(RED_SCI_SNR[o], snr_percentile)

        # Save SNR arrays to the object
        self.GREEN_SNR     = GREEN_SNR
        self.RED_SNR       = RED_SNR
        self.GREEN_SNR_WAV = GREEN_SNR_WAV
        self.RED_SNR_WAV   = RED_SNR_WAV


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
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10,14), tight_layout=True)

        # Plot the data on each subplot
        ax1.scatter(self.GREEN_SNR_WAV, self.GREEN_SNR[:,5], marker="8", color='darkgreen', label='SCI1+SCI2+SCI3')
        ax1.scatter(self.GREEN_SNR_WAV, self.GREEN_SNR[:,1], marker=">", color='darkgreen', label='SCI1')
        ax1.scatter(self.GREEN_SNR_WAV, self.GREEN_SNR[:,2], marker="s", color='darkgreen', label='SCI2')
        ax1.scatter(self.GREEN_SNR_WAV, self.GREEN_SNR[:,3], marker="<", color='darkgreen', label='SCI3')
        ax1.scatter(self.RED_SNR_WAV,   self.RED_SNR[:,5],   marker="8", color='r', label='SCI1+SCI2+SCI3')
        ax1.scatter(self.RED_SNR_WAV,   self.RED_SNR[:,1],   marker=">", color='r', label='SCI1')
        ax1.scatter(self.RED_SNR_WAV,   self.RED_SNR[:,2],   marker="s", color='r', label='SCI2')
        ax1.scatter(self.RED_SNR_WAV,   self.RED_SNR[:,3],   marker="<", color='r', label='SCI3')
        ax1.grid()
        ax2.scatter(self.GREEN_SNR_WAV, self.GREEN_SNR[:,4], marker="D", color='darkgreen', label='SKY')
        ax2.scatter(self.RED_SNR_WAV,   self.RED_SNR[:,4],   marker="D", color='r', label='SKY')
        ax2.grid()
        ax3.scatter(self.GREEN_SNR_WAV, self.GREEN_SNR[:,0], marker="D", color='darkgreen', label='CAL')
        ax3.scatter(self.RED_SNR_WAV,   self.RED_SNR[:,0],   marker="D", color='r', label='CAL')
        ax3.grid()
        ax3.set_xlim(4450,8700)

        # Add legend
        ax1.legend(["SCI1+SCI2+SCI3","SCI1","SCI2","SCI3"], ncol=4)

        # Set titles and labels for each subplot
        ax1.set_title(self.ObsID + ' - ' + self.name + ': ' + r'$\mathrm{SNR}_{'+str(self.snr_percentile)+'}$ = '+str(self.snr_percentile)+'th percentile (Signal / $\sqrt{\mathrm{Variance}}$)', fontsize=16)
        ax3.set_xlabel('Wavelength (Ang)', fontsize=14)
        ax1.set_ylabel('SNR - SCI', fontsize=14)
        ax2.set_ylabel('SNR - SKY', fontsize=14)
        ax3.set_ylabel('SNR - CAL', fontsize=14)
        ax3.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        ax2.yaxis.set_tick_params(labelsize=14)
        ax3.yaxis.set_tick_params(labelsize=14)

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0)
        plt.tight_layout()

        # Display the plot
        if fig_path != None:
            plt.savefig(fig_path, dpi=300, facecolor='w')
        if show_plot == True:
            plt.show()
        plt.close()

    def plot_L1_spectrum(self, orderlet=None, fig_path=None, show_plot=False):
        """
        Generate a rainbow-colored plot L1 spectrum.  One must select an orderlet.

        Args:
            orderlet (string) - "CAL", "SCI1", "SCI2", "SCI3", "SKY"
            fig_path (string) - set to the path for the file
                to be generated.
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
        ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        ax.set_title('L1 Spectrum of ' + orderlet.upper() + ': ' + str(self.ObsID) + ' - ' + self.name, fontsize=28)
        plt.tight_layout()

        # Display the plot
        if fig_path != None:
            plt.savefig(fig_path, dpi=288, facecolor='w')
        if show_plot == True:
            plt.show()
        plt.close()
        

    def plot_1D_spectrum_single_order(self, chip=None, order=11, ylog=False, 
                                            orderlet=['SCI1', 'SCI2', 'SCI3'], 
                                            fig_path=None, show_plot=False):

        """

        Generate a plot of a single order of the L1 spectrum showing all orderlets.

        Args:
            chip (string) - "green" or "red"
            order (int) - spectral order to plot
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
            plt.savefig(fig_path, dpi=400, facecolor='w')
        if show_plot == True:
            plt.show()
        plt.close()

    def measure_orderlet_flux_ratio(self):

        """
        TO-DO: MOVE THE ANALYSIS CODE FROM THE QLP HERE.
               THIS IS A PLACEHOLDER.

        Compute the flux ratios of SCI2/SCI1, SCI3/SCI1, CAL/SCI1, SKY/SCI1.

        Args:
            None

        Attributes:
            To be defined.

        Returns:
            None
        """
