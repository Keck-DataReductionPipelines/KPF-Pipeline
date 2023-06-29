import numpy as np
import matplotlib.pyplot as plt

class AnalyzeL1:

    """
    Description:
        This class contains functions to analyze L1 spectra (storing them
        as attributes) and functions to plot the results.

    Arguments:
        L0 - an L0 object

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
        self.L1 = L1
        if logger:
            self.logger = logger
            self.logger.debug('AnalyzeL0 class constructor')
        else:
            self.logger = None
            print('---->AnalyzeL0 class constructor')

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

        self.snr_percentile = snr_percentile

        # Determine the number of orders
        norders_green = (L1.data['GREEN_SKY_WAVE']).shape[0]
        norders_red   = (L1.data['RED_SKY_WAVE']).shape[0]
        orderlets = {'CAL','SCI1','SCI2','SCI3','SKY'}
        norderlets = len(orderlets)

        # Define SNR arrays (needed for operations below where VAR = 0)
        GREEN_SCI_SNR1 = 0 * L1.data['GREEN_SCI_VAR1']
        GREEN_SCI_SNR2 = 0 * L1.data['GREEN_SCI_VAR2']
        GREEN_SCI_SNR3 = 0 * L1.data['GREEN_SCI_VAR3']
        GREEN_CAL_SNR  = 0 * L1.data['GREEN_CAL_VAR']
        GREEN_SKY_SNR  = 0 * L1.data['GREEN_SKY_VAR']
        GREEN_SCI_SNR  = 0 * L1.data['GREEN_SCI_VAR1']
        RED_SCI_SNR1   = 0 * L1.data['RED_SCI_VAR1']
        RED_SCI_SNR2   = 0 * L1.data['RED_SCI_VAR2']
        RED_SCI_SNR3   = 0 * L1.data['RED_SCI_VAR3']
        RED_CAL_SNR    = 0 * L1.data['RED_CAL_VAR']
        RED_SKY_SNR    = 0 * L1.data['RED_SKY_VAR']
        RED_SCI_SNR    = 0 * L1.data['RED_SCI_VAR1']

        # Create Arrays
        GREEN_SNR = np.zeros((norders_green, norderlets+1))
        RED_SNR   = np.zeros((norders_red, norderlets+1))
        GREEN_SNR_WAV = np.zeros(norders_green)
        RED_SNR_WAV   = np.zeros(norders_red)

        # Compute SNR arrays for each of the orders, orderlets, and CCDs.
        GREEN_SCI_SNR1 = np.divide(L1.data['GREEN_SCI_FLUX1'],
                                   np.sqrt(abs(L1.data['GREEN_SCI_VAR1'])),
                                   where=(L1.data['GREEN_SCI_VAR1']!=0))
        GREEN_SCI_SNR2 = np.divide(L1.data['GREEN_SCI_FLUX2'],
                                   np.sqrt(abs(L1.data['GREEN_SCI_VAR2'])),
                                   where=(L1.data['GREEN_SCI_VAR2']!=0))
        GREEN_SCI_SNR3 = np.divide(L1.data['GREEN_SCI_FLUX3'],
                                   np.sqrt(abs(L1.data['GREEN_SCI_VAR3'])),
                                   where=(L1.data['GREEN_SCI_VAR3']!=0))
        GREEN_SCI_SNR  = np.divide(L1.data['GREEN_SCI_FLUX1']+L1.data['GREEN_SCI_FLUX3']+L1.data['GREEN_SCI_FLUX3'],
                                   np.sqrt(abs(L1.data['GREEN_SCI_VAR1'])+abs(L1.data['GREEN_SCI_VAR2'])+abs(L1.data['GREEN_SCI_VAR3'])),
                                   where=(L1.data['GREEN_SCI_VAR1']+L1.data['GREEN_SCI_VAR2']+L1.data['GREEN_SCI_VAR3']!=0))
        GREEN_CAL_SNR  = np.divide(L1.data['GREEN_CAL_FLUX'],
                                   np.sqrt(abs(L1.data['GREEN_CAL_VAR'])),
                                   where=(L1.data['GREEN_CAL_VAR']!=0))
        GREEN_SKY_SNR  = np.divide(L1.data['GREEN_SKY_FLUX'],
                                   np.sqrt(abs(L1.data['GREEN_SKY_VAR'])),
                                   where=(L1.data['GREEN_SKY_VAR']!=0))
        RED_SCI_SNR1   = np.divide(L1.data['RED_SCI_FLUX1'],
                                   np.sqrt(abs(L1.data['RED_SCI_VAR1'])),
                                   where=(L1.data['RED_SCI_VAR1']!=0))
        RED_SCI_SNR2   = np.divide(L1.data['RED_SCI_FLUX2'],
                                   np.sqrt(abs(L1.data['RED_SCI_VAR2'])),
                                   where=(L1.data['RED_SCI_VAR2']!=0))
        RED_SCI_SNR3   = np.divide(L1.data['RED_SCI_FLUX3'],
                                   np.sqrt(abs(L1.data['RED_SCI_VAR3'])),
                                   where=(L1.data['RED_SCI_VAR3']!=0))
        RED_SCI_SNR    = np.divide(L1.data['RED_SCI_FLUX1']+L1.data['RED_SCI_FLUX3']+L1.data['RED_SCI_FLUX3'],
                                   np.sqrt(abs(L1.data['RED_SCI_VAR1'])+abs(L1.data['RED_SCI_VAR2'])+abs(L1.data['RED_SCI_VAR3'])),
                                   where=(L1.data['RED_SCI_VAR1']+L1.data['RED_SCI_VAR2']+L1.data['RED_SCI_VAR3']!=0))
        RED_CAL_SNR    = np.divide(L1.data['RED_CAL_FLUX'],
                                   np.sqrt(abs(L1.data['RED_CAL_VAR'])),
                                   where=(L1.data['RED_CAL_VAR']!=0))
        RED_SKY_SNR    = np.divide(L1.data['RED_SKY_FLUX'],
                                   np.sqrt(abs(L1.data['RED_SKY_VAR'])),
                                   where=(L1.data['RED_SKY_VAR']!=0))

        # Compute SNR per order and per orderlet
        for o in range(norders_green):
            GREEN_SNR_WAV[o] = L1.data['GREEN_SCI_WAVE1'][o,2040]
            GREEN_SNR[o,0] = np.percentile(GREEN_CAL_SNR[o], snr_percentile)
            GREEN_SNR[o,1] = np.percentile(GREEN_SCI_SNR1[o], snr_percentile)
            GREEN_SNR[o,2] = np.percentile(GREEN_SCI_SNR2[o], snr_percentile)
            GREEN_SNR[o,3] = np.percentile(GREEN_SCI_SNR3[o], snr_percentile)
            GREEN_SNR[o,4] = np.percentile(GREEN_SKY_SNR[o], snr_percentile)
            GREEN_SNR[o,5] = np.percentile(GREEN_SCI_SNR[o], snr_percentile)
        for o in range(norders_red):
            RED_SNR_WAV[o] = L1.data['RED_SCI_WAVE1'][o,2040]
            RED_SNR[o,0] = np.percentile(RED_CAL_SNR[o], snr_percentile)
            RED_SNR[o,1] = np.percentile(RED_SCI_SNR1[o], snr_percentile)
            RED_SNR[o,2] = np.percentile(RED_SCI_SNR2[o], snr_percentile)
            RED_SNR[o,3] = np.percentile(RED_SCI_SNR3[o], snr_percentile)
            RED_SNR[o,4] = np.percentile(RED_SKY_SNR[o], snr_percentile)
            RED_SNR[o,5] = np.percentile(RED_SCI_SNR[o], snr_percentile)

        # Save SNR arrays to the object
        self.GREEN_SNR = GREEN_SNR
        self.RED_SNR   = RED_SNR
        self.GREEN_SNR_WAV = GREEN_SNR_WAV
        self.RED_SNR_WAV   = RED_SNR_WAV

    def plot_L1_snr(self,ObsID, fig_path=None, show_plot=False):

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
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10,14))

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
        ax1.set_title(ObsID + ' - ' + r'$\mathrm{SNR}_{'+str(self.snr_percentile)+'}$ = '+str(self.snr_percentile)+'th percentile (Signal / $\sqrt{\mathrm{Variance}}$)', fontsize=18)
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
