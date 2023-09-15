import time
import numpy as np
import matplotlib.pyplot as plt
from modules.Utils.kpf_parse import HeaderParse
from datetime import datetime

class AnalyzeWLS:

    """
    Description:
        This class contains functions to analyze wavelength solutions 
        (storing them as attributes) and functions to plot the results.

    Arguments:
        L1 - an L1 object
        L1b (optiona) - a second L1 object to compare to L1

    Attributes:
        None so far
    """

    def __init__(self, L1, logger=None):
        if logger:
            self.logger = logger
            self.logger.debug('Initializing AnalyzeWLS object')
        else:
            self.logger = None
        self.L1 = L1
        #self.header = L1['PRIMARY'].header
        primary_header = HeaderParse(L1, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        
        #self.ObsID = primary_header.get_obsid()
        # need a filename instead or in addition


#    def __init__(self, L1, logger=None):
#        if self.logger:
#            self.logger = logger
#            self.logger.debug('Initializing AnalyzeWLS object.')
#        else:
#            self.logger = None
#        self.L1 = L1
#        #self.L1b = L1b
#        primary_header = HeaderParse(L1, 'PRIMARY')
#        self.header = primary_header.header
#        self.name = primary_header.get_name()
#        
#        #self.ObsID = primary_header.get_obsid()
#        # need a filename instead or in addition


    def plot_WLS_orderlet_diff(self, chip=None, fig_path=None, show_plot=False):
        """
        Make a plot of differences between the wavelength solutions for the orders.

        Args:
            chip (string) - "green" or "red"
            fig_path (string) - set to the path for a SNR vs. wavelength file
                to be generated.
            show_plot (boolean) - show the plot in the current environment 

        Returns:
            PNG plot in fig_path or shows the plot it the current environment
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
            self.logger.debug('chip not supplied.  Exiting plot_WLS_orderlet_diff')
            print('chip not supplied.  Exiting plot_WLS_orderlet_diff')
            return

# placeholder line
        fig = plt.plot(np.arange(10))

# Substitute good plotting stuff here.
#        # Make 3-panel plot. First, create the figure and subplots
#        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10,10), tight_layout=True)
#
#        # Plot the data on each subplot
#        ax1.scatter(self.GREEN_SNR_WAV, self.GREEN_SNR[:,5], marker="8", color='darkgreen', label='SCI1+SCI2+SCI3')
#        ax1.scatter(self.GREEN_SNR_WAV, self.GREEN_SNR[:,1], marker=">", color='darkgreen', label='SCI1')
#        ax1.scatter(self.GREEN_SNR_WAV, self.GREEN_SNR[:,2], marker="s", color='darkgreen', label='SCI2')
#        ax1.scatter(self.GREEN_SNR_WAV, self.GREEN_SNR[:,3], marker="<", color='darkgreen', label='SCI3')
#        ax1.scatter(self.RED_SNR_WAV,   self.RED_SNR[:,5],   marker="8", color='r', label='SCI1+SCI2+SCI3')
#        ax1.scatter(self.RED_SNR_WAV,   self.RED_SNR[:,1],   marker=">", color='r', label='SCI1')
#        ax1.scatter(self.RED_SNR_WAV,   self.RED_SNR[:,2],   marker="s", color='r', label='SCI2')
#        ax1.scatter(self.RED_SNR_WAV,   self.RED_SNR[:,3],   marker="<", color='r', label='SCI3')
#        ax1.yaxis.set_major_locator(MaxNLocator(nbins=12))
#        ax1.grid()
#        ax2.scatter(self.GREEN_SNR_WAV, self.GREEN_SNR[:,4], marker="D", color='darkgreen', label='SKY')
#        ax2.scatter(self.RED_SNR_WAV,   self.RED_SNR[:,4],   marker="D", color='r', label='SKY')
#        ax2.yaxis.set_major_locator(MaxNLocator(nbins=12))
#        ax2.grid()
#        ax3.scatter(self.GREEN_SNR_WAV, self.GREEN_SNR[:,0], marker="D", color='darkgreen', label='CAL')
#        ax3.scatter(self.RED_SNR_WAV,   self.RED_SNR[:,0],   marker="D", color='r', label='CAL')
#        ax3.yaxis.set_major_locator(MaxNLocator(nbins=12))
#        ax3.grid()
#        ax3.set_xlim(4450,8700)
#
#        # Add legend
#        ax1.legend(["SCI1+SCI2+SCI3","SCI1","SCI2","SCI3"], ncol=4)
#
#        # Set titles and labels for each subplot
#        ax1.set_title(self.ObsID + ' - ' + self.name + ': ' + r'$\mathrm{SNR}_{'+str(self.snr_percentile)+'}$ = '+str(self.snr_percentile)+'th percentile (Signal / $\sqrt{\mathrm{Variance}}$)', fontsize=16)
#        ax3.set_xlabel('Wavelength (Ang)', fontsize=14)
#        ax1.set_ylabel('SNR - SCI', fontsize=14)
#        ax2.set_ylabel('SNR - SKY', fontsize=14)
#        ax3.set_ylabel('SNR - CAL', fontsize=14)
#        ax3.xaxis.set_tick_params(labelsize=14)
#        ax1.yaxis.set_tick_params(labelsize=14)
#        ax2.yaxis.set_tick_params(labelsize=14)
#        ax3.yaxis.set_tick_params(labelsize=14)
#
#        # Adjust spacing between subplots
#        plt.subplots_adjust(hspace=0)
#        plt.tight_layout()

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=300, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')
