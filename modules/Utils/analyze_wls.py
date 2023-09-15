import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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


        if chip == 'green':
            wav1 = self.L1.GREEN_SCI_WAVE1
            wav2 = self.L1.GREEN_SCI_WAVE2
            wav3 = self.L1.GREEN_SCI_WAVE3
            cal  = self.L1.GREEN_CAL_WAVE
            sky  = self.L1.GREEN_SKY_WAVE
        elif chip == 'red':
            wav1 = self.L1.RED_SCI_WAVE1
            wav2 = self.L1.RED_SCI_WAVE2
            wav3 = self.L1.RED_SCI_WAVE3
            cal  = self.L1.RED_CAL_WAVE
            sky  = self.L1.RED_SKY_WAVE
            
        
        wls = [wav1, wav3, cal, sky]
        labels = ['WLS1', 'WL3', 'CAL', 'SKY']
        num_orders = len(wav2)
        
        fig, ax = plt.subplots(4, 1, sharex='col', sharey='row', figsize=(18, 12))
        
        for i, w in enumerate(wls):
            # plot the data
            for order in range(num_orders):
                ax[i].plot(wav2[order,:], w[order,:]-wav2[order,:], c=cm.gist_rainbow(0.9*(1-order/num_orders)), lw=3)
                
            # make the axes pretty
            y02 = np.percentile(w-wav2, 2.5)
            y50 = np.percentile(w-wav2, 50)
            y98 = np.percentile(w-wav2, 97.5)
            dy = 0.5*((y98-y50)+(y50-y02))
    
            ax[i].tick_params(axis='both', which='major', labelsize=14)
            ax[i].set_xlim(wav2.min()-25,wav2.max()+25)
            ax[i].set_ylim(y50-dy, y50+dy)
            ax[i].set_ylabel('{0}-WLS2'.format(labels[i]), fontsize=18)
            
        title = "{0} Chip:  {1}".format(CHIP, self.L1.header['PRIMARY']['OFNAME'])
        ax[0].set_title(title, fontsize=22)
        plt.xlabel('Wavelength (Ang)', fontsize=18)
    
        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=300, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')
