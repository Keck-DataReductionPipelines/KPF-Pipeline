import numpy as np
import matplotlib.pyplot as plt
from modules.Utils.kpf_parse import HeaderParse
from astropy.table import Table

class AnalyzeL2:

    """
    Description:
        This class contains functions to analyze L2 spectra (storing them
        as attributes) and functions to plot the results.

    Arguments:
        L2 - an L2 object

    Attributes:
        TBD
    """

    def __init__(self, L2, logger=None):
        if logger:
            self.logger = logger
            self.logger.debug('Initializing AnalyzeL2 object')
        else:
            self.logger = None
        self.L2 = L2
        #self.header = L2['PRIMARY'].header
        primary_header = HeaderParse(L2, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        self.ObsID = primary_header.get_obsid()
        self.rv_header = HeaderParse(L2, 'RV').header
        self.df_RVs = self.L2['RV'] # Table of RVs per order and orderlet

    def plot_CCF_grid(self, chip=None, annotate=False, 
                      zoom=False, fig_path=None, show_plot=False):
        """

        Generate a plot of SNR per order as compuated using the compute_l1_snr
        function.

        Args:
            chip (string) - "green" or "red"
            annotate (boolean) - show text annotations, especially on SCI 
            fig_path (string) - set to the path for a SNR vs. wavelength file
                to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it the current environment
            (e.g., in a Jupyter Notebook).

        """

        # Define variables
        if chip.lower() == 'green':
            self.chip = chip.lower()
            chip = 'GREEN_CCF'
            CCD = 'CCD1'
            chip_title = 'Green'
            offset = 0
        elif chip.lower() == 'red':
            self.chip = chip.lower()
            chip = 'RED_CCF'
            CCD = 'CCD2'
            chip_title = 'Red'
            offset = self.L2['GREEN_CCF'].data.shape[1]
        else:
            self.logger.error("Need to specify 'chip' in plot_CCF_grid.")
        CCF_header =  HeaderParse(self.L2, chip).header
        RV_start = CCF_header['STARTV']
        nsteps   = CCF_header['TOTALV']
        delta_RV = CCF_header['STEPV']
        RVgrid = np.arange(RV_start, RV_start + nsteps*delta_RV, delta_RV)
        CCF_data = np.array(self.L2[chip].data)
        n_orders = self.L2[chip].data.shape[1]

        # Measure the flux in each CCF (need to update this -- see below)
        #CCF_flux_array = np.zeros((5, n_orders))
        #for oo in np.arange(5):
        #    for o in np.arange(n_orders):
        #        CCF_flux_array[oo,o] = np.nansum(CCF_data[oo,o,:])
        
        #CCF2 = L2[chip].data[:,:,:]
        #mean_CCF = CCF2/np.percentile(np.average(CCF2),[99.9])

        
        # Set up plot
        fig, axes = plt.subplots(1, 5, figsize=(25, 15), tight_layout=True)
        
        # Iterate over orderlets
        for oo, orderlet in enumerate(['SCI1', 'SCI2', 'SCI3', 'CAL', 'SKY']):
            ax = axes[oo]
            
            # Plot vertical lines for RVs and add top annotations
            if   oo == 0: 
                this_RV = self.rv_header['CCD1RV1']
                this_RV_text = f"{this_RV:.5f}" + r' km s$^{-1}$'
            elif oo == 1: 
                this_RV = self.rv_header['CCD1RV2']
                this_RV_text = f"{this_RV:.5f}" + r' km s$^{-1}$'
            elif oo == 2: 
                this_RV = self.rv_header['CCD1RV3']
                this_RV_text = f"{this_RV:.5f}" + r' km s$^{-1}$'
            elif oo == 3: 
                this_RV = self.rv_header['CCD1RVC'] 
            elif oo == 4: 
                this_RV = self.rv_header['CCD1RVS'] 
            if oo < 3: 
                # Annotation and line for orderlet-averaged RV
                ax.plot([this_RV, this_RV], [0, n_orders*0.5+0.5], color='k')
                ax.text(this_RV, n_orders*0.5+0.7, this_RV_text, color='k', horizontalalignment='center', fontsize=11)
                # Annotation for Delta RV
                ax.text(RVgrid[2],    n_orders*0.5+0.7,r'$\Delta$RV (this - avg)', 
                                verticalalignment='center', horizontalalignment='left', color='k', fontsize=11)
                #Annotation for weight
                ax.text(RVgrid[-1]-3, n_orders*0.5+0.7, 'weight', 
                                verticalalignment='center', horizontalalignment='right', color='k', fontsize=11)
                # need to update this
                #ax.text(RVgrid[-1]-30,n_orders*0.5+0.7, 'flux', 
                #                verticalalignment='center', horizontalalignment='right', color='k', fontsize=11)

            ax.set_ylim(0, n_orders*0.5+1)
            #if zoom:
            #    ax.set_xlim(middle_left, middle_right)

            # Iterate over orders
            for o in range(n_orders):
                if orderlet == 'CAL':
                    this_CCF = CCF_data[oo, o, :]
                    if np.nanpercentile(this_CCF,99) < 0:
                        norm_CCF = np.divide(this_CCF+np.nanpercentile(this_CCF,0.1), 
                                             np.nanpercentile(this_CCF+np.nanpercentile(this_CCF,0.1),90))
                    else:
                        if np.nanpercentile(this_CCF,[90]) == 0:
                            norm_CCF = this_CCF
                        else:
                            norm_CCF = np.divide(this_CCF, np.nanpercentile(this_CCF,[90]))
                elif orderlet == 'SKY':
                    this_CCF = CCF_data[oo, o, :]
                    if np.nanpercentile(this_CCF,99) < 0:
                        norm_CCF = np.divide(this_CCF+np.nanpercentile(this_CCF,0.1), 
                                             np.nanpercentile(this_CCF+np.nanpercentile(this_CCF,0.1),90))
                    else:
                        if np.nanpercentile(this_CCF,[90]) == 0:
                            norm_CCF = this_CCF
                        else:
                            norm_CCF = np.divide(this_CCF, np.nanpercentile(this_CCF,[90]))
                elif (np.sum(CCF_data[oo, o, :]) != 0): # SCI1/SCI2/SCI3 - only show if CCF was computed
                    this_CCF = CCF_data[oo, o, :]
                    norm_CCF = np.divide(this_CCF, np.nanpercentile(this_CCF,[99]))
                # The zoom feature is not yet implemented
                #if zoom:
                #    middle = len(RVgrid) // 4
                #    middle_left  = min(RVgrid[quarter: -quarter])
                #    middle_right = max(RVgrid[quarter: -quarter])
                #    norm_CCF = norm_CCF[middle_left:middle_right]
                #    current_RVgrid = xx

                # Plot CCF for this order and orderlet
                ax.plot([min(RVgrid)], [0])  # make a dot in the corner to insure that the colors are different on each row
                current_color = ax.lines[-1].get_color()
                if np.sum(CCF_data[oo, o, :]) != 0:
                    ax.plot(RVgrid, norm_CCF + o*0.5, color=current_color)
                
                # Add additional annotations
                ax.text(RVgrid[-1]+0.75,  1+o*0.5-0.10, str(o), color=current_color, verticalalignment='center', horizontalalignment='left', fontsize=11)
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                if   oo == 0: rv = self.df_RVs['orderlet1'][offset+o] - this_RV
                elif oo == 1: rv = self.df_RVs['orderlet2'][offset+o] - this_RV
                elif oo == 2: rv = self.df_RVs['orderlet3'][offset+o] - this_RV
                if (oo < 3) and (np.sum(CCF_data[oo, o, :]) != 0): 
                    # RV annotation
                    ax.text(RVgrid[2], 1+o*0.5-0.3, f"{rv:.4f}" + r' km s$^{-1}$', 
                            color=current_color, verticalalignment='center', fontsize=11)
                    # CCF weight annotation
                    ax.text(RVgrid[-1]-3,  1+o*0.5-0.3, f"{self.df_RVs['CCF Weights'][o]:.2f}", 
                            color=current_color, verticalalignment='center', horizontalalignment='right', fontsize=11)
                    # Flux annotation (fix)
#                    ax.text(RVgrid[-1]-30, 1+o*0.5-0.3, f"{CCF_flux_array[o]:.2f}", 
#                            color=current_color, verticalalignment='center', horizontalalignment='right', fontsize=11)
                
            # Determine mask per order and include in title
            if (oo == 0) or (oo == 1) or (oo == 2): 
                this_mask = ' (' + CCF_header['SCI_MASK'] + ' mask)'
            if  oo == 3:
                this_mask = ' (' + CCF_header['CAL_MASK'] + ' mask)'
            if  oo == 4:
                this_mask = ' (' + CCF_header['SKY_MASK'] + ' mask)'
            ax.grid(False)
            ax.set_title(orderlet + ' CCF' + this_mask, fontsize=18)
            ax.set_xlabel('RV (km/s)', fontsize=18)

        # Add overall title to array of plots
        ax = fig.add_subplot(111, frame_on=False)
        ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        ax.set_title('2D - ' + chip_title + ' CCD: ' + str(self.ObsID) + ' - ' + self.name + '\n', fontsize=30)

        # Display the plot
        if fig_path != None:
            plt.savefig(fig_path, dpi=300, facecolor='w')
        if show_plot == True:
            plt.show()
        plt.close('all')
