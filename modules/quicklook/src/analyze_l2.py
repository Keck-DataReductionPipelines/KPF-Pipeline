import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from modules.Utils.kpf_parse import HeaderParse, get_data_products_L2
from astropy.table import Table

class AnalyzeL2:

    """
    Description:
        This class contains functions to analyze L2 spectra (storing them
        as attributes) and functions to plot the results.

    Arguments:
        L2 - an L2 object

    Attributes:
        name - name of source (e.g., 'Bias', 'Etalon', '185144')
        ObsID - observation  ID (e.g. 'KP.20230704.02326.27')
        header - header of the PRIMARY extension of the L2 object
        rv_header - header of the RV extension
    
    To do:
        Add plot showing combined CCF - https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/issues/940
        Add plot showing correlations between per-order RVs and per-chip RVs and overall RVs.
    """

    def __init__(self, L2, logger=None):
        if logger:
            self.logger = logger
            self.logger.debug('Initializing AnalyzeL2 object')
        else:
            self.logger = None
        self.L2 = L2
        self.df_RV = self.L2['RV']
        self.n_green_orders = 35
        self.n_red_orders   = 32
        primary_header = HeaderParse(L2, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        self.ObsID = primary_header.get_obsid()
        self.rv_header = HeaderParse(L2, 'RV').header
        self.df_RVs = self.L2['RV'] # Table of RVs per order and orderlet
        self.data_products = get_data_products_L2(self.L2)
        self.green_present = 'Green' in self.data_products
        self.red_present = 'Red' in self.data_products
        self.texp = self.header['ELAPSED']

        self.compute_statistics()
        
        
    def compute_statistics(self):
        """
        Compute various metrics of dispersion of the per-order BJD values
        """
        # compute weighted Barycentric RV correction
        x = self.df_RV['Bary_RVC']
        w = self.df_RV['CCF Weights']
        self.CCFBCV = np.sum(w * x) / np.sum(w)

        # compute weighted BJD (this should be computed elsewhere and read from the L2 header)
        x = self.df_RV['CCFBJD']
        w = self.df_RV['CCF Weights']
        self.CCFBJD = np.sum(w * x) / np.sum(w)

        # compute per-order BJD differences
        self.df_RV['Delta_CCFBJD'] = self.df_RV['CCFBJD'].copy()
        self.df_RV['Delta_CCFBJD'] -= self.CCFBJD
        #    compute weighted standard deviation
        x = self.df_RV['Delta_CCFBJD']
        w = self.df_RV['CCF Weights']
        nonzero_mask = w != 0
        wmean = np.sum(w * x) / np.sum(w)
        var_pop = np.sum(w * (x - wmean)**2) / np.sum(w) # weighted variance
        self.Delta_CCFBJD_weighted_std = np.sqrt(var_pop) * 24*60*60  # seconds
        self.Delta_CCFBJD_weighted_range = (x[nonzero_mask].max() - x[nonzero_mask].min()) * 24*60*60  # seconds

        # compute per-order Barycentric RV differences
        self.df_RV['Delta_Bary_RVC'] = self.df_RV['Bary_RVC'].copy()
        self.df_RV['Delta_Bary_RVC'] -= self.CCFBCV
        #    compute weighted standard deviation
        x = self.df_RV['Delta_Bary_RVC']
        wmean = np.sum(w * x) / np.sum(w)
        var_pop = np.sum(w * (x - wmean)**2) / np.sum(w) # weighted variance
        self.Delta_Bary_RVC_weighted_std = np.sqrt(var_pop) * 1000 # m/s
        self.Delta_Bary_RVC_weighted_range = (x[nonzero_mask].max() - x[nonzero_mask].min()) * 1000 # m/s



    def plot_CCF_grid(self, chip=None, annotate=False, 
                      zoom=False, fig_path=None, show_plot=False):
        """

        Generate a plot of CCFs for each order and orderlet.

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
        
        # Set up plot
        fig, axes = plt.subplots(1, 5, figsize=(25, 15), tight_layout=True)
        
        # Iterate over orderlets
        for oo, orderlet in enumerate(['SCI1', 'SCI2', 'SCI3', 'CAL', 'SKY']):
            ax = axes[oo]
            
            # Plot vertical lines for RVs and add top annotations
            if   oo == 0:
                try:
                    this_RV = self.rv_header['CCD1RV1']
                    this_RV_text = f"{this_RV:.5f}" + r' km s$^{-1}$'
                    data_present = True
                except:
                    this_RV = 0.
                    this_RV_text = 'No RV reported - CCD1RV1 missing from header'
                    data_present = False
            elif oo == 1: 
                try:
                    this_RV = self.rv_header['CCD1RV2']
                    this_RV_text = f"{this_RV:.5f}" + r' km s$^{-1}$'
                    data_present = True
                except:
                    this_RV = 0.
                    this_RV_text = 'No RV reported - CCD1RV1 missing from header'
                    data_present = False
            elif oo == 2: 
                try:
                    this_RV = self.rv_header['CCD1RV3']
                    this_RV_text = f"{this_RV:.5f}" + r' km s$^{-1}$'
                    data_present = True
                except:
                    this_RV = 0.
                    this_RV_text = 'No RV reported - CCD1RV1 missing from header'
                    data_present = False
            elif oo == 3: 
                try:
                    this_RV = self.rv_header['CCD1RVC'] 
                    data_present = True
                except:
                    this_RV = 0
                    data_present = False
            elif oo == 4: 
                try:
                    this_RV = self.rv_header['CCD1RVS'] 
                    data_present = True
                except:
                    this_RV = 0.
                    data_present = False
            if oo < 3: 
                # Annotation and line for orderlet-averaged RV
                ax.plot([this_RV, this_RV], [0, n_orders*0.5+0.5], color='k')
                ax.text(this_RV, n_orders*0.5+0.7, this_RV_text, color='k', horizontalalignment='center', fontsize=11)
                # Annotation for Delta RV
                ax.text(RVgrid[2],    n_orders*0.5+0.55,r'$\Delta$RV (this - avg)', 
                                verticalalignment='center', horizontalalignment='left', color='k', fontsize=11)
                #Annotation for weight
                ax.text(RVgrid[-1]-3, n_orders*0.5+0.55, 'weight', 
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
                    if data_present:
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
                    if data_present:
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
                    if data_present:
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
                if data_present:
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
                    ax.text(RVgrid[-1]-3,  1+o*0.5-0.3, f"{self.df_RVs['CCF Weights'][offset+o]:.2f}", 
                            color=current_color, verticalalignment='center', horizontalalignment='right', fontsize=11)
                    # Flux annotation (fix)
#                    ax.text(RVgrid[-1]-30, 1+o*0.5-0.3, f"{CCF_flux_array[o]:.2f}", 
#                            color=current_color, verticalalignment='center', horizontalalignment='right', fontsize=11)
                
            # Determine mask per order and include in title
            if data_present:
                if (oo == 0) or (oo == 1) or (oo == 2): 
                    this_mask = ' (' + CCF_header['SCI_MASK'] + ' mask)'
                if  oo == 3:
                    this_mask = ' (' + CCF_header['CAL_MASK'] + ' mask)'
                if  oo == 4:
                    this_mask = ' (' + CCF_header['SKY_MASK'] + ' mask)'
                ax.set_title(orderlet + ' CCF' + this_mask, fontsize=18)
            ax.set_xlabel('RV (km/s)', fontsize=18)
            ax.grid(False)

        # Add overall title to array of plots
        ax = fig.add_subplot(111, frame_on=False)
        ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        ax.set_title('L2 - ' + chip_title + ' CCD: ' + str(self.ObsID) + ' - ' + self.name + '\n', fontsize=30)
            
        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    xytext=(0, -50), textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)     

        # Display the plot
        if fig_path != None:
            plt.savefig(fig_path, dpi=300, facecolor='w')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def plot_BJD_BCV_grid(self, fig_path=None, show_plot=False):
        """

        Generate a plot of BJD and Barycentric RV vs. spectral order.

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
        
        # Set up plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), tight_layout=True)
        
        # Iterate over panels
        for p, panel in enumerate(['BJD', 'BCRV']):
            ax = axes[p]
            ax.grid(True)
            ax.xaxis.set_tick_params(labelsize=14)
            ax.yaxis.set_tick_params(labelsize=14)
            ax.axhline(0, color='black', lw=2, zorder=0)
            ax.set_xlim(-1,self.n_green_orders+self.n_red_orders)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=7))

            # BJD panel
            if p == 0:
                for o in np.arange(self.n_green_orders):
                    if self.df_RV['CCF Weights'][o] != 0:
                        ax.scatter(o, self.df_RV['Delta_CCFBJD'][o]*24*60*60, s=45, c='darkgreen')
                    else:
                        ax.scatter(o, self.df_RV['Delta_CCFBJD'][o]*24*60*60, s=45, c='darkseagreen')
                for o in np.arange(self.n_green_orders, self.n_green_orders+self.n_red_orders):
                    if self.df_RV['CCF Weights'][o] != 0:
                        ax.scatter(o, self.df_RV['Delta_CCFBJD'][o]*24*60*60, s=45, c='darkred')
                    else:
                        ax.scatter(o, self.df_RV['Delta_CCFBJD'][o]*24*60*60, s=45, c='lightcoral')
                ax.set_ylabel(r'$\Delta$BJD' + r'$_\mathrm{pw}$' + ' (sec)' + '\n(order - average' + r'$_\mathrm{chip}$' + ')', fontsize=16)
                legend_handle = Line2D([], [], linestyle='none', label=r"$\sigma$ = " + f"{self.Delta_CCFBJD_weighted_std:.2g}" + ' sec\nrange = ' + f"{self.Delta_CCFBJD_weighted_range:.2g}" + ' sec')
                ax.legend(handles=[legend_handle], loc='upper right', fontsize=14)

            # Barycentric RV panel
            elif p == 1: 
                for o in np.arange(self.n_green_orders):
                    if self.df_RV['CCF Weights'][o] != 0:
                        ax.scatter(o, self.df_RV['Delta_Bary_RVC'][o]*1000, s=45, c='darkgreen')
                    else:
                        ax.scatter(o, self.df_RV['Delta_Bary_RVC'][o]*1000, s=45, c='darkseagreen')
                for o in np.arange(self.n_green_orders, self.n_green_orders+self.n_red_orders):
                    if self.df_RV['CCF Weights'][o] != 0:
                        ax.scatter(o, self.df_RV['Delta_Bary_RVC'][o]*1000, s=45, c='darkred')
                    else:
                        ax.scatter(o, self.df_RV['Delta_Bary_RVC'][o]*1000, s=45, c='lightcoral')
                ax.set_ylabel(r'$\Delta$ Barycentric RV (m s$^{-1}$)' + '\n(order - average' + r'$_\mathrm{chip}$' + ')', fontsize=16)
                ax.set_xlabel('Order Index', fontsize=16)
                legend_handle = Line2D([], [], linestyle='none', label=r"$\sigma$ = " + f"{self.Delta_Bary_RVC_weighted_std:.2g}" + ' m/s\nrange = ' + f"{self.Delta_Bary_RVC_weighted_range:.2g}" + ' m/s')
                ax.legend(handles=[legend_handle], loc='upper right', fontsize=14)
            
        # Add overall title to array of plots
        ax = fig.add_subplot(111, frame_on=False)
        ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        ax.set_title('Dispersion of Photon-weighted BJDs and Barycentric RVs\nL2: ' + str(self.ObsID) + ' - ' + self.name + r' (T$_\mathrm{exp}$ = ' + str(int(self.texp)) + ' sec)', fontsize=18)

        # Create a timestamp and annotate in the lower right corner
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp_label = f"KPF QLP: {current_time}"
        plt.annotate(timestamp_label, xy=(1, 0), xycoords='axes fraction', 
                    fontsize=8, color="darkgray", ha="right", va="bottom",
                    xytext=(0, -50), textcoords='offset points')
        plt.subplots_adjust(bottom=0.1)     

        # Display the plot
        if fig_path != None:
            plt.savefig(fig_path, dpi=300, facecolor='w')
        if show_plot == True:
            plt.show()
        plt.close('all')
