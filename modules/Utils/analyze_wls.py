import time
import json
import gzip
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as transforms
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import SymmetricalLogLocator, ScalarFormatter, MultipleLocator
from matplotlib.patches import Rectangle
from datetime import datetime
from modules.Utils.kpf_parse import HeaderParse
from modules.Utils.utils import DummyLogger

class AnalyzeWLS:

    """
    Description:
        This class contains functions to analyze wavelength solutions 
        (storing them as attributes) and functions to plot the results.

    Arguments:
        L1 - an L1 object
        L1b (optional) - a second L1 object to compare to L1

    Attributes:
        None so far
    """

    def __init__(self, L1, logger=None):
        if logger:
            self.logger = logger
        else:
            self.logger = None
        self.L1 = L1
        #self.header = L1['PRIMARY'].header
        primary_header = HeaderParse(L1, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        
        #self.ObsID = primary_header.get_obsid()
        # need a filename instead or in addition


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
            self.logger.debug('"chip" not supplied.  Exiting plot_WLS_orderlet_diff')
            print('"chip" not supplied.  Exiting plot_WLS_orderlet_diff')
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
        labels = ['SCI1', 'SCI3', 'CAL', 'SKY']
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
            ax[i].set_ylabel('{0} - SCI2'.format(labels[i]), fontsize=18)
            ax[i].grid(True)
            
        title = "{0} Chip:  {1}".format(chip_title, self.L1.header['PRIMARY']['OFNAME'])
        ax[0].set_title(title, fontsize=22)
        ax[0].axhline(y=0, color='k', linestyle='-')        
        ax[1].axhline(y=0, color='k', linestyle='-')        
        ax[2].axhline(y=0, color='k', linestyle='-')        
        ax[3].axhline(y=0, color='k', linestyle='-')        
        plt.xlabel('Wavelength (Ang)', fontsize=18)
        plt.tight_layout()

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=300, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


class AnalyzeWLSDict:

    """
    Description:
        This class contains functions to analyze aspects of wavelength solutions 
        stored as diectionaries.

    Arguments:
        WLSDict - a WLS Dictionary, stored either the filename of a 
                  (possibly) zip-compressed JSON file, or a TBD object

    Attributes:
        wls_dict - dictionary of wavelength solution
        
    To do:
    	wave vs. pixel per order
    	heatmap or 1D plot for a parameter (like mu_diff)
    	# lines per order/orderlet
    	chi^2 distribution (perhaps by order/orderlet)
    	develop cuts: amp > 0, sig > 1
    """

    def __init__(self, WLSDict_filename, logger=None):
        self.logger = logger if logger is not None else DummyLogger()        
        self.wls_dict = read_wls_json(WLSDict_filename) 
        try:
            self.chip = self.wls_dict['chip']
        except:
            self.chip = '<chip>'


    def plot_WLS_line(self, orderlet, order, line, fig_path=None, show_plot=False):
        """
        Generate a plot of a single spectral line.

        Args:
            orderlet (string) - 'SCI1', 'SCI2', 'SCI3', 'CAL', or 'SKY'
            order (integer) - order number
            line (integer) - line number
            fig_path (string) - set to the path for the file to be generated.
                                default=None
            show_plot (boolean) - show the plot in the current environment.
                                  default=False

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).
        """

        linedict = self.wls_dict['orderlets'][orderlet]['orders'][order]['lines'][line]
        data  = linedict['data']
        model = linedict['model']
        err = np.sqrt(np.abs(data))
        resid = model - data
        npix = len(data)
        pix = np.arange(npix)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})  
        ax1.set_title(self.chip + ' ' + orderlet + ' - order ' + str(order) + ', line ' + str(line) + r' ($\chi^2$ = ' + f'{linedict["chi2"]:.3g}'+ ')', fontsize=14)
        ax1.step(pix, data, where='mid', c='b', label='Data')
        ax1.step(pix, model, where='mid', c='r', label='Model')
        ax1.errorbar(pix, data, yerr=err, c='b', fmt='none')
        ax1.errorbar(pix, model, yerr=err, c='r', fmt='none')
        ax1.set_ylabel('Intensity [e-]', fontsize=14)
        ax1.tick_params(axis='both', labelsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True)
        ax2.set_xticks(np.arange(0, max(data), 2))
        ax2.step(pix, resid, where='mid', c='purple', label='Data - Model')
        ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax2.errorbar(pix, resid, yerr=err, c='purple', fmt='none')
        ax2.set_xlabel('Pixel Number', fontsize=14)
        ax2.set_ylabel('Residuals', fontsize=14)
        ax2.legend(fontsize=12)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.grid(True)
        plt.tight_layout()    

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=200, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def plot_WLS_lines_in_order(self, orderlet, order, fig_path=None, show_plot=False):
        """
        Generate an array plots of spectral lines.

        Args:
            orderlet (string) - 'SCI1', 'SCI2', 'SCI3', 'CAL', or 'SKY'
            order (integer) - order number
            fig_path (string) - set to the path for the file to be generated.
                                default=None
            show_plot (boolean) - show the plot in the current environment.
                                  default=False

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).
        """
        orderdict = self.wls_dict['orderlets'][orderlet]['orders'][order]
        nlines = count_dict(orderdict['lines'])
        sidelength = int(np.ceil(np.sqrt(nlines)))

        # font and box sizes for different size arrays
        if sidelength > 10:  # big array
            size = sidelength*2
            fs_annotation = 8
            fs_title = 36
        else:
            size = sidelength*2.5
            fs_annotation = 9 #medium-size array
            fs_title = 24

        fig, axes = plt.subplots(sidelength, sidelength, figsize=(size,size))
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        norm = mcolors.Normalize(vmin=1, vmax=10)  # Create Normalize object
        cmap = plt.cm.Reds
        
        # Loop through the axes and create plots
        for i in range(sidelength):
            for j in range(sidelength):
                try:
                    linedict = orderdict['lines'][i+sidelength*j]
                    data  = linedict['data']
                    model = linedict['model']
                    err = np.sqrt(np.abs(data))
                    npix = len(data)
                    pix = np.arange(npix)
                    minval = np.min(np.concatenate((data, model.T), axis=0))
                    maxval = np.max(np.concatenate((data, model.T), axis=0))
                    sigma_A    = np.sqrt(np.abs(linedict['covar'][0,0]))
                    sigma_mu   = np.sqrt(np.abs(linedict['covar'][1,1]))
                    sigma_sig  = np.sqrt(np.abs(linedict['covar'][2,2]))
                    sigma_c    = np.sqrt(np.abs(linedict['covar'][3,3]))
                    annotation1  = r'$\mu$ = ' + f'{linedict["mu"]:.3f}' + r' $\pm$ ' + f'{sigma_mu:.3f}' + ' px\n'
                    annotation1 += r'$\sigma$ = ' + f'{linedict["sig"]:.3f}' + ' px\n'
                    annotation1 += 'A = ' + str(int(linedict["amp"])) + '\n'
                    annotation1 += 'c = ' + f'{linedict["const"]:.2f}' + '\n'
                    annotation2  = 'line = ' + str(i+j*sidelength) + '\n'
                    annotation2 += r'$\chi^2$ = ' + f'{linedict["chi2"]:.3g}' + '\n'
                    annotation2 += 'RMS = ' + f'{linedict["rms"]:.3g}' + '\n'
                    annotation2 += r'$\Delta\mu$ = ' + f'{linedict["mu_diff"]:.3g}' + ' px\n'
                    axes[i,j].step(pix, data,  c='b', where='mid')
                    axes[i,j].step(pix, model, c='r', where='mid')
                    axes[i,j].errorbar(pix, data,  yerr=err, c='b', fmt='none')
                    axes[i,j].axhline(y=0, color='darkgray', linestyle='--')
                    axes[i,j].set_facecolor(cmap(norm(linedict['chi2']), alpha=0.5))
                    axes[i,j].set_xticks([])  
                    axes[i,j].set_yticks([])  
                    #axes[i,j].set_ylim(minval, 1.1*maxval, axis=0) 
                    ylim = axes[i,j].get_ylim()
                    axes[i,j].set_ylim(ylim[0], 0.08*(ylim[1]-ylim[0]) + ylim[1])
                    axes[i,j].annotate(annotation1, xy=(0.02, 0.97), 
                                xycoords='axes fraction', fontsize=fs_annotation, color='k', va='top')
                    axes[i,j].annotate(annotation2, xy=(0.98, 0.97), 
                                xycoords='axes fraction', fontsize=fs_annotation, color='k', va='top', ha='right')
                except:
                    axes[i,j].set_xticks([])  
                    axes[i,j].set_yticks([])  
        
        plt.suptitle(self.chip + ' ' + orderlet + ' (order ' + str(order) + ') - ' + str(nlines) + '/' + \
                     str(orderdict['num_detected_peaks']) + ' lines', fontsize=fs_title, y=1.01)

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=300, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')

    
    def plot_WLS_lines_in_orderlet(self, orderlet, fig_path=None, show_plot=False):
        """
        Generate an array plots of spectral lines for all orders of a given orderlet.

        Args:
            orderlet (string) - 'SCI1', 'SCI2', 'SCI3', 'CAL', or 'SKY'
            order (integer) - order number
            fig_path (string) - set to the path for the file to be generated.
                                default=None
            show_plot (boolean) - show the plot in the current environment.
                                  default=False

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).
        """
        orderletdict = self.wls_dict['orderlets'][orderlet]
        norders = orderletdict['norders']
        nlines_arr = np.zeros(norders, dtype=np.int)
        for o in np.arange(norders):
            nlines_arr[o] = count_dict(orderletdict['orders'][o]['lines'])
        nrows = norders
        ncolumns = int(np.max(nlines_arr))

        # font and box sizes for different size arrays
        figsize = (1.8*ncolumns+3, 1.8*nrows+3)
        fs_annotation = 6
        fs_axis = 12
        fs_title = 48
#        if sidelength > 10:  # big array
#            size = sidelength*2
#            fs_annotation = 8
#            fs_title = 36
#        else:
#            size = sidelength*2.5
#            fs_annotation = 9 #medium-size array
#            fs_title = 24

        fig, axes = plt.subplots(nrows, ncolumns, figsize=figsize)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.suptitle(self.chip + ' ' + orderlet, fontsize=fs_title, y=1.01)
        norm = mcolors.Normalize(vmin=1, vmax=10)  # Create Normalize object
        cmap = plt.cm.Reds
        
        # Loop through the axes and create plots
        for i in np.arange(norders):
            o = norders-1-i # make Order 0 on the bottom
            for j in range(ncolumns):
                try:
                    linedict = orderletdict['orders'][o]['lines'][j]
                    data  = linedict['data']
                    model = linedict['model']
                    err = np.sqrt(np.abs(data))
                    npix = len(data)
                    pix = np.arange(npix)
                    sigma_A    = np.sqrt(np.abs(linedict['covar'][0,0]))
                    sigma_mu   = np.sqrt(np.abs(linedict['covar'][1,1]))
                    sigma_sig  = np.sqrt(np.abs(linedict['covar'][2,2]))
                    sigma_c    = np.sqrt(np.abs(linedict['covar'][3,3]))
                    annotation1  = r'$\mu$ = ' + f'{linedict["mu"]:.3f}' + r' $\pm$ ' + f'{sigma_mu:.3f}' + ' px\n'
                    annotation1 += r'$\sigma$ = ' + f'{linedict["sig"]:.3f}' + ' px\n'
                    annotation1 += 'A = ' + str(int(linedict["amp"])) + '\n'
                    annotation1 += 'c = ' + f'{linedict["const"]:.2f}' + '\n'
                    annotation2  = 'line = ' + str(j) + '\n'
                    annotation2 += r'$\chi^2$ = ' + f'{linedict["chi2"]:.3g}' + '\n'
                    annotation2 += 'RMS = ' + f'{linedict["rms"]:.3g}' + '\n'
                    annotation2 += r'$\Delta\mu$ = ' + f'{linedict["mu_diff"]:.3g}' + ' px\n'
                    axes[o,j].step(pix, data,  c='b', where='mid')
                    axes[o,j].step(pix, model, c='r', where='mid')
                    axes[o,j].errorbar(pix, data,  yerr=err, c='b', fmt='none')
                    axes[o,j].axhline(y=0, color='darkgray', linestyle='--')
                    axes[o,j].set_facecolor(cmap(norm(linedict['chi2']), alpha=0.5))
                    axes[o,j].set_xticks([])  
                    axes[o,j].set_yticks([])  
                    ylim = axes[o,j].get_ylim()
                    axes[o,j].set_ylim(ylim[0], 0.08*(ylim[1]-ylim[0]) + ylim[1])
                    axes[o,j].annotate(annotation1, xy=(0.02, 0.97), 
                                xycoords='axes fraction', fontsize=fs_annotation, color='k', va='top')
                    axes[o,j].annotate(annotation2, xy=(0.98, 0.97), 
                                xycoords='axes fraction', fontsize=fs_annotation, color='k', va='top', ha='right')
                    if j == 0:
                        axes[o,j].set_ylabel('Order ' + str(o))
                except Exception as e:
                    #print(e)
                    axes[o,j].set_xticks([])  
                    axes[o,j].set_yticks([])  
        
        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=300, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')

    
    def plot_wave_diff_final_initial(self, orderlet, fig_path=None, show_plot=False, warning_ms=10, alarm_ms=100): 
        """
        Generate an array of plots of the difference between initial and initial 
        wavelength solutions for the orders of a given orderlet.

        Args:
            fig_path (string) - set to the path for the file to be generated.
                                default=None
            show_plot (boolean) - show the plot in the current environment.
                                  default=False                                  
            warming_ms (double) - level in m/s where the orange 'warning' boxes start
                                  default=10
            alarm_ms (double) - level in m/s where the orange 'warning' boxes end
                                and the red 'alarm' boxes being
                                default=100

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).
        """

        orderletdict = self.wls_dict['orderlets'][orderlet]
        norders = self.wls_dict['orderlets'][orderlet]['norders']
        nrows = 9
        ncolumns = 4
        avg_delta_rv_arr = np.zeros(norders, dtype=np.int)

        fig, axes = plt.subplots(nrows, ncolumns, figsize=(36, 25))
        plt.subplots_adjust(wspace=0.10, hspace=0.15, left=0.10, right=0.99, top=0.95, bottom=0.08)
        plt.suptitle(self.chip + ' ' + orderlet, fontsize=48)
        fig.text(0.04, 0.5, r'$\Delta$WLS (final - initial) [m/s]', fontsize=36, va='center', rotation='vertical')
        fig.text(0.5, 0.04, r'$\lambda$ (final) [Ang]', fontsize=36, va='center', rotation='horizontal')

        for i in np.arange(nrows):
            for j in range(ncolumns):
                o = nrows*j + i
                try:
                    # Plot data
                    orderdict = orderletdict['orders'][o]
                    delta_rv = 2.998e8*(orderdict['fitted_wls']-orderdict['initial_wls'])/orderdict['fitted_wls']
                    avg_delta_rv_arr[o] = np.mean(delta_rv)
                    axes[i,j].plot(orderdict['fitted_wls'], delta_rv, linewidth=4)
            
                    # Draw a rectangular boxes
                    xmin, xmax = axes[i,j].get_xlim()  # Get the current x-axis limits to span the entire range horizontally
                    ymin_green,  ymax_green  = -warning_ms, warning_ms
                    ymin_orange, ymax_orange =  warning_ms, alarm_ms
                    ymin_red,    ymax_red    =  alarm_ms,   alarm_ms*100
                    alpha_green   = 0.20
                    alpha_orange1 = 0.20
                    alpha_orange2 = 0.20
                    alpha_red1    = 0.15
                    alpha_red2    = 0.15
                    if np.all((delta_rv <  warning_ms) & (delta_rv > -warning_ms)): 
                        alpha_green = 0.50
                    if np.any((delta_rv >  warning_ms) & (delta_rv <  alarm_ms)): # highlight WLS problems
                        alpha_orange1 = 0.50
                    if np.any((delta_rv < -warning_ms) & (delta_rv > -alarm_ms)):
                        alpha_orange2 = 0.50
                    if np.any((delta_rv >  alarm_ms)):
                        alpha_red1 = 0.40
                    if np.any((delta_rv < -alarm_ms)):
                        alpha_red2 = 0.40
                    rect_green   = Rectangle((xmin, ymin_green),   xmax-xmin,   ymax_green -ymin_green,   facecolor='green',  alpha=alpha_green)
                    rect_orange1 = Rectangle((xmin, ymin_orange),  xmax-xmin,   ymax_orange-ymin_orange,  facecolor='orange', alpha=alpha_orange1)
                    rect_orange2 = Rectangle((xmin, -ymin_orange), xmax-xmin, -(ymax_orange-ymin_orange), facecolor='orange', alpha=alpha_orange2)
                    rect_red1    = Rectangle((xmin, ymin_red),     xmax-xmin,   ymax_red   -ymin_red,     facecolor='red',    alpha=alpha_red1)
                    rect_red2    = Rectangle((xmin, -ymin_red),    xmax-xmin, -(ymax_red   -ymin_red),    facecolor='red',    alpha=alpha_red2)
                    axes[i,j].add_patch(rect_green)
                    axes[i,j].add_patch(rect_orange1)
                    axes[i,j].add_patch(rect_orange2)
                    axes[i,j].add_patch(rect_red1)
                    axes[i,j].add_patch(rect_red2)

                    # Dots, lines, annotations
                    blend_transform = transforms.blended_transform_factory(axes[i,j].transData, axes[i,j].transAxes)
                    for l in np.arange(len(orderdict['known_wavelengths_vac'])):
                        axes[i,j].axvline(orderdict['known_wavelengths_vac'][l], color='darkgray', linestyle='-', linewidth=0.5)
                        axes[i,j].plot(orderdict['known_wavelengths_vac'][l], 0.95, 'ko', transform=blend_transform, markersize=2)
                    axes[i,j].annotate(r'<$\Delta$WLS> = ' + str(int(avg_delta_rv_arr[o])) + ' m/s', xy=(0.99, 0.03), xycoords='axes fraction', 
                                 fontsize=10, ha='right', va='bottom',
                                 bbox=dict(boxstyle="square,pad=0.3", facecolor="white", alpha=0.75))
                    
                    # Axes setup
                    if i == nrows-1 or o == norders-1:
                        axes[i,j].set_xlabel(r'Wavelength (final) [Ang]', fontsize=18)
                    axes[i,j].set_ylabel('Order ' + str(o) + '', fontsize=18)
                    axes[i,j].tick_params(axis='both', labelsize=12)
                    axes[i,j].axhline(0, color='black', linestyle='--', linewidth=1)
                    axes[i,j].set_xlim(np.max(orderdict['fitted_wls']), np.min(orderdict['fitted_wls']))
                    axes[i,j].set_ylim(-alarm_ms*100, alarm_ms*100)
                    axes[i,j].set_yscale('symlog', linthresh=warning_ms/10, linscale=1)
                    if j == 0:
                        locator = SymmetricalLogLocator(base=10, linthresh=warning_ms/10, subs=[1])
                        axes[i,j].yaxis.set_major_locator(locator)
                        axes[i,j].yaxis.set_major_formatter(ScalarFormatter())
                        yticks = axes[i,j].get_yticks()
                        labels = ['' if label == '0.0' else label for label in yticks.astype(str)]
                        axes[i,j].set_yticks(yticks) 
                        axes[i,j].set_yticklabels(labels)
                    else:
                        axes[i,j].set_yticklabels([])
                            
                except Exception as e:
                    #print(e)
                    if o != nrows*ncolumns-1:
                        axes[i,j].set_xticks([])  
                        axes[i,j].set_yticks([])  
                        for spine in axes[i,j].spines.values():
                            spine.set_visible(False)
                    pass

        # Delta RV vs order number plot in lower-right corner
        pos = axes[nrows-1, ncolumns-1].get_position()
        new_pos = [pos.x0+0.05, pos.y0, pos.width * 0.75, pos.height * 0.7]  # Example adjustment
        axes[nrows-1, ncolumns-1].set_position(new_pos)
        axes[nrows-1, ncolumns-1].scatter(np.arange(norders), avg_delta_rv_arr, s=50, c='tab:blue')
        axes[nrows-1, ncolumns-1].axhline(0, color='black', linestyle='-', linewidth=2)
        axes[nrows-1, ncolumns-1].set_xlabel('Order Number', fontsize=16)
        axes[nrows-1, ncolumns-1].set_ylabel(r'<$\Delta$WLS> (m/s)', fontsize=16)
        axes[nrows-1, ncolumns-1].tick_params(axis='both', labelsize=14)
        axes[nrows-1, ncolumns-1].grid(True, linewidth=1.5)
        axes[nrows-1, ncolumns-1].xaxis.set_minor_locator(MultipleLocator(1))
        for spine in axes[nrows-1, ncolumns-1].spines.values():
            spine.set_linewidth(3) 

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=500, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')

class AnalyzeTwoWLSDict:

    """
    Description:
        This class contains functions to compare two wavelength solutions 
        stored as dictionaries.  It is assumed that the two WLSs will be 
        of the same chip (Green or Red).

    Arguments:
        WLSDict1 - a WLS Dictionary, stored either the filename of a 
                  (possibly) zip-compressed JSON file, or a TBD object
        WLSDict2 - a second WLS Dictionary

    Attributes:
        wls_dict1 - dictionary of wavelength solution
        wls_dict2 - dictionary of wavelength solution
    """

    def __init__(self, WLSDict_filename1, WLSDict_filename2, name1='', name2='', logger=None):
        self.logger = logger if logger is not None else DummyLogger()        
        self.wls_dict1 = read_wls_json(WLSDict_filename1) 
        self.wls_dict2 = read_wls_json(WLSDict_filename2)
        self.name1 = name1
        self.name2 = name2
        try:
            self.chip = self.wls_dict1['chip']
        except:
            self.chip = '<chip>'


    def plot_wave_diff_wls(self, orderlet, fig_path=None, show_plot=False, warning_ms=10, alarm_ms=100): 
        """
        Generate an array of plots of the difference between two wavelength solutions 
        for the orders of a given orderlet.

        Args:
            orderlet (string) - 'SCI1', 'SCI2', 'SCI3', 'CAL', or 'SKY'
            fig_path (string) - set to the path for the file to be generated.
                                default=None
            show_plot (boolean) - show the plot in the current environment.
                                  default=False                                  
            warming_ms (double) - level in m/s where the orange 'warning' boxes start
                                  default=10
            alarm_ms (double) - level in m/s where the orange 'warning' boxes end
                                and the red 'alarm' boxes being
                                default=100

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).
        """

        orderletdict1 = self.wls_dict1['orderlets'][orderlet]
        orderletdict2 = self.wls_dict2['orderlets'][orderlet]
        norders = self.wls_dict2['orderlets'][orderlet]['norders']
        nrows = 9
        ncolumns = 4
        avg_delta_rv_arr = np.zeros(norders, dtype=np.int)

        fig, axes = plt.subplots(nrows, ncolumns, figsize=(36, 25))
        plt.subplots_adjust(wspace=0.10, hspace=0.15, left=0.10, right=0.99, top=0.95, bottom=0.08)
        plt.suptitle(self.chip + ' ' + orderlet, fontsize=48)
        fig.text(0.04, 0.5, r'$\Delta$WLS (' + self.name2 + ' - ' + self.name1 + ') [m/s]', fontsize=36, va='center', rotation='vertical')
        fig.text(0.5, 0.04, r'$\lambda$  [Ang]', fontsize=36, va='center', rotation='horizontal')

        for i in np.arange(nrows):
            for j in range(ncolumns):
                o = nrows*j + i
                try:
                    # Plot data
                    orderdict1 = orderletdict1['orders'][o]
                    orderdict2 = orderletdict2['orders'][o]
#                    wls1 = 
                    delta_rv = 2.998e8*(orderdict2['fitted_wls']-orderdict1['fitted_wls'])/orderdict1['fitted_wls']
                    avg_delta_rv_arr[o] = np.mean(delta_rv)
                    axes[i,j].plot(orderdict1['fitted_wls'], delta_rv, linewidth=4)
            
                    # Draw a rectangular boxes
                    xmin, xmax = axes[i,j].get_xlim()  # Get the current x-axis limits to span the entire range horizontally
                    ymin_green,  ymax_green  = -warning_ms, warning_ms
                    ymin_orange, ymax_orange =  warning_ms, alarm_ms
                    ymin_red,    ymax_red    =  alarm_ms,   alarm_ms*100
                    alpha_green   = 0.20
                    alpha_orange1 = 0.20
                    alpha_orange2 = 0.20
                    alpha_red1    = 0.15
                    alpha_red2    = 0.15
                    if np.all((delta_rv <  warning_ms) & (delta_rv > -warning_ms)): 
                        alpha_green = 0.50
                    if np.any((delta_rv >  warning_ms) & (delta_rv <  alarm_ms)): # highlight WLS problems
                        alpha_orange1 = 0.50
                    if np.any((delta_rv < -warning_ms) & (delta_rv > -alarm_ms)):
                        alpha_orange2 = 0.50
                    if np.any((delta_rv >  alarm_ms)):
                        alpha_red1 = 0.40
                    if np.any((delta_rv < -alarm_ms)):
                        alpha_red2 = 0.40
                    rect_green   = Rectangle((xmin, ymin_green),   xmax-xmin,   ymax_green -ymin_green,   facecolor='green',  alpha=alpha_green)
                    rect_orange1 = Rectangle((xmin, ymin_orange),  xmax-xmin,   ymax_orange-ymin_orange,  facecolor='orange', alpha=alpha_orange1)
                    rect_orange2 = Rectangle((xmin, -ymin_orange), xmax-xmin, -(ymax_orange-ymin_orange), facecolor='orange', alpha=alpha_orange2)
                    rect_red1    = Rectangle((xmin, ymin_red),     xmax-xmin,   ymax_red   -ymin_red,     facecolor='red',    alpha=alpha_red1)
                    rect_red2    = Rectangle((xmin, -ymin_red),    xmax-xmin, -(ymax_red   -ymin_red),    facecolor='red',    alpha=alpha_red2)
                    axes[i,j].add_patch(rect_green)
                    axes[i,j].add_patch(rect_orange1)
                    axes[i,j].add_patch(rect_orange2)
                    axes[i,j].add_patch(rect_red1)
                    axes[i,j].add_patch(rect_red2)

                    # Dots, lines, annotations
                    #blend_transform = transforms.blended_transform_factory(axes[i,j].transData, axes[i,j].transAxes)
                    #for l in np.arange(len(orderdict['known_wavelengths_vac'])):
                        #axes[i,j].axvline(orderdict['known_wavelengths_vac'][l], color='darkgray', linestyle='-', linewidth=0.5)
                        #axes[i,j].plot(orderdict['known_wavelengths_vac'][l], 0.95, 'ko', transform=blend_transform, markersize=2)
                    axes[i,j].annotate(r'<$\Delta$WLS> = ' + str(int(avg_delta_rv_arr[o])) + ' m/s', xy=(0.99, 0.03), xycoords='axes fraction', 
                                 fontsize=10, ha='right', va='bottom',
                                 bbox=dict(boxstyle="square,pad=0.3", facecolor="white", alpha=0.75))
                    
                    # Axes setup
                    if i == nrows-1 or o == norders-1:
                        axes[i,j].set_xlabel(r'Wavelength (final) [Ang]', fontsize=18)
                    axes[i,j].set_ylabel('Order ' + str(o) + '', fontsize=18)
                    axes[i,j].tick_params(axis='both', labelsize=12)
                    axes[i,j].axhline(0, color='black', linestyle='--', linewidth=1)
                    axes[i,j].set_xlim(np.max(orderdict1['fitted_wls']), np.min(orderdict1['fitted_wls']))
                    axes[i,j].set_ylim(-alarm_ms*100, alarm_ms*100)
                    axes[i,j].set_yscale('symlog', linthresh=warning_ms/10, linscale=1)
                    if j == 0:
                        locator = SymmetricalLogLocator(base=10, linthresh=warning_ms/10, subs=[1])
                        axes[i,j].yaxis.set_major_locator(locator)
                        axes[i,j].yaxis.set_major_formatter(ScalarFormatter())
                        yticks = axes[i,j].get_yticks()
                        labels = ['' if label == '0.0' else label for label in yticks.astype(str)]
                        axes[i,j].set_yticks(yticks) 
                        axes[i,j].set_yticklabels(labels)
                    else:
                        axes[i,j].set_yticklabels([])
                            
                except Exception as e:
                    #print(e)
                    if o != nrows*ncolumns-1:
                        axes[i,j].set_xticks([])  
                        axes[i,j].set_yticks([])  
                        for spine in axes[i,j].spines.values():
                            spine.set_visible(False)
                    pass

        # Delta RV vs order number plot in lower-right corner
        pos = axes[nrows-1, ncolumns-1].get_position()
        new_pos = [pos.x0+0.05, pos.y0, pos.width * 0.75, pos.height * 0.7]  # Example adjustment
        axes[nrows-1, ncolumns-1].set_position(new_pos)
        axes[nrows-1, ncolumns-1].scatter(np.arange(norders), avg_delta_rv_arr, s=50, c='tab:blue')
        axes[nrows-1, ncolumns-1].axhline(0, color='black', linestyle='-', linewidth=2)
        axes[nrows-1, ncolumns-1].set_xlabel('Order Number', fontsize=16)
        axes[nrows-1, ncolumns-1].set_ylabel(r'<$\Delta$WLS> (m/s)', fontsize=16)
        axes[nrows-1, ncolumns-1].tick_params(axis='both', labelsize=14)
        axes[nrows-1, ncolumns-1].grid(True, linewidth=1.5)
        axes[nrows-1, ncolumns-1].xaxis.set_minor_locator(MultipleLocator(1))
        for spine in axes[nrows-1, ncolumns-1].spines.values():
            spine.set_linewidth(3) 

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=500, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


def count_dict(wls_dict):
    """
    Count the number of lines or orders (whichever is the next level of hierarchy) 
    in a WLS dictionary.

    Args:
        wls_dict - WLS dictionary, e.g. red_WLSDict.wls_dict['orderlets']['SCI1']['orders'][10]

    Returns:
        number of lines or orders in the dictionary
    """
    nlines = 0 # number of lines or orders
    for key in wls_dict.keys():
        if isinstance(key, int):
            nlines += 1
    return nlines


# These methods are used to read and write JSON-formatted files that store WLS dictionaries.
def numpy_to_list(obj):
    """
    Converts a dictionary with Numpy arrays into a dictionary with Python lists.
    """
    if isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def list_to_numpy(obj):
    """
    Inverse of the above function.  This method also recognizes 'lines' and 'orders'
    and makes the next level down an integer not a string.
    This method is called recursively in read_wls_json() to it needs to be separate
    from that.
    """
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            # Check if the key is 'orders' and the value is a dictionary
            if k == 'orders' and isinstance(v, dict):
                new_obj[k] = {}
                for sub_k, sub_v in v.items():
                    # Try to convert sub_k to an integer if possible
                    try:
                        sub_k = int(sub_k)
                    except ValueError:
                        pass  # Not an integer, keep it as is
                    new_obj[k][sub_k] = list_to_numpy(sub_v)
            # Check if the key is 'lines' and the value is a dictionary
            elif k == 'lines' and isinstance(v, dict):
                new_obj[k] = {}
                for sub_k, sub_v in v.items():
                    # Try to convert sub_k to an integer if possible
                    try:
                        sub_k = int(sub_k)
                    except ValueError:
                        pass  # Not an integer, keep it as is
                    new_obj[k][sub_k] = list_to_numpy(sub_v)
            else:
                new_obj[k] = list_to_numpy(v)
        return new_obj
    elif isinstance(obj, list):
        return np.array(obj).astype(np.float64)
    elif isinstance(obj, str):
        try:
            return int(obj)  # Try to convert string to integer
        except ValueError:
            return obj  # If conversion fails, keep it as a string
    else:
        return obj
       
def write_wls_json(dictionary, filename):
    """
    Write a zip-compressed JSON file, converting any numpy arrays (assumed to be doubles!) 
    into lists along the way.
    """
    new_dictionary = numpy_to_list(dictionary)
    json_str = json.dumps(new_dictionary)
    with gzip.open(filename, 'wt', encoding='utf-8') as zipfile:
        zipfile.write(json_str)
            
def read_wls_json(filename):
    """
    Read a zip-compressed JSON file, converting any lists (assumed to be doubles!) 
    into numpy arrays.
    """
    # Read and decompress the JSON string from the file
    with gzip.open(filename, 'rt', encoding='utf-8') as zipfile:
        json_str = zipfile.read()
    dictionary = json.loads(json_str) # Convert JSON string to Python object
    new_dictionary = list_to_numpy(dictionary)
    return new_dictionary

   
