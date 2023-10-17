import time
import json
import gzip
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    	LFC - line positions vs. initial guess
    	heatmap or 1D plot for a parameter (like mu_diff)
    	# lines per order/orderlet
    	chi^2 distribution (perhaps by order/orderlet)
    	develop cuts: amp > 0, sig > 1
    """

    def __init__(self, WLSDict_filename, logger=None):
        self.logger = logger if logger is not None else DummyLogger()
        #self.logger.debug('Initializing AnalyzeWLSDict object')
        
        self.wls_dict = read_wls_json(WLSDict_filename) 


    def plot_WLS_line(self, orderlet, order, line, fig_path=None, show_plot=False):
        """
        Generate a plot of a single spectral line.

        Args:
            orderlet (string) - 'SCI1', 'SCI2', 'SCI3', 'CAL', or 'SKY'
            order (integer) - order number
            line (integer) - line number

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
        ax1.set_title(orderlet + ' - order ' + str(order) + ', line ' + str(line) + r' ($\chi^2$ = ' + f'{linedict["chi2"]:.3g}'+ ')', fontsize=14)
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

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).
        """
        orderdict = self.wls_dict['orderlets'][orderlet]['orders'][order]
        nlines = count_dict(orderdict['lines'])
        sidelength = int(np.ceil(np.sqrt(nlines)))

        fig, axes = plt.subplots(sidelength, sidelength, figsize=(sidelength*2.5,sidelength*2.5))
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
                    #amp     = linedict['amp']
                    #mu      = linedict['mu']
                    #sig     = linedict['sig']
                    #const   = linedict['const']
                    #mu_diff = linedict['mu_diff']
                    #chi2    = linedict['chi2']
                    annotation1 =  'A = ' + str(int(linedict["amp"])) + '\n'
                    annotation1 += r'$\mu$ = ' + f'{linedict["mu"]:.4f}' + '\n'
                    annotation1 += r'$\sigma$ = ' + f'{linedict["sig"]:.4f}' + '\n'
                    annotation1 += 'c = ' + f'{linedict["const"]:.2f}' + '\n'
                    annotation2  = 'line = ' + str(i+j*sidelength) + '\n'
                    annotation2 += r'$\chi^2$ = ' + f'{linedict["chi2"]:.3g}' + '\n'
                    annotation2 += 'RMS = ' + f'{linedict["rms"]:.3g}' + '\n'
                    annotation2 += r'$\Delta\mu$ = ' + f'{linedict["mu_diff"]:.3g}' + '\n'
                    axes[i,j].step(pix, data,  c='b', where='mid')
                    axes[i,j].step(pix, model, c='r', where='mid')
                    axes[i,j].errorbar(pix, data,  yerr=err, c='b', fmt='none')
                    axes[i,j].set_facecolor(cmap(norm(linedict['chi2']), alpha=0.5))
                    axes[i,j].set_xticks([])  
                    axes[i,j].set_yticks([])  
                    axes[i,j].annotate(annotation1, xy=(0.02, 0.97), 
                                xycoords='axes fraction', fontsize=8, color='k', va='top')
                    axes[i,j].annotate(annotation2, xy=(0.98, 0.97), 
                                xycoords='axes fraction', fontsize=8, color='k', va='top', ha='right')
                except:
                    axes[i,j].set_xticks([])  
                    axes[i,j].set_yticks([])  
        
        plt.suptitle(orderlet + ' (' + str(order) + ') - ' + str(nlines) + ' lines', fontsize=36, y=1.01)

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=300, facecolor='w')
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

   
