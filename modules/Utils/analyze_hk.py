import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from modules.Utils.kpf_parse import HeaderParse

class AnalyzeHK:

    """
    Description:
        This class contains functions to analyze data from KPF's Ca H&K spectrometer
        (storing the results as attributes) and functions for plot the data products.
        Some of the functions need to be filled in.

    Arguments:
        L0 - a KPF L0 object

    Attributes:
        TBD
    """

#[pipeline_20230720.log][INFO]:/data/masters/kpfMaster_HKwave20220909_sci.csv
#        plot_trace_boxes(hdulist['ca_hk'].data,trace_location,trace_location_sky)

    def __init__(self, L0, trace_file=None, offset=-1, wavesoln_file=None, logger=None):

        if logger:
            self.logger = logger
            self.logger.debug('Initializing AnalyzeHK object.')
        else:
            self.logger = None
        self.trace_file = trace_file
        self.wavesoln_file = wavesoln_file
        self.offset = offset
        self.image = np.array(L0['CA_HK'].data)
        primary_header = HeaderParse(L0, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        self.ObsID = primary_header.get_obsid()
        self.rv_shift = self.header['TARGRADV'] #TO-DO: check if keyword exists
        if trace_file != None:
            for fiber in ['sci','sky']:
                order_col_name = 'order'
                fiber_col_name = 'fiber'
                loc_col_names = ['y0', 'x0', 'yf','xf']
                loc_result = pd.read_csv(trace_file, header=0, sep=' ')
                loc_vals = np.array(loc_result.values)
                loc_cols = np.array(loc_result.columns)
                loc_idx = {c: np.where(loc_cols == c)[0][0] for c in loc_col_names}
                order_idx = np.where(loc_cols == order_col_name)[0][0]
                fiber_idx = np.where(loc_cols == fiber_col_name)[0][0]
                loc_for_fiber = loc_vals[np.where(loc_vals[:, fiber_idx] == fiber)[0], :]  # rows with the same fiber
                trace_location = dict()
                for loc in loc_for_fiber: # add each row from loc_for_fiber to trace_location for fiber
                    trace_location[loc[order_idx]] = {'x1': loc[loc_idx['y0']]-self.offset,
                                                      'x2': loc[loc_idx['yf']]-self.offset,
                                                      'y1': loc[loc_idx['x0']],
                                                      'y2': loc[loc_idx['xf']]}
                if fiber == 'sci': self.trace_location_sci = trace_location # sci/sky had been backwards in previous versions
                if fiber == 'sky': self.trace_location_sky = trace_location
        if wavesoln_file != None:
            self.wave_lib = pd.read_csv(wavesoln_file, header=None, sep = ' ', comment = '#')
            self.wave_lib *= 1 - self.rv_shift/3e5 # Doppler shift wavelength solution


    def plot_HK_image_2D(self, fig_path=None, show_plot=False):

        """
        Generate a 2D image of the Ca H& K spectrum.  

        Args:
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """
        fig, ax = plt.subplots(figsize = (12,5),tight_layout=True)
        im = ax.imshow(self.image, 
                       vmin = np.nanpercentile(self.image,1),
                       vmax = np.nanpercentile(self.image,99.5), 
                       interpolation = 'None',
                       origin = 'lower',
                       aspect='auto')

        if self.trace_file != None:
            for i in self.trace_location_sci.keys():
                height = self.trace_location_sci[i]['x2'] - self.trace_location_sci[i]['x1']
                width  = self.trace_location_sci[i]['y2'] - self.trace_location_sci[i]['y1']
                ax.add_patch(patches.Rectangle((self.trace_location_sci[i]['y1'], 
                                                self.trace_location_sci[i]['x1']),
                                                width, height,
                                                linewidth=0.5, 
                                                edgecolor='r',
                                                facecolor='none'))
                if i == 0: 
                    ax.add_patch(patches.Rectangle((self.trace_location_sci[i]['y1'], 
                                                    self.trace_location_sci[i]['x1']),
                                                    width, height,
                                                    linewidth=0.5, 
                                                    edgecolor='r',
                                                    facecolor='none',
                                                    label = 'Sci'))
            for i in self.trace_location_sky.keys():
                height = self.trace_location_sky[i]['x2'] - self.trace_location_sky[i]['x1']
                width  = self.trace_location_sky[i]['y2'] - self.trace_location_sky[i]['y1']
                ax.add_patch(patches.Rectangle((self.trace_location_sky[i]['y1'], 
                                                self.trace_location_sky[i]['x1']),
                                                width, height,
                                                linewidth=0.5, 
                                                edgecolor='white',
                                                facecolor='none'))
                if i == 0: ax.add_patch(patches.Rectangle((self.trace_location_sky[i]['y1'], 
                                                           self.trace_location_sky[i]['x1']),
                                                           width, height, 
                                                           linewidth=0.5, 
                                                           edgecolor='white', 
                                                           facecolor='none',
                                                           label = 'Sky'))
        fig.colorbar(im, orientation='vertical',label = 'Counts (ADU) - Saturation at '+str(64232))
        ax.set_title('Ca H&K CCD: ' + str(self.ObsID) + ' - ' + self.name, fontsize=18)
        ax.set_xlabel('Column (pixel number)', fontsize=18, labelpad=10)
        ax.set_ylabel('Row (pixel number)', fontsize=18, labelpad=10)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        plt.legend()

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=500, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')

    def plot_HK_spectrum_1D(self, fig_path=None, show_plot=False):

        orders = np.array(self.wave_lib.columns)
        padding = 200

        plt.figure(figsize=(12,5),tight_layout=True)
        color_grid = ['purple','blue','green','yellow','orange','red']
        chk_bandpass  = [384.0, 401.7]
        caK           = [393.2, 393.5]
        caH           = [396.7, 397.0]
        Vcont         = [389.9, 391.9]
        Rcont         = [397.4, 399.4]

        fig, ax = plt.subplots(1, 1, figsize=(12,5))
        ax.fill_between(chk_bandpass, y1=0, y2=1, facecolor='gray', alpha=0.3, zorder=-100)
        ax.fill_between(caH,          y1=0, y2=1, facecolor='m',    alpha=0.3)
        ax.fill_between(caK,          y1=0, y2=1, facecolor='m',    alpha=0.3)
        ax.fill_between(Vcont,        y1=0, y2=1, facecolor='c',    alpha=0.3)
        ax.fill_between(Rcont,        y1=0, y2=1, facecolor='c',    alpha=0.3)

        ax.text(np.mean(Vcont)-0.6,0.08,'V cont.')
        ax.text(np.mean(Rcont)-0.6,0.08,'R cont.')
        ax.text(np.mean(caK)-0.155,0.08,'K')
        ax.text(np.mean(caH)-0.155,0.08,'H')
        ax.set_xlim(384,400)
        ax.set_ylim(0,1)
        ax.plot([396.847,396.847],[0,1],':',color ='black')
        ax.plot([393.366,393.366],[0,1],':',color ='black')
 
        # Add labels
        for i in range(len(orders)):
            wav = self.wave_lib[i]
            flux = np.sum(self.image[self.trace_location_sci[i]['x1']:self.trace_location_sci[i]['x2'],:],axis=0)
            ax.plot(wav[padding:-padding],flux[padding:-padding]/np.nanpercentile(flux[padding:-padding],99.9),color = color_grid[i],linewidth = 0.5)
        ax.set_xlabel('Wavelength (nm)',fontsize=18)
        ax.set_ylabel('Flux',fontsize=18)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        plt.title('Ca H&K Spectrum: ' + str(self.ObsID) + ' - ' + self.name, fontsize=18)

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=200, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')

