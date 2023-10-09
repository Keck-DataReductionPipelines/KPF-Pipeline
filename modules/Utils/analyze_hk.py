import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import ScalarFormatter
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
        self.color_grid = ['purple','darkblue','darkgreen','gold','darkorange','darkred']


    def plot_HK_image_2D(self, fig_path=None, kpftype='L0', show_plot=False):

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
                       vmin = np.nanpercentile(self.image,0.01),
                       vmax = np.nanpercentile(self.image,99.9), 
                       interpolation = 'None',
                       origin = 'lower',
                       cmap='viridis',
                       aspect='auto')
        ax.grid(False)

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
        ax.set_title('Ca H&K CCD: ' + str(self.ObsID) + ' (' + str(kpftype)+ ') - ' + self.name, fontsize=18)
        ax.set_xlabel('Column (pixel number)', fontsize=18, labelpad=10)
        ax.set_ylabel('Row (pixel number)', fontsize=18, labelpad=10)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        plt.legend(facecolor='lightgray')

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=500, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')

    def plot_HK_2D_column_cut(self, column=512, fig_path=None, kpftype='L0', show_plot=False):

        """
        Generate a column cut plot of a 2D image of the Ca H& K spectrum.  

        Args:
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """
        lw = 1 # linewidth
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, tight_layout=True)
        axs[0].step(np.arange(255), self.image[:,column], color='k', linewidth=lw)
        axs[1].step(np.arange(255), self.image[:,column], color='k', linewidth=lw)
        
        ymin = np.percentile(self.image[:,column],1)
        ymax = np.percentile(self.image[:,column],99)
        yrange = ymax-ymin

        # to determine ymin/ymax, create subarray with missing pixels for SCI orders 
        remove_indices = []
        remove_indices.extend(range(0,25)) # hack because there's an unextracted order not in the trace locations
        remove_indices.extend(range(230,255)) # hack because there's an unextracted order not in the trace locations
        for key, value in self.trace_location_sky.items():
            remove_indices.extend(range(value['x1']-2, value['x2']+2))
        new_cut = np.delete(self.image[:,column], remove_indices)
        ymin_off_order = np.percentile(new_cut,1)
        ymax_off_order = np.percentile(new_cut,99)
        yrange_off_order = ymax_off_order-ymin_off_order

        # Add color highlights
        for o in range(-1,5):  # iterate through the order definitions -- -1 to +5
            axs[0].step(range(self.trace_location_sky[o]['x1'],self.trace_location_sky[o]['x2']), 
                        self.image[self.trace_location_sky[o]['x1']:self.trace_location_sky[o]['x2'],column], 
                        color= self.color_grid[o+1], linewidth=lw)
            axs[0].step(range(self.trace_location_sci[o]['x1'],self.trace_location_sci[o]['x2']), 
                        self.image[self.trace_location_sci[o]['x1']:self.trace_location_sci[o]['x2'],column], 
                        color= self.color_grid[o+1], linewidth=lw)
            axs[1].step(range(self.trace_location_sky[o]['x1'],self.trace_location_sky[o]['x2']), 
                        self.image[self.trace_location_sky[o]['x1']:self.trace_location_sky[o]['x2'],column], 
                        color= self.color_grid[o+1], linewidth=lw)
            axs[1].step(range(self.trace_location_sci[o]['x1'],self.trace_location_sci[o]['x2']), 
                        self.image[self.trace_location_sci[o]['x1']:self.trace_location_sci[o]['x2'],column], 
                        color= self.color_grid[o+1], linewidth=lw)

        # Set axis parameters
        axs[0].set_title('Ca H&K - Cut at Column #' + str(column) + ': ' + str(self.ObsID) + ' - ' + self.name, fontsize=18)
        axs[1].set_xlabel('Row Number',fontsize=18)
        axs[0].set_ylabel('Flux (ADU)',fontsize=18)
        axs[1].set_ylabel('Flux (ADU)',fontsize=18)
        axs[0].plot(0,255,[0,0],':',color ='white')
        axs[0].xaxis.set_tick_params(labelsize=14)
        axs[0].yaxis.set_tick_params(labelsize=14)
        axs[1].xaxis.set_tick_params(labelsize=14)
        axs[1].yaxis.set_tick_params(labelsize=14)
        axs[0].fill_between([0,255], y1=-1000000, y2=1000000, facecolor='lightgray', alpha=0.3, zorder=-100)
        axs[1].fill_between([0,255], y1=-1000000, y2=1000000, facecolor='lightgray', alpha=0.3, zorder=-100)
        axs[0].set_xlim(0,255)
        axs[1].set_xlim(0,255)
        axs[0].set_ylim(min([0,ymin-0.05*yrange]), ymax+0.05*yrange)
        axs[1].set_ylim(min([0,ymin_off_order-0.05*yrange_off_order]), ymax_off_order+0.05*yrange_off_order)

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=300, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')


    def plot_HK_spectrum_1D(self, trace='sci', fig_path=None, show_plot=False):

        """
        Generate a 1D spectrum plot of the Ca H& K spectrum.  

        Args:
            trace (string) - 'sci' or 'sky' to select the fiber
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """

        orders = np.array(self.wave_lib.columns)
        padding = 1

        #plt.figure(figsize=(12,5),tight_layout=True)
        fig, ax = plt.subplots(1, 1, figsize=(12,5))

        color_grid = ['purple','darkblue','darkgreen','gold','darkorange','darkred']
        chk_bandpass  = [383.0, 401.0]#[384.0, 401.7]
        caK           = [393.3663-0.150, 393.3663+0.150]
        caH           = [396.8469-0.150, 396.8469+0.150]
        Vcont         = [389.9, 391.9]
        Rcont         = [397.4, 399.4]

        # Compute and Plot spectra
        specHK = np.zeros((len(orders), self.image.shape[1]), dtype=np.float64)
        for i in range(len(orders)):
            wav = self.wave_lib[i]
            # This should be correct, but the sky and sci orders are flipped in the definition file
            if trace == 'sci':
                specHK[i,:] = np.sum(self.image[self.trace_location_sky[i]['x1']:self.trace_location_sky[i]['x2'],:],axis=0)
            elif trace == 'sky':
                specHK[i,:] = np.sum(self.image[self.trace_location_sci[i]['x1']:self.trace_location_sci[i]['x2'],:],axis=0)
            ax.plot(wav[padding:-padding],
                    specHK[i,padding:-padding],
                    color = color_grid[i],
                    linewidth = 1.0)
        ymin =      np.nanpercentile(specHK[:,padding:-padding],0.1)
        ymax = 1.15*np.nanpercentile(specHK[:,padding:-padding],99.9)
        yrange = (ymax-ymin)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(chk_bandpass[0], chk_bandpass[1])

        # Add labels
        plt.title('Ca H&K ' + trace.upper() + ' Spectrum: ' + str(self.ObsID) + ' - ' + self.name, fontsize=18)
        ax.set_xlabel('Wavelength (nm)',fontsize=18)
        ax.set_ylabel('Flux (ADU)',fontsize=18)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.fill_between(chk_bandpass, y1=ymin, y2=ymin+yrange, facecolor='gray', alpha=0.3, zorder=-100)
        ax.fill_between(caH,          y1=ymin, y2=ymin+yrange, facecolor='m',    alpha=0.3)
        ax.fill_between(caK,          y1=ymin, y2=ymin+yrange, facecolor='m',    alpha=0.3)
        ax.fill_between(Vcont,        y1=ymin, y2=ymin+yrange, facecolor='c',    alpha=0.3)
        ax.fill_between(Rcont,        y1=ymin, y2=ymin+yrange, facecolor='c',    alpha=0.3)
        ax.text(np.mean(Vcont)-0.89, 0.95*ymax, 'V continuum')
        ax.text(np.mean(Rcont)-0.89, 0.95*ymax, 'R continuum')
        ax.text(np.mean(caK)  -0.40, 0.95*ymax, 'K')
        ax.text(np.mean(caH)  -0.40, 0.95*ymax, 'H')
        ax.plot([396.847,396.847],[ymin,ymax],':',color ='black')
        ax.plot([393.366,393.366],[ymin,ymax],':',color ='black')
        ax.plot([chk_bandpass[0], chk_bandpass[1]],[0,0],':',color ='white')

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=200, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')

    def plot_HK_spectrum_1D_zoom(self, trace='sci', fig_path=None, show_plot=False):

        """
        Generate a 1D spectrum plot zoomed in on the C&K lines

        Args:
            trace (string) - 'sci' or 'sky' to select the fiber
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """

        orders = np.array(self.wave_lib.columns)
        padding = 1

        fig, (axk, axh) = plt.subplots(1,2, figsize=(12,5))

        color_grid = ['purple','darkblue','darkgreen','gold','darkorange','darkred']
        chk_bandpass  = [383.0, 401.0]#[384.0, 401.7]
        caK           = [393.3663-0.150, 393.3663+0.150]
        caH           = [396.8469-0.150, 396.8469+0.150]
        Vcont         = [389.9, 391.9]
        Rcont         = [397.4, 399.4]

        # Compute and Plot spectra
        specHK = np.zeros((len(orders), self.image.shape[1]), dtype=np.float64)
        for i in range(len(orders)):
            # This should be correct, but the sky and sci orders are flipped in the definition file
            if trace == 'sci':
                specHK[i,:] = np.sum(self.image[self.trace_location_sky[i]['x1']:self.trace_location_sky[i]['x2'],:],axis=0)
            elif trace == 'sky':
                specHK[i,:] = np.sum(self.image[self.trace_location_sci[i]['x1']:self.trace_location_sci[i]['x2'],:],axis=0)

        axh.set_xlim(np.mean(caH)-1.25, np.mean(caH)+1.25)
        for o in [4,5]:
            wav = self.wave_lib[o]
            axh.step(wav[padding:-padding],
                    specHK[o,padding:-padding],
                    color = color_grid[o],
                    linewidth = 2.0)
        axk.set_xlim(np.mean(caK)-1.25, np.mean(caK)+1.25)
        for o in [3]:
            wav = self.wave_lib[o]
            axk.step(wav[padding:-padding],
                    specHK[o,padding:-padding],
                    color = color_grid[o],
                    linewidth = 2.0)

        # Find y ranges
        indh = (self.wave_lib[5] > (np.mean(caH)-1.25)).values * (self.wave_lib[5] < (np.mean(caH)+1.25)).values
        indk = (self.wave_lib[3] > (np.mean(caK)-1.25)).values * (self.wave_lib[3] < (np.mean(caK)+1.25)).values
        yminh = min([0,np.nanpercentile(specHK[5,indh],0.1)])
        ymaxh = 1.15*np.nanpercentile(specHK[5,indh],99.9)
        yrangeh = (ymaxh-yminh)
        ymink = min([0,np.nanpercentile(specHK[3,indk],0.1)])
        ymaxk = 1.15*np.nanpercentile(specHK[3,indk],99.9)
        yrangek = (ymaxk-ymink)
        axh.set_ylim(yminh, ymaxh)
        axk.set_ylim(ymink, ymaxk)

        # Add labels
        axk.set_xlabel('Wavelength (nm)',fontsize=18)
        axk.set_ylabel('Flux (ADU)',fontsize=18)
        axk.xaxis.set_tick_params(labelsize=14)
        axk.yaxis.set_tick_params(labelsize=14)
        axh.set_xlabel('Wavelength (nm)',fontsize=18)
        axh.set_ylabel('Flux (ADU)',fontsize=18)
        axh.xaxis.set_tick_params(labelsize=14)
        axh.yaxis.set_tick_params(labelsize=14)
        axk.fill_between(chk_bandpass, y1=ymink, y2=ymink+yrangek, facecolor='gray', alpha=0.3, zorder=-100)
        axh.fill_between(chk_bandpass, y1=yminh, y2=yminh+yrangeh, facecolor='gray', alpha=0.3, zorder=-100)
        axh.fill_between(caH,          y1=yminh, y2=yminh+yrangeh, facecolor='m',    alpha=0.3)
        axk.fill_between(caK,          y1=ymink, y2=ymink+yrangek, facecolor='m',    alpha=0.3)
        axk.text(np.mean(caK)  -0.10, ymink+0.92*yrangek, 'K', fontsize=14)
        axh.text(np.mean(caH)  -0.10, yminh+0.92*yrangeh, 'H', fontsize=14)
        axh.plot([396.847,396.847],[yminh,ymaxh],':',color ='black')
        axk.plot([393.366,393.366],[ymink,ymaxk],':',color ='black')
        axk.plot([chk_bandpass[0], chk_bandpass[1]],[0,0],':',color ='white')
        axh.plot([chk_bandpass[0], chk_bandpass[1]],[0,0],':',color ='white')

        # Set y-axis to display in scientific notation
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1,1))
        axh.yaxis.set_major_formatter(formatter)
        axk.yaxis.set_major_formatter(formatter)

        # Add overall title to array of plots
        ax = fig.add_subplot(111, frame_on=False)
        ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        ax.set_title('Ca H&K ' + trace.upper() + ' Spectrum: ' + str(self.ObsID) + ' - ' + self.name + '\n', fontsize=18)
        ax.grid(False)

        # Display the plot
        if fig_path != None:
            t0 = time.process_time()
            plt.savefig(fig_path, dpi=200, facecolor='w')
            self.logger.info(f'Seconds to execute savefig: {(time.process_time()-t0):.1f}')
        if show_plot == True:
            plt.show()
        plt.close('all')

