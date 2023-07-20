import numpy as np
import matplotlib.pyplot as plt
#from astropy.table import Table
from modules.Utils.kpf_parse import HeaderParse

class AnalyzeEM:

    """
    Description:
        This class contains functions to analyze data from KPF's Ca H&K spectrometer
        (storing the results as attributes) and functions for plot the data products.
        Some of the functions need to be filled in.

    Arguments:
        L0 - an L0 object

    Attributes:
        TBD
    """

    def __init__(self, L0, logger=None):

        if logger:
            self.logger = logger
            self.logger.debug('Initializing AnalyzeHK object.')
        else:
            self.logger = None
        self.L0 = L0 
        primary_header = HeaderParse(L0, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        self.ObsID = primary_header.get_obsid()

    def plot_HK_spectrum_2D(self, fig_path=None, show_plot=False):

        """
        Generate a 2D image of the Ca H& K spectrum.  

        Args:
            fig_path (string) - set to the path for the file to be generated.
            show_plot (boolean) - show the plot in the current environment.

        Returns:
            PNG plot in fig_path or shows the plot it in the current environment 
            (e.g., in a Jupyter Notebook).

        """


##########
        if 'CA_HK' in hdulist and len(hdulist['CA_HK'].data)>=1:
            print('Working QLP for on Ca HK data')
            if not os.path.exists(output_dir+'/'+exposure_name+'/CaHK'):
                os.makedirs(output_dir+'/'+exposure_name+'/CaHK')

            def plot_trace_boxes(data,trace_location,trace_location_sky):

                fig, ax = plt.subplots(figsize = (12,6),tight_layout=True)
                im = ax.imshow(data,vmin = np.percentile(data.ravel(),1),vmax = np.percentile(data.ravel(),99), interpolation = 'None',origin = 'lower',aspect='auto')
                for i in trace_location.keys():
                    height = trace_location[i]['x2'] - trace_location[i]['x1']
                    width = trace_location[i]['y2'] - trace_location[i]['y1']
                    ax.add_patch(patches.Rectangle((trace_location[i]['y1'], trace_location[i]['x1']),width,height,linewidth=0.5, edgecolor='r',facecolor='none'))
                    if i == 0: ax.add_patch(patches.Rectangle((trace_location[i]['y1'], trace_location[i]['x1']),width,height,linewidth=0.5, edgecolor='r',facecolor='none',label = 'Sci (Saturation at '+str(64232)+')'))

                for i in trace_location_sky.keys():
                    height = trace_location_sky[i]['x2'] - trace_location_sky[i]['x1']
                    width = trace_location_sky[i]['y2'] - trace_location_sky[i]['y1']
                    ax.add_patch(patches.Rectangle((trace_location_sky[i]['y1'], trace_location_sky[i]['x1']),width,height,linewidth=0.5, edgecolor='white',facecolor='none'))
                    if i == 0: ax.add_patch(patches.Rectangle((trace_location_sky[i]['y1'], trace_location_sky[i]['x1']),width,height,linewidth=0.5, edgecolor='white',facecolor='none',label = 'Sky'))
                fig.colorbar(im, orientation='vertical',label = 'Counts (ADU)')
                plt.xlabel('y (pixel number)')
                plt.ylabel('x (pixel number)')
                plt.title('Ca H&K 2D '+exposure_name)#
                plt.legend()
                plt.savefig(output_dir+'/'+exposure_name+'/CaHK/'+exposure_name+'_CaHK_2D_zoomable.png', dpi=1000)
                plt.close()


            def load_trace_location(fiber,trace_path,offset=0):
                loc_result = pd.read_csv(trace_path,header =0, sep = ' ')
                #print(loc_result)
                loc_vals = np.array(loc_result.values)
                loc_cols = np.array(loc_result.columns)
                #print(loc_cols)
                order_col_name = 'order'
                fiber_col_name = 'fiber'
                loc_col_names = ['y0', 'x0', 'yf','xf']#['x0', 'y0', 'xf','yf']

                loc_idx = {c: np.where(loc_cols == c)[0][0] for c in loc_col_names}
                order_idx = np.where(loc_cols == order_col_name)[0][0]
                fiber_idx = np.where(loc_cols == fiber_col_name)[0][0]
                loc_for_fiber = loc_vals[np.where(loc_vals[:, fiber_idx] == fiber)[0], :]  # rows with the same fiber
                trace_location = dict()
                for loc in loc_for_fiber:       # add each row from loc_for_fiber to trace_location for fiber
                    trace_location[loc[order_idx]] = {'x1': loc[loc_idx['y0']]-offset,'x2': loc[loc_idx['yf']]-offset,'y1': loc[loc_idx['x0']],'y2': loc[loc_idx['xf']]}

                return trace_location


            trace_file = self.config['CaHK']['trace_file']
            trace_location = load_trace_location('sky',trace_file,offset=-1)
            trace_location_sky = load_trace_location('sci',trace_file,offset=-1)
            plot_trace_boxes(hdulist['ca_hk'].data,trace_location,trace_location_sky)
            
            def extract_HK_spectrum(data,trace_location,rv_shift,wavesoln ):

                wave_lib = pd.read_csv(wavesoln,header =None, sep = ' ',comment = '#')
                wave_lib*=1-rv_shift/3e5
                #print(trace_location)
                orders = np.array(wave_lib.columns)
                padding = 200

                plt.figure(figsize=(12,6),tight_layout=True)
                color_grid = ['purple','blue','green','yellow','orange','red']
                chk_bandpass  =  [384, 401.7]
                caK = [393.2,393.5]
                caH = [396.7,397.0]
                Vcont = [389.9,391.9]
                Rcont = [397.4,399.4]

                fig, ax = plt.subplots(1, 1, figsize=(9,4))
                ax.fill_between(chk_bandpass,y1=0,y2=1,facecolor='gray',alpha=0.3,zorder=-100)
                ax.fill_between(caH,y1=0,y2=1,facecolor='m',alpha=0.3)
                ax.fill_between(caK,y1=0,y2=1,facecolor='m',alpha=0.3)
                ax.fill_between(Vcont,y1=0,y2=1,facecolor='c',alpha=0.3)
                ax.fill_between(Rcont,y1=0,y2=1,facecolor='c',alpha=0.3)

                ax.text(np.mean(Vcont)-0.6,0.08,'V cont.')
                ax.text(np.mean(Rcont)-0.6,0.08,'R cont.')
                ax.text(np.mean(caK)-0.15,0.08,'K')
                ax.text(np.mean(caH)-0.15,0.08,'H')

                #ax.plot([chk_bandpass[0]-1, chk_bandpass[1]+1], [0.04,0.04],'k--',lw=0.7)
                #ax.text(385.1,0.041,'Requirement',fontsize=9)

                #ax.plot(x,t_all,label=label) instead iterate over spectral orders plottign
                ax.set_xlim(388,400)
                #ax.set_ylim(0,0.09)

                ax.set_xlabel('Wavelength (nm)',fontsize=10)
                ax.set_ylabel('Flux',fontsize=10)

                ax.plot([396.847,396.847],[0,1],':',color ='black')
                ax.plot([393.366,393.366],[0,1],':',color ='black')


                for i in range(len(orders)):
                    wav = wave_lib[i]
                    #print(i,trace_location[i]['x1'],trace_location[i]['x2'])
                    flux = np.sum(hdulist['ca_hk'].data[trace_location[i]['x1']:trace_location[i]['x2'],:],axis=0)
                    ax.plot(wav[padding:-padding],flux[padding:-padding]/np.percentile(flux[padding:-padding],99.9),color = color_grid[i],linewidth = 0.5)
                plt.title('Ca H&K Spectrum '+exposure_name)#
                plt.legend()
                plt.savefig(output_dir+'/'+exposure_name+'/CaHK/'+exposure_name+'_CaHK_Spectrum.png', dpi=1000)
                plt.close()
            #print(np.shape(hdulist['ca_hk'].data))
            rv_shift = hdulist[0].header['TARGRADV']
            extract_HK_spectrum(hdulist['ca_hk'].data,trace_location,rv_shift,wavesoln = self.config['CaHK']['cahk_wav'])


