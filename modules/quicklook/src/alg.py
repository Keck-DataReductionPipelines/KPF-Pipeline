import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0
from kpfpipe.models.level1 import KPF1
from keckdrpframework.models.arguments import Arguments
import traceback
import os
import pandas as pd
import glob
import math
from astropy import modeling
from astropy.time import Time
from datetime import datetime
from modules.Utils.analyze_l0 import AnalyzeL0
from modules.Utils.analyze_guider import AnalyzeGuider
from modules.Utils.analyze_em import AnalyzeEM
from modules.Utils.analyze_2d import Analyze2D
from modules.Utils.analyze_l1 import AnalyzeL1
from modules.Utils.analyze_l2 import AnalyzeL2
from modules.Utils.kpf_parse import HeaderParse
import kpfpipe.pipelines.fits_primitives as fits_primitives
from modules.Utils.kpf_parse import get_data_products_L0
from modules.Utils.kpf_parse import get_data_products_2D
from modules.Utils.kpf_parse import get_data_products_L1
from modules.Utils.kpf_parse import get_data_products_L2

class QuicklookAlg:
    """

    """

    def __init__(self,config=None,logger=None):

        """

        """
        self.config=config
        self.logger=logger

    def qlp_L0(self, kpf0, output_dir):
        """
        Description:
            Generates the standard quicklook data products for an L0 file.
    
        Arguments:
            kpf0 - an L0 filename
            output_dir - directory for output QLP files (if show_plot=False)
            show_plot - plots are generated inline (e.g., for Jupyter Notebooks) 
                        instead of saving files
    
        Attributes:
            None
        """
        
        primary_header = HeaderParse(kpf0, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        self.ObsID = primary_header.get_obsid()
        self.data_products = get_data_products_L0(kpf0)
        L0_QLP_file_base = output_dir + self.ObsID + '/'
        self.logger.info('Working on L0 QLP for ' + str(self.ObsID) + '.')
        self.logger.info('Data products found: ' + str(self.data_products))
        chips = []
        if 'Green' in self.data_products: chips.append('green')
        if 'Red'   in self.data_products: chips.append('red')

        # Make Exposure Meter images
        if 'ExpMeter' in self.data_products:
            try:
                savedir = L0_QLP_file_base +'EM/'
                os.makedirs(savedir, exist_ok=True) # make directories if needed
    
                # Exposure Meter spectrum plot
                myEM = AnalyzeEM(kpf0, logger=self.logger)
                filename = savedir + self.ObsID + '_EM_spectrum_zoomable.png'
                self.logger.info('Generating QLP image ' + filename)
                myEM.plot_EM_spectrum(fig_path=filename, show_plot=False)
    
                # Exposure Meter time series plot
                filename = savedir + self.ObsID + '_EM_time_series_zoomable.png'
                self.logger.info('Generating QLP image ' + filename)
                myEM.plot_EM_time_series(fig_path=filename, show_plot=False)

            except Exception as e:
                self.logger.error(f"Failure in Exposure Meter quicklook pipeline: {e}\n{traceback.format_exc()}")

        # Make CaHK plots#
        if 'HK' in self.data_products:    
            try:    
                savedir = L0_QLP_file_base +'HK/'    
                os.makedirs(savedir, exist_ok=True) # make directories if needed    
        
                # Exposure Meter spectrum plot    
                trace_file = self.config['CaHK']['trace_file']    
                wavesoln_file = self.config['CaHK']['cahk_wav']    
                myHK = AnalyzeHK(kpf0, trace_file = trace_file,     
                                       wavesoln_file = wavesoln_file,     
                                       logger=self.logger)    
                filename = savedir + self.ObsID + '_HK_image_zoomable.png'    
                self.logger.info('Generating QLP image ' + filename)    
                myHK.plot_HK_image_2D(fig_path=filename, show_plot=False)    
        
                # Exposure Meter time series plot    
                filename = savedir + self.ObsID + '_HK_spectrum_zoomable.png'    
                self.logger.info('Generating QLP image ' + filename)    
                myHK.plot_HK_spectrum_1D(fig_path=filename, show_plot=False)    

            except Exception as e:    
                self.logger.error(f"Failure in CaHK quicklook pipeline: {e}\n{traceback.format_exc()}")

        # Make stitched L0 images
        try:
            myL0 = AnalyzeL0(kpf0, logger=self.logger)
            for chip in chips:
                savedir = L0_QLP_file_base +'L0/'
                os.makedirs(savedir, exist_ok=True) # make directories if needed
                filename = savedir + self.ObsID + '_L0_stitched_image_' + chip + '_zoomable.png'
                self.logger.info('Generating QLP image ' + filename)
                myL0.plot_L0_stitched_image(fig_path=filename, 
                                            chip=chip, show_plot=False)
        except Exception as e:
            self.logger.error(f"Failure in L0 quicklook pipeline: {e}\n{traceback.format_exc()}")


    def qlp_2D(self, kpf2d, output_dir):
        """
        Description:
            Generates the standard quicklook data products for a 2D file.
    
        Arguments:
            kpf2d - a 2d filename
            output_dir - directory for output QLP files (if show_plot=False)
            show_plot - plots are generated inline (e.g., for Jupyter Notebooks) 
                        instead of saving files
    
        Attributes:
            None
        """
        
        primary_header = HeaderParse(kpf2d, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        self.ObsID = primary_header.get_obsid()
        self.data_products = get_data_products_2D(kpf2d)
        self.D2 = kpf2d
        
        D2_QLP_file_base = output_dir + self.ObsID + '/'
        self.logger.info('Working on 2D QLP for ' + str(self.ObsID) + '.')
        self.logger.info('Data products found: ' + str(self.data_products))
        chips = []
        if 'Green' in self.data_products: chips.append('green')
        if 'Red'   in self.data_products: chips.append('red')

        # Make Guider images
        if 'Guider' in self.data_products:
            try:
                savedir = D2_QLP_file_base +'Guider/'
                os.makedirs(savedir, exist_ok=True) # make directories if needed
                myGuider = AnalyzeGuider(kpf2d, logger=self.logger)
                myGuider.measure_seeing()

                # Guider image plot
                filename = savedir + self.ObsID + '_guider_image_zoomable.png'
                self.logger.info('Generating QLP image ' + filename)
                myGuider.plot_guider_image(fig_path=filename, show_plot=False)

                # Guider error time series
                filename = savedir + self.ObsID + '_error_time_series_zoomable.png'
                self.logger.info('Generating QLP image ' + filename)
                myGuider.plot_guider_error_time_series(fig_path=filename, show_plot=False)
                
                # Guider flux time series
                filename = savedir + self.ObsID + '_flux_time_series_zoomable.png'
                self.logger.info('Generating QLP image ' + filename)
                myGuider.plot_guider_flux_time_series(fig_path=filename, show_plot=False)
                
                # Guider FWHM time series
                filename = savedir + self.ObsID + '_fwhm_time_series_zoomable.png'
                self.logger.info('Generating QLP image ' + filename)
                myGuider.plot_guider_fwhm_time_series(fig_path=filename, show_plot=False)

            except Exception as e:
                self.logger.error(f"Failure in Guider quicklook pipeline: {e}\n{traceback.format_exc()}")

        # Make 2D images
        # to-do: process bias and dark differently
        if ('Green' in self.data_products) or ('Red' in self.data_products):
            try:
                savedir = D2_QLP_file_base +'2D/'
                os.makedirs(savedir, exist_ok=True) # make directories if needed
                my_2D = AnalyzeL0(kpf2d, logger=self.logger)    
                for chip in chips:
                    # next line not working yet
                    #Analyze2D.measure_2D_dark_current(self, chip=chip)
                    filename = savedir + self.ObsID + '_2D_image_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    Analyze2D.plot_2D_image(self, chip=chip, fig_path=filename, 
                                                  show_plot=False)

            except Exception as e:
                self.logger.error(f"Failure in 2D quicklook pipeline: {e}\n{traceback.format_exc()}")


        # Make 2D images - 3x3 arrays
        if ('Green' in self.data_products) or ('Red' in self.data_products):
            try:
                savedir = D2_QLP_file_base +'2D_Analysis/'
                os.makedirs(savedir, exist_ok=True) # make directories if needed
                my_2D = Analyze2D(kpf2d, logger=self.logger)
                for chip in chips:
                    filename = savedir + self.ObsID + '_2D_image_3x3zoom_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    Analyze2D.plot_2D_image_zoom_3x3(self, chip=chip, fig_path=filename, 
                                                           show_plot=False)

            except Exception as e:
                self.logger.error(f"Failure in 2D quicklook pipeline: {e}\n{traceback.format_exc()}")


        # TO-DO Add bias histogram


        # Make 2D image histograms
        if ('Green' in self.data_products) or ('Red' in self.data_products):
            try:
                my_2D = Analyze2D(kpf2d, logger=self.logger)
                for chip in chips:
                    filename = savedir + self.ObsID + '_2D_histogram_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    Analyze2D.plot_2D_image_histogram(self, chip=chip, fig_path=filename, 
                                                            show_plot=False)

            except Exception as e:
                self.logger.error(f"Failure in 2D quicklook pipeline: {e}\n{traceback.format_exc()}")

        # Make 2D column cuts
        if ('Green' in self.data_products) or ('Red' in self.data_products):
            try:
                for chip in chips:
                    filename = savedir + self.ObsID + '_2D_column_cut_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    Analyze2D.plot_2D_column_cut(self, chip=chip, fig_path=filename, 
                                                       show_plot=False)

            except Exception as e:
                self.logger.error(f"Failure in 2D quicklook pipeline: {e}\n{traceback.format_exc()}")
        
        
    def qlp_L1(self, kpf1, output_dir):
        """
        Description:
            Generates the standard quicklook data products for an L1 file.
    
        Arguments:
            kpf1 - an L1 filename
            output_dir - directory for output QLP files (if show_plot=False)
            show_plot - plots are generated inline (e.g., for Jupyter Notebooks) 
                        instead of saving files
    
        Attributes:
            None
        """
        
        primary_header = HeaderParse(kpf1, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        self.ObsID = primary_header.get_obsid()        
        self.data_products = get_data_products_L1(kpf1)
        self.logger.info('Data products found: ' + str(self.data_products))
        chips = []
        if 'Green' in self.data_products: chips.append('green')
        if 'Red'   in self.data_products: chips.append('red')

        L1_QLP_file_base = output_dir + self.ObsID + '/'
        self.logger.info('Working on L1 QLP for ' + str(self.ObsID) + '.')

        # Make L1 SNR plot
        try:
            savedir = L1_QLP_file_base +'L1/'
            os.makedirs(savedir, exist_ok=True) # make directories if needed
            filename = savedir + self.ObsID + '_2D_L1_SNR_zoomable.png'
            self.logger.info('Generating QLP image ' + filename)
            myL1 = AnalyzeL1(kpf1, logger=self.logger)
            myL1.measure_L1_snr()
            myL1.plot_L1_snr(fig_path=filename, show_plot=False)

        except Exception as e:
            self.logger.error(f"Failure in L1 quicklook pipeline: {e}\n{traceback.format_exc()}")

        # Make L1 spectra plots
        try:
            for oo, orderlet in enumerate(['SCI1', 'SCI2', 'SCI3', 'CAL', 'SKY']):
                filename = savedir + self.ObsID + '_L1_spectrum_' + orderlet + '_zoomable.png'
                self.logger.info('Generating QLP image ' + filename)
                myL1.plot_L1_spectrum(orderlet=orderlet, fig_path=filename, show_plot=False)

        except Exception as e:
            self.logger.error(f"Failure in L1 quicklook pipeline: {e}\n{traceback.format_exc()}")

        # Make L1 spectra plots (single order)
        if ('Green' in self.data_products) or ('Red' in self.data_products):
            try:
                myL1 = AnalyzeL1(kpf1, logger=self.logger)
                for chip in chips:
                    if chip == 'green': order = 11
                    if chip == 'red': order = 11
                    filename = savedir + self.ObsID + '_L1_spectrum_SCI_order11_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    myL1.plot_1D_spectrum_single_order(chip=chip, order=11, ylog=False, 
                                                       orderlet=['SCI1', 'SCI2', 'SCI3'], 
                                                       fig_path=filename, show_plot=False)
                    filename = savedir + self.ObsID + '_L1_spectrum_SKY_order11_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    myL1.plot_1D_spectrum_single_order(chip=chip, order=11, ylog=False, 
                                                       orderlet=['SKY'], 
                                                       fig_path=filename, show_plot=False)
                    filename = savedir + self.ObsID + '_L1_spectrum_CAL_order11_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    myL1.plot_1D_spectrum_single_order(chip=chip, order=11, ylog=False, 
                                                       orderlet=['CAL'], 
                                                       fig_path=filename, show_plot=False)

            except Exception as e:
                self.logger.error(f"Failure in L1 quicklook pipeline: {e}\n{traceback.format_exc()}")


    def qlp_L2(self, kpf2, output_dir):
        """
        Description:
            Generates the standard quicklook data products for an L2 file.
    
        Arguments:
            kpf2 - an L0 filename
            output_dir - directory for output QLP files (if show_plot=False)
            show_plot - plots are generated inline (e.g., for Jupyter Notebooks) 
                        instead of saving files
    
        Attributes:
            None
        """

        primary_header = HeaderParse(kpf2, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        self.ObsID = primary_header.get_obsid()
        self.data_products = get_data_products_L2(kpf2)
        L2_QLP_file_base = output_dir + self.ObsID + '/'
        self.logger.info('Working on L2 QLP for ' + str(self.ObsID) + '.')
        self.logger.info('Data products found: ' + str(self.data_products))
        chips = []
        if 'Green' in self.data_products: chips.append('green')
        if 'Red'   in self.data_products: chips.append('red')

        # Make CCF grid plots
        if ('Green' in self.data_products) or ('Red' in self.data_products):
            try:
                myL2 = AnalyzeL2(kpf2, logger=self.logger)
                for chip in chips:
                    savedir = L2_QLP_file_base +'L2/'
                    os.makedirs(savedir, exist_ok=True) # make directories if needed
                    filename = savedir + self.ObsID + '_CCF_grid_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    myL2.plot_CCF_grid(chip=chip, fig_path=filename, show_plot=False)

            except Exception as e:
                self.logger.error(f"Failure in CCF quicklook pipeline: {e}\n{traceback.format_exc()}")


###### Below is the old QLP code, partially deconstructed.                #######
###### There are some notes about elements to put in to analysis modules. #######
    def qlp_procedures(self,kpf0_file,output_dir,end_of_night_summary):

        saturation_limit = int(self.config['2D']['saturation_limit'])*1.
        plt.rcParams.update({'font.size': 8})
        plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']

        #check if output locations exist, if not create them
        exposure_name = kpf0_file.filename.replace('_2D.fits', '.fits')[:-5]
        date = exposure_name[3:11]
        print('test',exposure_name, date)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        #if not os.path.exists(output_dir+'/fig'):
        #    os.makedirs(output_dir+'/fig')

        if not os.path.exists(output_dir+'/'+exposure_name+'/2D'):
            os.makedirs(output_dir+'/'+exposure_name+'/2D')

        if not os.path.exists(output_dir+'/'+exposure_name+'/2D_analysis'):
            os.makedirs(output_dir+'/'+exposure_name+'/2D_analysis')

        if not os.path.exists(output_dir+'/'+exposure_name+'/L1'):
            os.makedirs(output_dir+'/'+exposure_name+'/L1')

        if not os.path.exists(output_dir+'/'+exposure_name+'/CCF'):
            os.makedirs(output_dir+'/'+exposure_name+'/CCF')
        #print('working on',file_name)

        # try:
        #     exposure_name = kpf0_file.header['PRIMARY']['OFNAME'][:-5]#file_name[18:-5]#hdr['PRIMARY']['OFNAME'][:-5]
        # except:


        if end_of_night_summary == True:
            print('Working on end of night summary of '+date)
            file_list = glob.glob(self.config['IO']['input_prefix_l0']+date+'/*.fits')
            #print(len(file_list),file_list)

            '''
            #pull temps from all the fits header and draw the temps
            date_obs = []
            temp = []
            for file in file_list:
                print(file)
                hdulist = fits.open(file)
                #print(hdulist.info())
                hdr = hdulist[0].header

                date_obs.append(hdr['DATE-OBS'])
                temp.append(hdr['TEMP'])
            date_obs = np.array(date_obs,'str')
            date_obs = Time(date_obs, format='isot', scale='utc')
            plt.scatter(date_obs.jd,temp, marker = '.')
            plt.xlabel('Time')
            plt.ylabel('Temperature')
            plt.savefig(output_dir+'fig/end_of_night_summary_temperature.png')
            plt.close()


            #plot how the order trace changes
            order_trace_master_file = '/data/order_trace/20220601/KP.20220601.37734.61_GREEN_CCD.csv'
            order_trace_list = glob.glob('/data/order_trace/*/*_GREEN_CCD.csv', recursive = True)
            for order_trace_file in order_trace_list:
                order_trace = pd.read_csv(order_trace_file)
                order_trace_master = pd.read_csv(order_trace_master_file)
                print(np.shape(order_trace),np.shape(order_trace_master))
                for i in range(1,np.shape(order_trace)[0]-2,1):#[50]:#range(np.shape(order_trace)[0])
                    x_grid_master = np.linspace(order_trace_master.iloc[i]['X1'],order_trace_master.iloc[i]['X2'],int(order_trace_master.iloc[i]['X2']-order_trace_master.iloc[i]['X1']))
                    y_grid_master = order_trace_master.iloc[i]['Coeff0']+x_grid_master*order_trace_master.iloc[i]['Coeff1']+x_grid_master**2*order_trace_master.iloc[i]['Coeff2']+x_grid_master**3*order_trace_master.iloc[i]['Coeff3']

                    x_grid = np.linspace(order_trace.iloc[i]['X1'],order_trace.iloc[i]['X2'],int(order_trace.iloc[i]['X2']-order_trace.iloc[i]['X1']))
                    y_grid = order_trace.iloc[i]['Coeff0']+x_grid*order_trace.iloc[i]['Coeff1']+x_grid**2*order_trace.iloc[i]['Coeff2']+x_grid**3*order_trace.iloc[i]['Coeff3']

                    print(order_trace_file,'order',i,np.nanstd(y_grid_master-y_grid),abs(order_trace.iloc[i]['BottomEdge']-order_trace_master.iloc[i]['BottomEdge']),abs(order_trace.iloc[i]['TopEdge']-order_trace_master.iloc[i]['TopEdge']))

                    plt.plot(x_grid,y_grid,color ='red',linewidth = 0.2)
                    plt.plot(x_grid,y_grid-order_trace.iloc[i]['BottomEdge'],color ='white',linewidth = 0.2,alpha = 1)
                    plt.plot(x_grid,y_grid+order_trace.iloc[i]['TopEdge'],color ='black',linewidth = 0.2,alpha = 1)
            plt.xlim(3200,4000)
            plt.ylim(3200,4000)
            plt.savefig(output_dir+'fig/order_trace_evolution.png')
            '''
            return
        print('Working on',date,exposure_name)

        #### L0 ####
        L0_filename = self.config['IO']['input_prefix_l0_pre']+date+'/'+exposure_name+'.fits'
        L0 = fits.open(L0_filename)

        #L0_kpf = fits_primitives.kpf0_from_fits(L0_filename)
        my_AnalyzeL0 = AnalyzeL0(L0)
        if os.path.exists(output_dir+'/'+exposure_name+'/L0/') == False: os.makedirs(output_dir+'/'+exposure_name+'/L0/')
        # temporarily comment out (for speed)
        my_AnalyzeL0.plot_L0_stitched_image(exposure_name,chip='green', fig_path=output_dir+'/'+exposure_name+'/L0/'+exposure_name+'_GREEN_L0_zoomable.png', show_plot=False)
        my_AnalyzeL0.plot_L0_stitched_image(exposure_name,chip='red', fig_path=output_dir+'/'+exposure_name+'/L0/'+exposure_name+'_RED_L0_zoomable.png', show_plot=False)

        #read ccd directly
        L0_data = self.config['IO']['input_prefix_l0']+date+'/'+exposure_name+'_2D.fits'
        hdulist = fits.open(L0_data)
        #print(hdulist.info())

        # Get CCD names
        ccd_color=[]
        ccd_list = self.config.items( "CCD_LIST")
        for key, path in ccd_list:
            ccd_color.append(path)

        if len(hdulist[ccd_color[0]].data)<1 and len(hdulist[ccd_color[1]].data)<1:
            print('skipping empty file')
            return

        #hdr = hdulist.header
        #version = hdr['PRIMARY']['IMTYPE']
        hdr = hdulist[0].header
        version = hdr['IMTYPE']
        Cal_Source = hdr['SCI-OBJ']
        #print('2d header',hdr,hdr['IMTYPE'],hdr['CAL-OBJ'],hdr['SCI-OBJ'],hdr['SKY-OBJ'])


        master_file = 'None'
        if version == 'Solar':
            master_file = self.config['2D']['master_solar']
        if version == 'Arclamp':
            if Cal_Source == 'Th_daily':
                master_file = '/data/masters/'+date+'/kpf_'+date+'_master_arclamp_autocal-thar-all-night.fits' #self.config['2D']['master_arclamp']
                if os.path.exists(master_file) == False: master_file = self.config['2D']['master_Th_daily']
            if Cal_Source == 'U_daily':
                master_file = '/data/masters/'+date+'/kpf_'+date+'_master_arclamp_autocal-une-all-eve.fits' #self.config['2D']['master_arclamp']
                if os.path.exists(master_file) == False: master_file = self.config['2D']['master_U_daily']
            if Cal_Source == 'EtalonFiber':
                master_file = '/data/masters/'+date+'/kpf_'+date+'_master_arclamp_autocal-etalon-all-eve.fits' #self.config['2D']['master_arclamp']
                if os.path.exists(master_file) == False: master_file = self.config['2D']['master_EtalonFiber']
            if Cal_Source == 'LFCFiber':
                master_file = '/data/masters/'+date+'/kpf_'+date+'_master_arclamp_autocal-lfc-all-eve.fits' #self.config['2D']['master_arclamp']
                if os.path.exists(master_file) == False: master_file = self.config['2D']['master_LFCFiber']
        if version == 'Flatlamp':
            master_file = '/data/masters/'+date+'/kpf_'+date+'_master_flat.fits' #
            if os.path.exists(master_file) == False: master_file = self.config['2D']['master_flatlamp']
        if version == 'Dark':
            master_file = '/data/masters/'+date+'/kpf_'+date+'_master_dark.fits' #
            if os.path.exists(master_file) == False: master_file = self.config['2D']['master_dark']
        if version == 'Bias':
            master_file = '/data/masters/'+date+'/kpf_'+date+'_master_bias.fits' #
            if os.path.exists(master_file) == False: master_file = self.config['2D']['master_bias']
        if version == 'Sol_All':
            master_file = self.config['2D']['master_socal']
        if version == 'Etalon_All':
            master_file = self.config['2D']['master_etalon']
        if version == 'Th_All':
            master_file = self.config['2D']['master_ThAr']
        if version == 'Une_All':
            master_file = self.config['2D']['master_Une']
        if version == 'UNe_All':
            master_file = self.config['2D']['master_Une']
        if version == 'LFC_SciCal':
            master_file = self.config['2D']['master_LFC']


        for i_color in range(len(ccd_color)):
            counts = np.array(hdulist[ccd_color[i_color]].data,'d')
            flatten_counts = np.ravel(counts)
            if len(flatten_counts)<1: continue
            master_flatten_counts='None'
            if master_file != 'None':
                hdulist1 = fits.open(master_file)
                master_counts = np.array(hdulist1[ccd_color[i_color]].data,'d')
                master_flatten_counts = np.ravel(master_counts)
                if version == 'Dark':#scale up dark exposures
                    master_flatten_counts*=hdr['EXPTIME']
            #print(version,hdr,hdr['EXPTIME'],type(hdr['EXPTIME']),hdulist1[0].header['EXPTIME'],Cal_Source,master_file,os.path.exists(master_file))
            #print(master_counts)
            #input("Press Enter to continue...")

            #looking at the fixed noise patterns
            '''
            if version =='Bias':
                a = np.copy(counts)
                a_med = np.nanmedian(a.ravel())
                a_std = np.nanstd(a.ravel())
                pdf, bin_edges = np.histogram(a.ravel(),bins=80, range = (a_med-8*a_std,a_med+8*a_std))
                print('bin edges',bin_edges)
                count_fit = (bin_edges[1:]+bin_edges[:-1])/2
                from astropy import modeling
                fitter = modeling.fitting.LevMarLSQFitter()#the gaussian fit of the ccf
                model = modeling.models.Gaussian1D()
                fitted_model = fitter(model,count_fit-a_med, pdf)
                amp =fitted_model.amplitude.value
                gamma =fitted_model.mean.value+a_med
                std =fitted_model.stddev.value
                print('1st gaussian',amp,gamma,std)
                plt.close('all')
                plt.plot(count_fit,amp*np.exp(-0.5*(count_fit-gamma)**2/std**2),':',color = 'red', label = '1st component')#1/std/np.sqrt(2*np.pi)*
                #plt.ylim(1,10*amp)
                fitter = modeling.fitting.LevMarLSQFitter()#the gaussian fit of the ccf
                model = modeling.models.Gaussian1D()
                fitted_model = fitter(model,count_fit[(count_fit<gamma-1*std) | (count_fit>gamma+1*std)]-a_med, pdf[(count_fit<gamma-1*std) | (count_fit>gamma+1*std)])
                amp2 =fitted_model.amplitude.value
                gamma2 =fitted_model.mean.value+a_med
                std2 =fitted_model.stddev.value
                print('2nd gaussian',amp2,gamma2,std2)

                plt.plot(count_fit,amp2*np.exp(-0.5*(count_fit-gamma2)**2/std2**2),':',color = 'green', label = '2nd component')#1/std/np.sqrt(2*np.pi)*
                #plt.ylim(1,10**7)
                plt.plot((bin_edges[1:]+bin_edges[:-1])/2,pdf, label = 'All')
                plt.scatter(np.array((bin_edges[1:]+bin_edges[:-1])/2,'d')[(count_fit<gamma-1*std) | (count_fit>gamma+1*std)],pdf[(count_fit<gamma-1*std) | (count_fit>gamma+1*std)], label = 'Larger Var Component')
                plt.legend()
                plt.xlabel('Counts (e-)')
                plt.ylabel('Number of Pixels')
                plt.yscale('log')
                plt.savefig(output_dir+'fig/'+exposure_name+'_bias_'+ccd_color[i_color]+'.png')
                plt.close('all')
            '''



            plot_2D_image(self, chip=None, overplot_dark_current=False, 
                            fig_path=None, show_plot=False)

            #2D image
            plt.figure(figsize=(5,4))
            plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
            plt.imshow(counts, vmin = np.percentile(flatten_counts,1),vmax = np.percentile(flatten_counts,99),interpolation = 'None',origin = 'lower')
            plt.xlabel('x (pixel number)')
            plt.ylabel('y (pixel number)')
            plt.title(ccd_color[i_color]+' '+version +' '+exposure_name)
            plt.colorbar(label = 'Counts (e-)')
            #plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Frame_'+ccd_color[i_color]+'.png')
            plt.savefig(output_dir+'/'+exposure_name+'/2D/'+exposure_name+'_2D_Frame_'+ccd_color[i_color]+'_zoomable.png', dpi=1000)
            #plt.close()


# commented out on July 16 -- determine how to add to QLP
#                plt.figure(figsize=(5,4))
#                plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
#                threshold = 2
#                high_var_counts = np.copy(counts)
#                low_var_counts = np.copy(counts)
#                high_var_counts[abs(counts-np.nanmedian(counts))<threshold*np.nanstd(counts)] = np.nan
#                low_var_counts[abs(counts-np.nanmedian(counts))>threshold*np.nanstd(counts)] = np.nan
#
#                plt.imshow(high_var_counts, vmin = np.percentile(flatten_counts,0.1),vmax = np.percentile(flatten_counts,99.9),interpolation = 'None',origin = 'lower',cmap = 'bwr')
#                plt.xlabel('x (pixel number)')
#                plt.ylabel('y (pixel number)')
#                plt.title(ccd_color[i_color]+' '+version+' High Variance '+exposure_name)
#                plt.colorbar(label = 'Counts (e-)')
#
#                plt.text(2200,3600, 'Nominal STD: %5.1f' % np.nanstd(np.ravel(low_var_counts)))
#                plt.text(2200,3300, 'Fixed Pattern STD: %5.1f' % np.nanstd(np.ravel(high_var_counts)))
#                #plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Frame_high_var_'+ccd_color[i_color]+'_zoomable.png')
#                plt.savefig(output_dir+'/'+exposure_name+'/2D_analysis/'+exposure_name+'_2D_Frame_high_var_'+ccd_color[i_color]+'_zoomable.png', dpi=1000)
#                plt.close()
#                '''
#                plt.figure(figsize=(5,4))
#                plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
#
#                plt.imshow(low_var_counts, vmin = np.percentile(flatten_counts,1),vmax = np.percentile(flatten_counts,99),interpolation = 'None',origin = 'lower')
#                plt.xlabel('x (pixel number)')
#                plt.ylabel('y (pixel number)')
#                plt.title(ccd_color[i_color]+' '+version+' Low Variance')
#                plt.colorbar(label = 'Counts (e-)')
#                plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Frame_low_var_'+ccd_color[i_color]+'_zoomable.png')
#                '''
#            #print('master file',version,i_color,master_file,len(master_flatten_counts))
#            if master_file != 'None' and len(master_flatten_counts)>1:
#                plt.figure(figsize=(5,4))
#                plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
#                #pcrint(counts,master_counts)
#                counts_norm = np.percentile(counts,99)
#                master_counts_norm = np.percentile(master_counts,99)
#                if np.shape(counts)!=np.shape(master_counts): continue
#                difference = counts/counts_norm-master_counts/master_counts_norm
#
#                plt.imshow(difference, vmin = np.percentile(difference,1),vmax = np.percentile(difference,99), interpolation = 'None',origin = 'lower')
#                plt.xlabel('x (pixel number)')
#                plt.ylabel('y (pixel number)')
#                plt.title(ccd_color[i_color]+' '+version+'- Master '+version+' '+exposure_name)
#                plt.colorbar(label = 'Fractional Difference')
#                #plt.savefig(output_dir+'fig/'+exposure_name+'_2D_Difference_'+ccd_color[i_color]+'_zoomable.png')
#                plt.savefig(output_dir+'/'+exposure_name+'/2D_analysis/'+exposure_name+'_2D_Difference_'+ccd_color[i_color]+'_zoomable.png', dpi=1000)
#             #Hisogram
#            plt.close()
#            plt.figure(figsize=(5,4))
#            plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
#
#            #print(np.percentile(flatten_counts,99.9),saturation_limit)
#            plt.hist(flatten_counts, bins = 50,alpha =0.5, label = 'Median: ' + '%4.1f; ' % np.nanmedian(flatten_counts)+'; Std: ' + '%4.1f' % np.nanstd(flatten_counts)+'; Saturated? '+str(np.percentile(flatten_counts,99.99)>saturation_limit),density = False, range = (np.percentile(flatten_counts,0.005),np.percentile(flatten_counts,99.995)))#[flatten_counts<np.percentile(flatten_counts,99.9)]
#            if master_file != 'None' and len(master_flatten_counts)>1: plt.hist(master_flatten_counts, bins = 50,alpha =0.5, label = 'Master Median: '+ '%4.1f' % np.nanmedian(master_flatten_counts)+'; Std: ' + '%4.1f' % np.nanstd(master_flatten_counts), histtype='step',density = False, color = 'orange', linewidth = 1 , range = (np.percentile(master_flatten_counts,0.005),np.percentile(master_flatten_counts,99.995))) #[master_flatten_counts<np.percentile(master_flatten_counts,99.9)]
#            #plt.text(0.1,0.2,np.nanmedian(flatten_counts))
#            plt.xlabel('Counts (e-)')
#            plt.ylabel('Number of Pixels')
#            plt.yscale('log')
#            plt.title(ccd_color[i_color]+' '+version+' Histogram '+exposure_name)
#            plt.legend(loc='lower right')
#            #plt.savefig(output_dir+'fig/'+exposure_name+'_Histogram_'+ccd_color[i_color]+'.png')
#            plt.savefig(output_dir+'/'+exposure_name+'/2D_analysis/'+exposure_name+'_Histogram_'+ccd_color[i_color]+'.png', dpi=200)
#
#            #Column cut
#            plt.close()
#            plt.figure(figsize=(8,4))
#            plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9)
#
#            column_sum = np.nansum(counts,axis = 0)
#            #print('which_column',np.where(column_sum==np.nanmax(column_sum))[0][0])
#            which_column = np.where(column_sum==np.nanmax(column_sum))[0][0] #int(np.shape(master_counts)[1]/2)
#
#            plt.plot(np.ones_like(counts[:,which_column])*saturation_limit,':',alpha = 0.5,linewidth =  1., label = 'Saturation Limit: '+str(saturation_limit), color = 'gray')
#            plt.plot(counts[:,which_column],alpha = 0.5,linewidth =  0.5, label = ccd_color[i_color]+' '+version, color = 'Blue')
#            if master_file != 'None' and len(master_flatten_counts)>1: plt.plot(master_counts[:,which_column],alpha = 0.5,linewidth =  0.5, label = 'Master', color = 'Orange')
#            plt.yscale('log')
#            plt.ylabel('log(Counts/e-)')
#            plt.xlabel('Row Number')
#            plt.title(ccd_color[i_color]+' '+version+' Column Cut Through Column '+str(which_column) + ' '+exposure_name)#(Middle of CCD)
#            plt.ylim(1,1.2*np.nanmax(counts[:,which_column]))
#            plt.legend()
#            '''
#            #show the order order_trace
#            if version == 'Flat_All':
#                 for i in range(np.shape(order_trace)[0]):#[50]:#range(np.shape(order_trace)[0])
#                     #print(order_trace.iloc[i]['X1'],int(order_trace.iloc[i]['X2']-order_trace.iloc[i]['X1']))
#                     x_grid = np.linspace(0,order_trace.iloc[i]['X2'],int(order_trace.iloc[i]['X2']-0))
#                     y_grid = order_trace.iloc[i]['Coeff0']+x_grid*order_trace.iloc[i]['Coeff1']+x_grid**2*order_trace.iloc[i]['Coeff2']+x_grid**3*order_trace.iloc[i]['Coeff3']
#                     #print(i,len(y_grid),which_column,y_grid[which_column])
#                     #plt.plot([y_grid[which_column],y_grid[which_column]],[1,1.*np.nanmax(counts[:,which_column])],color ='red',linewidth = 0.2)
#                     plt.plot([y_grid[which_column]-order_trace.iloc[i]['BottomEdge'],y_grid[which_column]-order_trace.iloc[i]['BottomEdge']],[1,1.*np.nanmax(counts[:,which_column])],color ='red',linewidth = 0.2)
#                     plt.plot([y_grid[which_column]+order_trace.iloc[i]['TopEdge'],y_grid[which_column]+order_trace.iloc[i]['TopEdge']],[1,1.*np.nanmax(counts[:,which_column])],color ='magenta',linewidth = 0.2)
#                     #plt.plot(x_grid[which_column],y_grid[which_column]-order_trace.iloc[i]['BottomEdge'],color ='white',linewidth = 0.2,alpha = 1)
#                     #plt.plot(x_grid[which_column],y_grid[which_column]+order_trace.iloc[i]['TopEdge'],color ='black',linewidth = 0.2,alpha = 1)
#            '''
#            #plt.savefig(output_dir+'fig/'+exposure_name+'_Column_cut_'+ccd_color[i_color]+'.png')
#            plt.savefig(output_dir+'/'+exposure_name+'/2D_analysis/'+exposure_name+'_Column_cut_'+ccd_color[i_color]+'_zoomable.png', dpi=200)
#            plt.close()


        
        #### L1 ####
        L1_filename = self.config['IO']['input_prefix_l1']+date+'/'+exposure_name+'_L1.fits'
        if os.path.exists(L1_filename):
            print('Working on', L1_filename)
            hdulist = fits.open(L1_filename)
            try: 
                if not os.path.exists(output_dir+'/'+exposure_name+'/L1'):
                    os.makedirs(output_dir+'/'+exposure_name+'/L1')
                L1 = KPF1.from_fits(L1_filename)
                my_L1 = AnalyzeL1(L1)
                my_L1.measure_L1_snr()
                my_L1.plot_L1_snr(exposure_name,fig_path=output_dir+'/'+exposure_name+'/L1/'+exposure_name+'_L1_spectrum_snr.png')
            except:
                print("Processing QLP for L1 failed")


            wav_green = np.array(hdulist['GREEN_CAL_WAVE'].data,'d')
            wav_red = np.array(hdulist['RED_CAL_WAVE'].data,'d')
            #print('test wav_green',wav_green)
            '''
            wave_soln = self.config['L1']['wave_soln']
            if wave_soln!='None':#use the master the wavelength solution
                hdulist1 = fits.open(wave_soln)
                wav_green = np.array(hdulist1['GREEN_CAL_WAVE'].data,'d')
                wav_red = np.array(hdulist1['RED_CAL_WAVE'].data,'d')
                hdulist1.close()
            '''
=
            #make a comparison plot of the three science fibres
            plt.close()
            plt.figure(figsize=(10,4))
            plt.subplots_adjust(left=0.2, bottom=0.15, right=0.9, top=0.9)
            for i_orderlet in [1,2,3]:
                flux_tmp = np.array(hdulist['GREEN_SCI_FLUX'+str(i_orderlet)].data,'d')
                if np.shape(flux_tmp)==(0,): continue
                plt.plot(wav_green[10,:],flux_tmp[10,:], label = 'GREEN_SCI_FLUX'+str(i_orderlet), linewidth =  0.3)
            plt.plot(wav_green[10,:],np.array(hdulist['GREEN_CAL_FLUX'].data,'d')[10,:], label = 'GREEN_CAL_FLUX', linewidth =  0.3)
            plt.legend()
            plt.yscale('log')
            plt.title('Science Orderlets in GREEN '+exposure_name)
            plt.ylabel('Counts (e-)',fontsize = 15)
            plt.xlabel('Wavelength (Ang)',fontsize = 15)
            plt.savefig(output_dir+'/'+exposure_name+'/L1/'+exposure_name+'_3_science_fibres_GREEN_CCD.png',dpi = 200)
            plt.close()

            plt.close()
            plt.figure(figsize=(10,4))
            plt.subplots_adjust(left=0.2, bottom=0.15, right=0.9, top=0.9)
            for i_orderlet in [1,2,3]:
                flux_tmp = np.array(hdulist['RED_SCI_FLUX'+str(i_orderlet)].data,'d')
                if np.shape(flux_tmp)==(0,): continue
                plt.plot(wav_red[10,:],flux_tmp[10,:], label = 'RED_SCI_FLUX'+str(i_orderlet), linewidth =  0.3)
            plt.plot(wav_red[10,:],np.array(hdulist['RED_CAL_FLUX'].data,'d')[10,:], label = 'RED_CAL_FLUX', linewidth =  0.3)
            plt.legend()
            plt.yscale('log')
            plt.title('Science Orderlets in RED '+exposure_name)
            plt.ylabel('Counts (e-)',fontsize = 15)
            plt.xlabel('Wavelength (Ang)',fontsize = 15)
            plt.savefig(output_dir+'/'+exposure_name+'/L1/'+exposure_name+'_3_science_fibres_RED_CCD.png',dpi = 200)
            plt.close()


            #plot the ratio between orderlets all relative to the first order, plot as a function of wav, label by order number, red and green in the same plot
            plt.close()
            plt.figure(figsize=(10,4))
            plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)
            is_fiber_on =[np.nanmedian(flux_green2/flux_green)>0.2,np.nanmedian(flux_green3/flux_green)>0.2,np.nanpercentile(flux_green_cal/flux_green,95)>0.05,np.nanmedian(flux_green_sky/flux_green)>0.05]
            #print('test orderlets', np.nanmedian(flux_green2/flux_green),np.nanmedian(flux_green3/flux_green),np.nanpercentile(flux_green_cal/flux_green,[95]),np.nanmedian(flux_green_sky/flux_green))
            plt.plot(np.nanmedian(wav_green,axis = 1),np.nanmedian(flux_green2/flux_green,axis = 1),marker = 'o', color = 'green', label = 'Sci2/Sci1; On: ' +str(is_fiber_on[0]))
            plt.plot(np.nanmedian(wav_green,axis = 1),np.nanmedian(flux_green3/flux_green,axis = 1),marker = 'o', color = 'red', label = 'Sci3/Sci1; On: ' +str(is_fiber_on[1]))
            plt.plot(np.nanmedian(wav_green,axis = 1),np.nanpercentile(flux_green_cal/flux_green,95,axis = 1),marker = 'o', color = 'blue', label = 'Cal/Sci1; On: ' +str(is_fiber_on[2]))
            plt.plot(np.nanmedian(wav_green,axis = 1),np.nanmedian(flux_green_sky/flux_green,axis = 1),marker = 'o', color = 'magenta', label = 'Sky/Sci1; On: ' +str(is_fiber_on[3]))

            plt.plot(np.nanmedian(wav_red,axis = 1),np.nanmedian(flux_red2/flux_red,axis = 1),marker = 'D', color = 'green')
            plt.plot(np.nanmedian(wav_red,axis = 1),np.nanmedian(flux_red3/flux_red,axis = 1),marker = 'D', color = 'red')
            plt.plot(np.nanmedian(wav_red,axis = 1),np.nanpercentile(flux_red_cal/flux_red,95,axis = 1),marker = 'D', color = 'blue')
            plt.plot(np.nanmedian(wav_red,axis = 1),np.nanmedian(flux_red_sky/flux_red,axis = 1),marker = 'D', color = 'magenta')
            plt.legend()
            plt.title('Orderlets Flux Ratios '+exposure_name)
            #plt.ylabel('Counts (e-)',fontsize = 15)
            plt.xlabel('Wavelength (Ang)',fontsize = 15)
            plt.savefig(output_dir+'/'+exposure_name+'/L1/'+exposure_name+'_orderlets_flux_ratio.png',dpi = 200)
            plt.close()
        else: print('L1 file does not exist')


        hdulist.close()

        plt.close('all')

