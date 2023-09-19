import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from modules.Utils.config_parser import ConfigHandler
from kpfpipe.models.level0 import KPF0
from kpfpipe.models.level1 import KPF1
from kpfpipe.models.level2 import KPF2
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
from modules.Utils.analyze_hk import AnalyzeHK
from modules.Utils.analyze_2d import Analyze2D
from modules.Utils.analyze_l1 import AnalyzeL1
from modules.Utils.analyze_wls import AnalyzeWLS
from modules.Utils.analyze_l2 import AnalyzeL2
from modules.Utils.kpf_parse import HeaderParse
#import kpfpipe.pipelines.fits_primitives as fits_primitives
from modules.Utils.kpf_parse import get_data_products_L0
from modules.Utils.kpf_parse import get_data_products_2D
from modules.Utils.kpf_parse import get_data_products_L1
from modules.Utils.kpf_parse import get_data_products_L2

class QuicklookAlg:
    """
    This class contains methods to generate Quicklook data products for L0/2D/L1/L2
    files from KPF.  The qlp_L0(), qlp_2D(), qlp_L1(), and qlp_L2() methods take a KPF 
    object of the appropriate type and generate the QLP plots using "Analysis" classes 
    and methods (e.g., AnalyzeL0 in analyze_l0.py).  Data products are put into standard 
    locations in /data/QLP/<datecode>/<L0/2D/L1/L2>/.
    
    The following recipes in KPF-Pipeline/recipes/ are useful for generating QLP data 
    products:
    
    quicklook_watch_dir.recipe -- this recipe watches a directory (recursively, if needed) 
        and triggers the QLP on file modification events.  It must be run in watch mode.  
        Separate instances should to be run for L0, 2D, L1, and L2 data directories.
        Example:
            > kpf --watch /data/L0/20230711/ -c configs/quicklook_watch_dir.cfg -r recipes/quicklook_watch_dir.recipe

    quicklook_date.recipe -- this recipe is run in non-watch mode and is useful for bulk
         (re)processing of QLP data products.  It computes all QLP data products 
         (L0, 2D, L1, and L2) for a given datecode (e.g., 20230711).  
         
         Example
             > kpf --date 20230711 -c configs/quicklook_date.cfg -r recipes/quicklook_date.recipe
             
    quicklook_match.recipe -- this recipe is used to manually produce a set of QLP 
         outputs that match the fullpath config variable.  It can be used to produce 
         QLP for a single data level (L0, 2D, L1, L2) for a single datecode (YYYYDDMM) 
         or combinations.  All of the examples below are executed using the command 
             > kpf -c configs/quicklook_match.cfg -r recipes/quicklook_match.recipe
         but with different values for the config variable 'fullpath'.
        
         Example - compute L0 data products for KP.20230724.48905.30:
            fullpath = '/data/L0/20230724/KP.20230724.48905.30.fits'
        
         Example - compute L0/2D/L1/L2 data products for KP.20230724.48905.30:
            fullpath = '/data/??/20230724/KP.20230724.48905.30*.fits'
        
         Example - compute L0/2D/L1/L2 data products for all ObsID on a particular date:
            fullpath = '/data/??/20230724/KP.*.fits'
        
         Example - compute L0/2D/L1/L2 data products for all ObsID on a range of ten dates:
            fullpath = '/data/??/2023072?/KP.*.fits'
    
    The QLP Analysis classes and methods can also be used outside of the pipeline 
    context, e.g., in a Jupyter Notebook.
    """

    def __init__(self,config=None,logger=None):

        self.config=config
        self.logger=logger

    #######################
    ##### QLP Level 0 #####
    #######################
    def qlp_L0(self, kpf0, output_dir):
        """
        Description:
            Generates the standard quicklook data products for an L0 object.

        Arguments:
            kpf0 - a L0 object
            output_dir - directory for output QLP files (if show_plot=False)
    
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

        # First create the base output directory, if needed
        try:
            basedir = L0_QLP_file_base 
            os.makedirs(basedir, exist_ok=True) 

        except Exception as e:
            self.logger.error(f"Failure creating base output diretory in Exposure Meter quicklook pipeline: {e}\n{traceback.format_exc()}")

        # Make Exposure Meter plots
        if 'ExpMeter' in self.data_products:
            try:
                savedir = L0_QLP_file_base +'EM/'
                os.makedirs(savedir, exist_ok=True) # make directories if needed
                myEM = AnalyzeEM(kpf0, logger=self.logger)

                # Exposure Meter time series plots
                filename = savedir + self.ObsID + '_EM_time_series_sci_zoomable.png'
                self.logger.info('Generating QLP image ' + filename)
                myEM.plot_EM_time_series(fiber='sci', fig_path=filename, show_plot=False)
                filename = savedir + self.ObsID + '_EM_time_series_sky_zoomable.png'
                self.logger.info('Generating QLP image ' + filename)
                myEM.plot_EM_time_series(fiber='sky', fig_path=filename, show_plot=False)
                filename = savedir + self.ObsID + '_EM_time_series_ratio_zoomable.png'
                self.logger.info('Generating QLP image ' + filename)
                myEM.plot_EM_time_series(fiber='ratio', fig_path=filename, show_plot=False)

                # Exposure Meter spectrum plot
                filename = savedir + self.ObsID + '_EM_spectrum_zoomable.png'
                self.logger.info('Generating QLP image ' + filename)
                myEM.plot_EM_spectrum(fig_path=filename, show_plot=False)

            except Exception as e:
                self.logger.error(f"Failure in Exposure Meter quicklook pipeline: {e}\n{traceback.format_exc()}")

        # Make CaHK plots
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
        if chips != []:    
            try:
                myL0 = AnalyzeL0(kpf0, logger=self.logger)
                for chip in chips:
                    savedir = L0_QLP_file_base +'L0/'
                    os.makedirs(savedir, exist_ok=True) # make directories if needed
                    filename = savedir + self.ObsID + '_L0_stitched_image_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    myL0.plot_L0_stitched_image(fig_path=filename, chip=chip, show_plot=False)
            except Exception as e:
                self.logger.error(f"Failure in L0 quicklook pipeline: {e}\n{traceback.format_exc()}")


    ##################
    ##### QLP 2D #####
    ##################
    def qlp_2D(self, kpf2d, output_dir):
        """
        Description:
            Generates the standard quicklook data products for a 2D object.
    
        Arguments:
            kpf2d - a 2D object
            output_dir - directory for output QLP files (if show_plot=False)
    
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

        # First create the base output directory, if needed
        try:
            basedir = D2_QLP_file_base 
            os.makedirs(basedir, exist_ok=True) 

        except Exception as e:
            self.logger.error(f"Failure creating base output diretory in Exposure Meter quicklook pipeline: {e}\n{traceback.format_exc()}")

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

                # Guider error time series, PSD, and other time series
                filename = savedir + self.ObsID + '_error_time_series_zoomable.png'
                self.logger.info('Generating QLP image ' + filename)
                #myGuider.plot_guider_error_time_series(fig_path=filename, show_plot=False)
                myGuider.plot_guider_error_time_series(fig_path=filename, show_plot=False)

            except Exception as e:
                self.logger.error(f"Failure in Guider quicklook pipeline: {e}\n{traceback.format_exc()}")

        # Make 2D images
        # to-do: process bias and dark differently
        if chips != []:    
            try:
                savedir = D2_QLP_file_base +'2D/'
                os.makedirs(savedir, exist_ok=True) # make directories if needed
                my_2D = Analyze2D(kpf2d, logger=self.logger)
                for chip in chips:
                    # next line not working yet
                    #Analyze2D.measure_2D_dark_current(self, chip=chip)
                    filename = savedir + self.ObsID + '_2D_image_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    my_2D.plot_2D_image(chip=chip, fig_path=filename, show_plot=False)

            except Exception as e:
                self.logger.error(f"Failure in 2D quicklook pipeline: {e}\n{traceback.format_exc()}")


        # Make 2D images - 3x3 arrays
        if chips != []:    
            try:
                savedir = D2_QLP_file_base +'2D/'
                os.makedirs(savedir, exist_ok=True) # make directories if needed
                my_2D = Analyze2D(kpf2d, logger=self.logger)
                for chip in chips:
                    filename = savedir + self.ObsID + '_2D_image_3x3zoom_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    my_2D.plot_2D_image_zoom_3x3(chip=chip, fig_path=filename, show_plot=False)

            except Exception as e:
                self.logger.error(f"Failure in 2D quicklook pipeline: {e}\n{traceback.format_exc()}")


        # TO-DO Add bias histogram


        # Make 2D image histograms
        if chips != []:    
            try:
                my_2D = Analyze2D(kpf2d, logger=self.logger)
                for chip in chips:
                    filename = savedir + self.ObsID + '_2D_histogram_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    my_2D.plot_2D_image_histogram(chip=chip, fig_path=filename, show_plot=False)

            except Exception as e:
                self.logger.error(f"Failure in 2D quicklook pipeline: {e}\n{traceback.format_exc()}")

        # Make 2D column cuts
        if chips != []:    
            try:
                my_2D = Analyze2D(kpf2d, logger=self.logger)
                for chip in chips:
                    filename = savedir + self.ObsID + '_2D_column_cut_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    my_2D.plot_2D_column_cut(chip=chip, fig_path=filename, show_plot=False)

            except Exception as e:
                self.logger.error(f"Failure in 2D quicklook pipeline: {e}\n{traceback.format_exc()}")
        
        
    #######################
    ##### QLP Level 1 #####
    #######################
    def qlp_L1(self, kpf1, output_dir):
        """
        Description:
            Generates the standard quicklook data products for an L1 object.
    
        Arguments:
            kpf1 - a L1 object
            output_dir - directory for output QLP files (if show_plot=False)
    
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

        # First create the base output directory, if needed
        try:
            basedir = L1_QLP_file_base 
            os.makedirs(basedir, exist_ok=True) 

        except Exception as e:
            self.logger.error(f"Failure creating base output diretory in Exposure Meter quicklook pipeline: {e}\n{traceback.format_exc()}")

        # Make WLS plots
        try:
            savedir = L1_QLP_file_base +'WLS/'
            os.makedirs(savedir, exist_ok=True) # make directories if needed
            if chips != []:    
                try:
                    for chip in chips:
                        filename = savedir + self.ObsID + '_WLS_orderlet_diff_' + chip + '_zoomable.png'
                        self.logger.info('Generating QLP image ' + filename)
                        myWLS = AnalyzeWLS(kpf1, logger=self.logger)
                        myWLS.plot_WLS_orderlet_diff(chip=chip, fig_path=filename, show_plot=False)
                except Exception as e:
                    self.logger.error(f"Failure in L1 quicklook pipeline: {e}\n{traceback.format_exc()}")

        except Exception as e:
            self.logger.error(f"Failure in L1 quicklook pipeline: {e}\n{traceback.format_exc()}")

        # Make L1 SNR plot
        try:
            savedir = L1_QLP_file_base +'L1/'
            os.makedirs(savedir, exist_ok=True) # make directories if needed
            filename = savedir + self.ObsID + '_L1_SNR_zoomable.png'
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
        if chips != []:    
            try:
                myL1 = AnalyzeL1(kpf1, logger=self.logger)
                if 'green' in chips:  # don't use 'for chip in chips:' here so that the file creation order is correct for Jump to display in a certain order
                    chip = 'green'
                    filename = savedir + self.ObsID + '_L1_spectrum_SCI_order11_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    myL1.plot_1D_spectrum_single_order(chip=chip, order=11, ylog=False, 
                                                       orderlet=['SCI1', 'SCI2', 'SCI3'], 
                                                       fig_path=filename, show_plot=False)
                if 'red' in chips:
                    chip = 'red'
                    filename = savedir + self.ObsID + '_L1_spectrum_SCI_order11_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    myL1.plot_1D_spectrum_single_order(chip=chip, order=11, ylog=False, 
                                                       orderlet=['SCI1', 'SCI2', 'SCI3'], 
                                                       fig_path=filename, show_plot=False)
                if 'green' in chips:
                    chip = 'green'
                    filename = savedir + self.ObsID + '_L1_spectrum_SKY_order11_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    myL1.plot_1D_spectrum_single_order(chip=chip, order=11, ylog=False, 
                                                       orderlet=['SKY'], 
                                                       fig_path=filename, show_plot=False)
                if 'red' in chips:
                    chip = 'red'
                    filename = savedir + self.ObsID + '_L1_spectrum_SKY_order11_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    myL1.plot_1D_spectrum_single_order(chip=chip, order=11, ylog=False, 
                                                       orderlet=['SKY'], 
                                                       fig_path=filename, show_plot=False)
                if 'green' in chips:
                    chip = 'green'
                    filename = savedir + self.ObsID + '_L1_spectrum_CAL_order11_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    myL1.plot_1D_spectrum_single_order(chip=chip, order=11, ylog=False, 
                                                       orderlet=['CAL'], 
                                                       fig_path=filename, show_plot=False)
                if 'red' in chips:
                    chip = 'red'
                    filename = savedir + self.ObsID + '_L1_spectrum_CAL_order11_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    myL1.plot_1D_spectrum_single_order(chip=chip, order=11, ylog=False, 
                                                       orderlet=['CAL'], 
                                                       fig_path=filename, show_plot=False)

            except Exception as e:
                self.logger.error(f"Failure in L1 quicklook pipeline: {e}\n{traceback.format_exc()}")

        # Make order ratio grid plots
        if chips != []:    
            try:
                for chip in chips:
                    filename = savedir + self.ObsID + '_L1_orderlet_flux_ratios_grid_' + chip + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    myL1 = AnalyzeL1(kpf1, logger=self.logger)
                    myL1.measure_orderlet_flux_ratios()
                    myL1.plot_orderlet_flux_ratios_grid(chip=chip, fig_path=filename, show_plot=False)

            except Exception as e:
                self.logger.error(f"Failure in L1 quicklook pipeline: {e}\n{traceback.format_exc()}")

        # Make order ratio plot
        if chips != []:    
            try:
                filename = savedir + self.ObsID + '_L1_orderlet_flux_ratios_zoomable.png'
                self.logger.info('Measuring orderlet flux ratios for ' + str(self.ObsID) + '.')
                self.logger.info('Generating QLP image ' + filename)
                myL1.plot_orderlet_flux_ratios(fig_path=filename, show_plot=False)

            except Exception as e:
                self.logger.error(f"Failure in L1 quicklook pipeline: {e}\n{traceback.format_exc()}")


    #######################
    ##### QLP Level 2 #####
    #######################
    def qlp_L2(self, kpf2, output_dir):
        """
        Description:
            Generates the standard quicklook data products for an L2 object.
    
        Arguments:
            kpf2 - a L2 object
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

        # First create the base output directory, if needed
        try:
            basedir = L2_QLP_file_base 
            os.makedirs(basedir, exist_ok=True) 

        except Exception as e:
            self.logger.error(f"Failure creating base output diretory in Exposure Meter quicklook pipeline: {e}\n{traceback.format_exc()}")

        # Make CCF grid plots
        if chips != []:    
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


    #######################
    ##### QLP MASTERS #####
    #######################
    def qlp_master(self, input_file, output_dir):
        """
        Description:
            Generates the standard quicklook data product for a master file.
    
        Arguments:
            input_file (string) - full path to the master file to be processed
            output_dir - directory for output QLP files (if show_plot=False)
    
        Attributes:
            None
        """

        self.input_file = input_file
        self.output_dir = output_dir
        master_type, data_type = determine_master_type(self.input_file)
        self.logger.info('The master file ' + str(self.input_file) + ' was determined to be a ' + 
                         str(data_type) + ' ' + str(master_type) + ' file.')
        if data_type == '2D':
            kpf2d = KPF0.from_fits(self.input_file)
            self.data_products = get_data_products_2D(kpf2d)
        if data_type == 'L1':
            kpf1 = KPF1.from_fits(self.input_file)
            self.data_products = get_data_products_L1(kpf1)
        if data_type == 'L2':
            kpf2 = KPF2.from_fits(self.input_file)
            self.data_products = get_data_products_L2(kpf2)
        chips = []
        if master_type != None:
            if 'Green' in self.data_products: chips.append('green')
            if 'Red'   in self.data_products: chips.append('red')

        # Make directory, if needed
        try:
            if master_type == 'bias':
                savedir = self.output_dir + 'Bias/' 
            elif master_type == 'dark':
                savedir = self.output_dir + 'Dark/' 
            elif master_type == 'flat':
                savedir = self.output_dir + 'Flat/' 
            elif master_type == 'lfc':
                savedir = self.output_dir + 'LFC/' 
            elif master_type == 'etalon':
                savedir = self.output_dir + 'Etalon/' 
            elif master_type == 'thar':
                savedir = self.output_dir + 'ThAr/' 
            elif master_type == 'une':
                savedir = self.output_dir + 'UNe/' 
            else:
                self.logger.error(f"Couldn't determine data type to create directory in Master quicklook pipeline.")
                savedir = self.output_dir + 'Unidentified/' 
            os.makedirs(savedir, exist_ok=True) 

        except Exception as e:
            self.logger.error(f"Failure creating base output diretory in Master quicklook pipeline: {e}\n{traceback.format_exc()}")

        ### 2D Masters ###
        if data_type == '2D':
            if chips != []:

                # Make 2D images
                try:
                    my_2D = Analyze2D(kpf2d, logger=self.logger)
                    for chip in chips:
                        filename = savedir + self.input_file.split('/')[-1].replace('.fits', '') + \
                                   '_2D_image_' + chip + '_zoomable.png'
                        self.logger.info('Generating QLP image ' + filename)
                        my_2D.plot_2D_image(chip=chip, fig_path=filename, show_plot=False)
    
                except Exception as e:
                    self.logger.error(f"Failure in Master quicklook pipeline: {e}\n{traceback.format_exc()}")

                # Make 2D images - 3x3 arrays
                try:
                    my_2D = Analyze2D(kpf2d, logger=self.logger)
                    for chip in chips:  
                        filename = savedir + self.input_file.split('/')[-1].replace('.fits', '') + \
                                   '_2D_image_3x3zoom_' + chip + '_zoomable.png'
                        self.logger.info('Generating QLP image ' + filename)
                        my_2D.plot_2D_image_zoom_3x3(chip=chip, fig_path=filename, show_plot=False)

                except Exception as e:
                    self.logger.error(f"Failure in Master quicklook pipeline: {e}\n{traceback.format_exc()}")

                # Make 2D image histograms
                try:
                    my_2D = Analyze2D(kpf2d, logger=self.logger)
                    for chip in chips:
                        filename = savedir + self.input_file.split('/')[-1].replace('.fits', '') + \
                                   '_2D_histogram_' + chip + '_zoomable.png'
                        self.logger.info('Generating QLP image ' + filename)
                        my_2D.plot_2D_image_histogram(chip=chip, fig_path=filename, show_plot=False)
    
                except Exception as e:
                    self.logger.error(f"Failure in Master quicklook pipeline: {e}\n{traceback.format_exc()}")
    
                # Make 2D column cuts
                try:
                    my_2D = Analyze2D(kpf2d, logger=self.logger)
                    for chip in chips:
                        filename = savedir + self.input_file.split('/')[-1].replace('.fits', '') + \
                                   '_2D_column_cut_' + chip + '_zoomable.png'
                        self.logger.info('Generating QLP image ' + filename)
                        my_2D.plot_2D_column_cut(chip=chip, fig_path=filename, show_plot=False)

                except Exception as e:
                    self.logger.error(f"Failure in Master quicklook pipeline: {e}\n{traceback.format_exc()}")
        
        
        ### L1 Masters ###
        if data_type == 'L1':

            # Make L1 SNR plot
            try:
                filename = savedir + self.input_file.split('/')[-1].replace('.fits', '') + \
                           '_L1_SNR_' + '_zoomable.png'
                self.logger.info('Generating QLP image ' + filename)
                myL1 = AnalyzeL1(kpf1, logger=self.logger)
                myL1.measure_L1_snr()
                myL1.plot_L1_snr(fig_path=filename, show_plot=False)
    
            except Exception as e:
                self.logger.error(f"Failure in Master quicklook pipeline: {e}\n{traceback.format_exc()}")

            # Make L1 spectra plots
            try:
                for oo, orderlet in enumerate(['SCI1', 'SCI2', 'SCI3', 'CAL', 'SKY']):
                    filename = savedir + self.input_file.split('/')[-1].replace('.fits', '') + \
                               '_L1_spectrum_' + orderlet  + '_zoomable.png'
                    self.logger.info('Generating QLP image ' + filename)
                    myL1.plot_L1_spectrum(orderlet=orderlet, fig_path=filename, show_plot=False)
    
            except Exception as e:
                self.logger.error(f"Failure in Master quicklook pipeline: {e}\n{traceback.format_exc()}")

            # Make L1 spectra plots (single order)
            if chips != []:    
                try:
                    myL1 = AnalyzeL1(kpf1, logger=self.logger)
                    if 'green' in chips:  # don't use 'for chip in chips:' here so that the file creation order is correct for Jump to display in a certain order
                        chip = 'green'
                        filename = savedir + self.input_file.split('/')[-1].replace('.fits', '') + \
                                   '_L1_spectrum_SCI_order11_' + chip + '_zoomable.png'
                        self.logger.info('Generating QLP image ' + filename)
                        myL1.plot_1D_spectrum_single_order(chip=chip, order=11, ylog=False, 
                                                           orderlet=['SCI1', 'SCI2', 'SCI3'], 
                                                           fig_path=filename, show_plot=False)
                    if 'red' in chips:
                        chip = 'red'
                        filename = savedir + self.input_file.split('/')[-1].replace('.fits', '') + \
                                   '_L1_spectrum_SCI_order11_' + chip + '_zoomable.png'
                        self.logger.info('Generating QLP image ' + filename)
                        myL1.plot_1D_spectrum_single_order(chip=chip, order=11, ylog=False, 
                                                           orderlet=['SCI1', 'SCI2', 'SCI3'], 
                                                           fig_path=filename, show_plot=False)
                    if 'green' in chips:
                        chip = 'green'
                        filename = savedir + self.input_file.split('/')[-1].replace('.fits', '') + \
                                   '_L1_spectrum_SKY_order11_' + chip + '_zoomable.png'
                        self.logger.info('Generating QLP image ' + filename)
                        myL1.plot_1D_spectrum_single_order(chip=chip, order=11, ylog=False, 
                                                           orderlet=['SKY'], 
                                                           fig_path=filename, show_plot=False)
                    if 'red' in chips:
                        chip = 'red'
                        filename = savedir + self.input_file.split('/')[-1].replace('.fits', '') + \
                                   '_L1_spectrum_SKY_order11_' + chip + '_zoomable.png'
                        self.logger.info('Generating QLP image ' + filename)
                        myL1.plot_1D_spectrum_single_order(chip=chip, order=11, ylog=False, 
                                                           orderlet=['SKY'], 
                                                           fig_path=filename, show_plot=False)
                    if 'green' in chips:
                        chip = 'green'
                        filename = savedir + self.input_file.split('/')[-1].replace('.fits', '') + \
                                   '_L1_spectrum_CAL_order11_' + chip + '_zoomable.png'
                        self.logger.info('Generating QLP image ' + filename)
                        myL1.plot_1D_spectrum_single_order(chip=chip, order=11, ylog=False, 
                                                           orderlet=['CAL'], 
                                                           fig_path=filename, show_plot=False)
                    if 'red' in chips:
                        chip = 'red'
                        filename = savedir + self.input_file.split('/')[-1].replace('.fits', '') + \
                                   '_L1_spectrum_CAL_order11_' + chip + '_zoomable.png'
                        self.logger.info('Generating QLP image ' + filename)
                        myL1.plot_1D_spectrum_single_order(chip=chip, order=11, ylog=False, 
                                                           orderlet=['CAL'], 
                                                           fig_path=filename, show_plot=False)

                except Exception as e:
                    self.logger.error(f"Failure in Master quicklook pipeline: {e}\n{traceback.format_exc()}")

            # Make order ratio grid plots
            if chips != []:    
                try:
                    for chip in chips:
                        filename = savedir + self.input_file.split('/')[-1].replace('.fits', '') + \
                                   '_L1_orderlet_flux_ratios_grid_' + chip + '_zoomable.png'
                        self.logger.info('Generating QLP image ' + filename)
                        myL1 = AnalyzeL1(kpf1, logger=self.logger)
                        myL1.measure_orderlet_flux_ratios()
                        myL1.plot_orderlet_flux_ratios_grid(chip=chip, fig_path=filename, show_plot=False)
    
                except Exception as e:
                    self.logger.error(f"Failure in Master quicklook pipeline: {e}\n{traceback.format_exc()}")
    
            # Make order ratio plot
            if chips != []:    
                try:
                    filename = savedir + self.input_file.split('/')[-1].replace('.fits', '') + \
                               '_L1_orderlet_flux_ratios_zoomable.png'
                    self.logger.info('Measuring orderlet flux ratios for ' + filename)
                    self.logger.info('Generating QLP image ' + filename)
                    myL1.plot_orderlet_flux_ratios(fig_path=filename, show_plot=False)

                except Exception as e:
                    self.logger.error(f"Failure in Master quicklook pipeline: {e}\n{traceback.format_exc()}")


        ### L2 Masters ###
        if data_type == 'L2':
        
        # Make CCF grid plots
            if chips != []:    
                try:
                    myL2 = AnalyzeL2(kpf2, logger=self.logger)
                    for chip in chips:
                        filename = savedir + self.input_file.split('/')[-1].replace('.fits', '') + \
                                   '_CCF_grid_' + chip + '_zoomable.png'
                        self.logger.info('Generating QLP image ' + filename)
                        myL2.plot_CCF_grid(chip=chip, fig_path=filename, show_plot=False)
    
                except Exception as e:
                    self.logger.error(f"Failure in CCF quicklook pipeline: {e}\n{traceback.format_exc()}")


def determine_master_type(fullpath):
    """
    Description:
        Generates the standard quicklook data products for all of the master files in 
        a directory.

    Arguments:
        fullpath (string) - full path to master file (usually a .fits)

    Outputs:
        master_type - possible values: None, 'bias', dark, 'flat', 'thar', 'une', 'lfc'
        data_type - possible values: None, 'L0', '2D', 'L1', 'L2', 'WLS'

    Attributes:
        None
    """
    fullpath = fullpath.lower()
    master_type = None
    data_type = None
    
    # Bias
    if (('bias' in fullpath) and fullpath.endswith('.fits')):
        master_type = 'bias'
        if 'l1' in fullpath:
            data_type = 'L1'
        else:
            data_type = '2D'
    # Dark
    if (('dark' in fullpath) and fullpath.endswith('.fits')):
        master_type = 'dark'
        if 'l1' in fullpath:
            data_type = 'L1'
        else:
            data_type = '2D'
    # Flat
    if (('flat' in fullpath) and fullpath.endswith('.fits')):
        master_type = 'flat'
        if 'l2' in fullpath:
            data_type = 'L2'
        elif 'l1' in fullpath:
            data_type = 'L1'
        else:
            data_type = '2D'
    # LFC
    if (('lfc' in fullpath) and fullpath.endswith('.fits')):
        master_type = 'lfc'
        if 'l2' in fullpath:
            data_type = 'L2'
        elif 'l1' in fullpath:
            data_type = 'L1'
        else:
            data_type = '2D'
    # Etalon
    if (('etalon' in fullpath) and fullpath.endswith('.fits')):
        master_type = 'etalon'
        if 'l2' in fullpath:
            data_type = 'L2'
        elif 'l1' in fullpath:
            data_type = 'L1'
        else:
            data_type = '2D'
    # ThAr
    if (('thar' in fullpath) and fullpath.endswith('.fits')):
        master_type = 'thar'
        if 'l2' in fullpath:
            data_type = 'L2'
        elif 'l1' in fullpath:
            data_type = 'L1'
        else:
            data_type = '2D'
    # UNe
    if (('une' in fullpath) and fullpath.endswith('.fits')):
        master_type = 'une'
        if 'l2' in fullpath:
            data_type = 'L2'
        elif 'l1' in fullpath:
            data_type = 'L1'
        else:
            data_type = '2D'

    return master_type, data_type

