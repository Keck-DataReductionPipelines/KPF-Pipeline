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
from modules.Utils.analyze_hk import AnalyzeHK
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

        # First create the base output directory, if needed
        try:
            basedir = L0_QLP_file_base 
            os.makedirs(basedir, exist_ok=True) 

        except Exception as e:
            self.logger.error(f"Failure in Exposure Meter quicklook pipeline: {e}\n{traceback.format_exc()}")

        # Make Exposure Meter images
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

        # First create the base output directory, if needed
        try:
            basedir = D2_QLP_file_base 
            os.makedirs(basedir, exist_ok=True) 

        except Exception as e:
            self.logger.error(f"Failure in Exposure Meter quicklook pipeline: {e}\n{traceback.format_exc()}")

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
        if chips != []:    
            try:
                savedir = D2_QLP_file_base +'2D/'
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
        if chips != []:    
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
        if chips != []:    
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

        # First create the base output directory, if needed
        try:
            basedir = L1_QLP_file_base 
            os.makedirs(basedir, exist_ok=True) 

        except Exception as e:
            self.logger.error(f"Failure in Exposure Meter quicklook pipeline: {e}\n{traceback.format_exc()}")

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

        # First create the base output directory, if needed
        try:
            basedir = L2_QLP_file_base 
            os.makedirs(basedir, exist_ok=True) 

        except Exception as e:
            self.logger.error(f"Failure in Exposure Meter quicklook pipeline: {e}\n{traceback.format_exc()}")

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
