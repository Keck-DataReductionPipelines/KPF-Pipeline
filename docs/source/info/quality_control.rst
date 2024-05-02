Quality Control
===============

The KPF DRP has several Quality Control (QC) methods that can be run on the L0, 2D, L1, and 2D objects.  
The QC tests are run during normal processing in the main recipe.  
The results of these QC checks are added to the primary headers of kpf objects, which are written to 2D, L1, and L2 FITS files (but not the L0 files, which with rare exceptions are not modified after data collection at WMKO).
The QC methods are defined in the ``QCDefinitions`` class in ``modules/quality_control/src/quality_control.py``.

The FITS header keywords produced by QC tests are defined in :doc:`data_format`.  
The QCDefinitions.list_qc_metrics () produces a list of QC tests and their characteristics (including the primary header keywords), as shown below.
Note that some QC tests are applied to multiple KPF data levels and spectrum types.

::

    > from modules.quality_control.src.quality_control import *
    > myQCdef = QCDefinitions()
    > myQCdef.list_qc_metrics()
    
    Quality Control tests for L0:
       QC Name: not_junk_check
          Description: Check if file is not in list of junk files.
          Data levels: ['L0', '2D', 'L1', 'L2']
          Data type: int
          Spectrum types: ['all']
          Keyword: NOTJUNK
          Comment: QC: Not in list of junk files
    
       QC Name: L0_data_products_check
          Description: Check if expected L0 data products are present with non-zero array sizes.
          Data levels: ['L0']
          Data type: int
          Spectrum types: ['all']
          Keyword: DATAPRL0
          Comment: QC: L0 data present
    
       QC Name: L0_header_keywords_present_check
          Description: Check if expected L0 header keywords are present.
          Data levels: ['L0']
          Data type: int
          Spectrum types: ['all']
          Keyword: KWRDPRL0
          Comment: QC: L0 keywords present
    
       QC Name: L0_datetime_checks
          Description: Check for timing consistency in L0 header keywords and Exp Meter table.
          Data levels: ['L0']
          Data type: int
          Spectrum types: ['all']
          Keyword: TIMCHKL0
          Comment: QC: L0 times consistent
    
       QC Name: exposure_meter_not_saturated_check
          Description: Check if 2+ reduced EM pixels are within 90% of saturation in EM-SCI or EM-SKY.
          Data levels: ['L0']
          Data type: int
          Spectrum types: ['all']
          Keyword: EMSAT
          Comment: QC: EM not saturated
    
       QC Name: exposure_meter_flux_not_negative_check
          Description: Check for negative flux in the EM-SCI and EM-SKY by looking for 20 consecuitive pixels in the summed spectra with negative flux.
          Data levels: ['L0']
          Data type: int
          Spectrum types: ['all']
          Keyword: EMNEG
          Comment: QC: EM not negative flux
    
    Quality Control tests for 2D:
       QC Name: not_junk_check
          Description: Check if file is not in list of junk files.
          Data levels: ['L0', '2D', 'L1', 'L2']
          Data type: int
          Spectrum types: ['all']
          Keyword: NOTJUNK
          Comment: QC: Not in list of junk files
    
       QC Name: data_2D_red_green_check
          Description: Check to see if red and green CCD data is present with expected array sizes.
          Data levels: ['2D']
          Data type: int
          Spectrum types: ['all']
          Keyword: DATAPR2D
          Comment: QC: 2D red and green data present check
          Database column: None
    
       QC Name: data_2D_CaHK_check
          Description: Check to see if CaHK CCD data is present with expected array sizes.
          Data levels: ['2D']
          Data type: int
          Spectrum types: ['all']
          Keyword: CaHKPR2D
          Comment: QC: 2D CaHK data present check
    
       QC Name: data_2D_bias_low_flux_check
          Description: Check to see if flux is low in bias exposure.
          Data levels: ['2D']
          Data type: int
          Spectrum types: ['Bias']
          Keyword: LOWBIAS
          Comment: QC: 2D bias low flux check
    
       QC Name: data_2D_dark_low_flux_check
          Description: Check to see if flux is low in dark exposure.
          Data levels: ['2D']
          Data type: int
          Spectrum types: ['Dark']
          Keyword: LOWDARK
          Comment: QC: 2D dark low flux check
    
       QC Name: D2_lfc_flux_check
          Description: Check if an LFC frame that goes into a master has sufficient flux
          Data levels: ['2D']
          Data type: int
          Spectrum types: ['LFC']
          Keyword: LFC2DFOK
          Comment: QC: LFC flux meets threshold of 4000 counts
    
    Quality Control tests for L1:
       QC Name: not_junk_check
          Description: Check if file is not in list of junk files.
          Data levels: ['L0', '2D', 'L1', 'L2']
          Data type: int
          Spectrum types: ['all']
          Keyword: NOTJUNK
          Comment: QC: Not in list of junk files
    
       QC Name: monotonic_wavelength_solution_check
          Description: Check if wavelength solution is monotonic.
          Data levels: ['L1']
          Data type: int
          Spectrum types: ['all']
          Keyword: MONOTWLS
          Comment: QC: Monotonic wavelength-solution
    
       QC Name: data_L1_red_green_check
          Description: Check to see if red and green data are present in L1 with expected shapes.
          Data levels: ['L1']
          Data type: int
          Spectrum types: ['all']
          Keyword: DATAPRL1
          Comment: QC: L1 red and green data present check
    
       QC Name: data_L1_CaHK_check
          Description: Check to see if CaHK data is present in L1 with expected shape.
          Data levels: ['L1']
          Data type: int
          Spectrum types: ['all']
          Keyword: CaHKPRL1
          Comment: QC: L1 CaHK present check
    
    Quality Control tests for L2:
       QC Name: not_junk_check
          Description: Check if file is not in list of junk files.
          Data levels: ['L0', '2D', 'L1', 'L2']
          Data type: int
          Spectrum types: ['all']
          Keyword: NOTJUNK
          Comment: QC: Not in list of junk files
    
       QC Name: L2_datetime_checks
          Description: Check for timing consistency in L2 files.
          Data levels: ['L2']
          Data type: int
          Spectrum types: ['all']
          Keyword: TIMCHKL2
          Comment: QC: L2 times consistent
    
       QC Name: data_L2_check
          Description: Check to see if all data is present in L2.
          Data levels: ['L2']
          Data type: int
          Spectrum types: ['all']
          Keyword: DATAPRL2
          Comment: QC: L2 data present check
