import os
import numpy as np
import numpy.ma as ma
import pandas as pd
from datetime import datetime
from scipy.ndimage import convolve1d
from modules.Utils.utils import DummyLogger
from modules.Utils.kpf_parse import HeaderParse, get_data_products_L0, get_datetime_obsid, get_kpf_level

"""
This module contains classes for KPF data quality control (QC).  Various QC metrics are defined in
class QCDefinitions.  Other classes QCL0, QC2D, QCL1, and QCL2 contain methods to compute QC values,
which are with the QC metrics, for specific data products, and then store them in the primary header
of the corresponding KPF object (which will be saved to a FITS file).  Normally QC values are stored 
headers, but storage in the KPF pipeline-operations database may be set up later by the database 
administrator, depending upon the special requirements for some QC metrics.
"""

iam = 'quality_control'
version = '1.3'

"""
The following are methods common across data levels, which are given at the beginning
of this module, before the QC classes are defined.

Includes helper functions that compute statistics of data of arbitrary shape.
"""

#####################################
# Module helper functions.
#####################################

def what_am_i():
    print('Software version:',iam + ' ' + version)

def avg_data_with_clipping(data_array,n_sigma = 3.0):
    """
    Statistics with outlier rejection (n-sigma data-trimming), ignoring NaNs, across all data array dimensions.
    """

    a = np.array(data_array)

    med = np.nanmedian(a)
    p16 = np.nanpercentile(a,16)
    p84 = np.nanpercentile(a,84)
    sigma = 0.5 * (p84 - p16)
    mdmsg = med - n_sigma * sigma
    b = np.less(a,mdmsg)
    mdpsg = med + n_sigma * sigma
    c = np.greater(a,mdpsg)
    d = np.where(np.isnan(a),True,False)
    mask = b | c | d
    mx = ma.masked_array(a, mask)
    avg = ma.getdata(mx.mean())
    std = ma.getdata(mx.std())
    cnt = ma.getdata(mx.count())

    return avg,std,cnt

def test_all_QCs(kpf_object, data_type):
    """
    Method to loop over all QC tests for the data level of the input KPF object 
    (an L0, 2D, L1, or L2 object).  This method is useful for testing (e.g., 
    in a Jupyter Notebook).  To run the QCs in a recipe, use methods in 
    quality_control_framework.py
    """
    
    logger = DummyLogger()
    
    data_level = get_kpf_level(kpf_object)
    
    # Define QC object
    if data_level == 'L0':
        qc_obj = QCL0(kpf_object)
    elif data_level == '2D':
        qc_obj = QC2D(kpf_object)
    elif data_level == 'L1':
        qc_obj = QCL1(kpf_object)
    elif data_level == 'L2':
        qc_obj = QCL2(kpf_object)
    else:
        print('data_level is not L0, 2D, L1, or L2.  Exiting.')
        
    if data_level != None:

        # Get a list of QC method names appropriate for the data level
        qc_names = []
        for qc_name in qc_obj.qcdefinitions.names:
            if data_level in qc_obj.qcdefinitions.kpf_data_levels[qc_name]:
                qc_names.append(qc_name)

        # Run the QC tests and add result to keyword to header
        primary_header = HeaderParse(kpf_object, 'PRIMARY')
        this_spectrum_type = primary_header.get_name(use_star_names=False)    
        logger.info(f'Spectrum type: {this_spectrum_type}')
        for qc_name in qc_names:
            try:
                spectrum_types = qc_obj.qcdefinitions.spectrum_types[qc_name]
                if (this_spectrum_type in spectrum_types) or ('all' in spectrum_types):
                    logger.info(f'Running QC: \033[1;34m{qc_name}\033[0m ({qc_obj.qcdefinitions.descriptions[qc_name]})')
                    method = getattr(qc_obj, qc_name) # get method with the name 'qc_name'
                    qc_value = method() # evaluate method
                    if qc_value == True: 
                        color_code = '\033[1;32m' # green
                    elif qc_value == False:
                        color_code = '\033[1;31m' # red
                    else:
                        color_code = '\033[1m'
                    logger.info(f'QC result:  {color_code}{qc_value}\033[0m (True = pass)')
                    qc_obj.add_qc_keyword_to_header(qc_name, qc_value)
                else:
                    logger.info(f'Not running QC: {qc_name} ({qc_obj.qcdefinitions.descriptions[qc_name]}) because spectrum type {this_spectrum_type} not in list of spectrum types: {spectrum_types}')
            except AttributeError as e:
                logger.info(f'Method {qc_name} does not exist in qc_obj or another AttributeError occurred: {e}')
                pass
            except Exception as e:
                logger.info(f'An error occurred when executing {qc_name}:', str(e))
                pass

#####################################################################

class QCDefinitions:

    """
    Description:
        This class defines QC metrics in a standard format.
        Dictionaries are used to associate unique metric names with various metric metadata.
        Modify this class to add new metrics.  Do not remove any metrics (we deprecate metrics
        simply by not using them any more).  When adding metrics to this class, ensure the length
        of the names list is equal to the number of dictionary entries.

    Class Attributes:
        names (list of strings): Each element is a unique and descriptive name for the metric.  No spaces allowed.
        descriptions (dictionary of strings): Each dictionary entry specifies a short description of the metric
            Try to keep it under 50 characters for brevity (this is not enforced but recommended).
        kpf_data_levels (dictionary of lists of strings): Each entry specifies the set of KPF data levels for the test.
            Possible values in the list: 'L0', '2D', 'L1', 'L2'
        data_types (dictionary of strings): Each entry specifies the Python data type of the metric.
            Only string, int, float are allowed.  Use 0/1 for boolean.
        spectrum_types (dictionary of arrays of strings ): Each entry specifies the types of spectra that the metric will be applied to.
            Possible strings in array: 'all', 'Bias', 'Dark', 'Flat', 'Wide Flat', 'LFC', 'Etalon', 'ThAr', 'UNe', 'Sun', 'Star', <starname>, ''
        fits_keywords (dictionary of strings): Each entry specifies the FITS-header keyword for the metric.
            Must be 8 characters or less, following the FITS standard.
        fits_comments (dictionary of strings): Each entry specifies the FITS-header comment for the metric.
            Must be a short string for brevity (say, under 35 characters), following the FITS standard.
        db_columns (dictionary of strings): Each entry specifies either database_table.column if applicable,
            or None if not.
    """

    def __init__(self, logger=None):

        self.logger = logger if logger is not None else DummyLogger()
        
        self.names = []
        self.descriptions = {}
        self.kpf_data_levels = {} 
        self.data_types = {}  # values are arrays; one or more of ['L0', '2D', 'L1', 'L2']
        self.spectrum_types = {} # values are arrays; one or more of []
        self.fits_keywords = {}
        self.fits_comments = {}
        self.db_columns = {}

        # Define QC metrics
        name0 = 'jarque_bera_test_red_amp1'
        self.names.append(name0)
        self.descriptions[name0] = 'Jarque-Bera test of pixel values for RED AMP-1 detector.'
        self.kpf_data_levels[name0] = ['L3'] # bogus value L3 to avoid executing
        self.data_types[name0] = 'float'
        self.spectrum_types[name0] = ['all', ]
        self.fits_keywords[name0] = 'JBTRED1'
        self.fits_comments[name0] = 'QC: J-B test for RED AMP-1 detector'
        self.db_columns[name0] = None

        name1 = 'not_junk_check'
        self.names.append(name1)
        self.descriptions[name1] = 'Check if file is not in list of junk files.'
        self.kpf_data_levels[name1] = ['L0', '2D', 'L1', 'L2']
        self.data_types[name1] = 'int'
        self.spectrum_types[name1] = ['all', ] # Need trailing comma to make list hashable
        self.fits_keywords[name1] = 'NOTJUNK'
        self.fits_comments[name1] = 'QC: Not in list of junk files'
        self.db_columns[name1] = None

        name2 = 'monotonic_wavelength_solution_check'
        self.names.append(name2)
        self.descriptions[name2] = 'Check if wavelength solution is monotonic.'
        self.kpf_data_levels[name2] = ['L1']
        self.data_types[name2] = 'int'
        self.spectrum_types[name2] = ['all', ]
        self.fits_keywords[name2] = 'MONOTWLS'
        self.fits_comments[name2] = 'QC: Monotonic wavelength-solution'
        self.db_columns[name2] = None

        name3 = 'L0_data_products_check'
        self.names.append(name3)
        self.kpf_data_levels[name3] = ['L0']
        self.descriptions[name3] = 'Check if expected L0 data products are present with non-zero array sizes.'
        self.data_types[name3] = 'int'
        self.spectrum_types[name3] = ['all', ]
        self.fits_keywords[name3] = 'DATAPRL0'
        self.fits_comments[name3] = 'QC: L0 data present'
        self.db_columns[name3] = None

        name4 = 'L0_header_keywords_present_check'
        self.names.append(name4)
        self.kpf_data_levels[name4] = ['L0']
        self.descriptions[name4] = 'Check if expected L0 header keywords are present.'
        self.data_types[name4] = 'int'
        self.spectrum_types[name4] = ['all', ]
        self.fits_keywords[name4] = 'KWRDPRL0'
        self.fits_comments[name4] = 'QC: L0 keywords present'
        self.db_columns[name4] = None

        name5 = 'L0_datetime_checks'
        self.names.append(name5)
        self.kpf_data_levels[name5] = ['L0']
        self.descriptions[name5] = 'Check for timing consistency in L0 header keywords and Exp Meter table.'
        self.data_types[name5] = 'int'
        self.spectrum_types[name5] = ['all', ]
        self.fits_keywords[name5] = 'TIMCHKL0'
        self.fits_comments[name5] = 'QC: L0 times consistent'
        self.db_columns[name5] = None

        name5b = 'L2_datetime_checks'
        self.names.append(name5b)
        self.kpf_data_levels[name5b] = ['L2']
        self.descriptions[name5b] = 'Check for timing consistency in L2 files.'
        self.data_types[name5b] = 'int'
        self.spectrum_types[name5b] = ['all', ]
        self.fits_keywords[name5b] = 'TIMCHKL2'
        self.fits_comments[name5b] = 'QC: L2 times consistent'
        self.db_columns[name5b] = None

        name6 = 'exposure_meter_not_saturated_check'
        self.names.append(name6)
        self.kpf_data_levels[name6] = ['L0']
        self.descriptions[name6] = 'Check if 2+ reduced EM pixels are within 90% of saturation in EM-SCI or EM-SKY.'
        self.data_types[name6] = 'int'
        self.spectrum_types[name6] = ['all', ]
        self.fits_keywords[name6] = 'EMSAT'
        self.fits_comments[name6] = 'QC: EM not saturated'
        self.db_columns[name6] = None

        name7 = 'exposure_meter_flux_not_negative_check'
        self.names.append(name7)
        self.kpf_data_levels[name7] = ['L0']
        self.descriptions[name7] = 'Check for negative flux in the EM-SCI and EM-SKY by looking for 20 consecuitive pixels in the summed spectra with negative flux.'
        self.data_types[name7] = 'int'
        self.spectrum_types[name7] = ['all', ]
        self.fits_keywords[name7] = 'EMNEG'
        self.fits_comments[name7] = 'QC: EM not negative flux'
        self.db_columns[name7] = None

        name8 = 'D2_lfc_flux_check'
        self.names.append(name8)
        self.kpf_data_levels[name8] = ['2D']
        self.descriptions[name8] = 'Check if an LFC frame that goes into a master has sufficient flux'
        self.data_types[name8] = 'int'
        self.spectrum_types[name8] = ['LFC', ]
        self.fits_keywords[name8] = 'LFC2DFOK'
        self.fits_comments[name8] = 'QC: LFC flux meets threshold of 4000 counts'
        self.db_columns[name8] = None

        name9 = 'data_2D_bias_low_flux_check'
        self.names.append(name9)
        self.kpf_data_levels[name9] = ['2D']
        self.descriptions[name9] = 'Check if flux is low in bias exposure.'
        self.data_types[name9] = 'int'
        self.spectrum_types[name9] = ['Bias', ]
        self.fits_keywords[name9] = 'LOWBIAS'
        self.fits_comments[name9] = 'QC: 2D bias low flux check'
        self.db_columns[name9] = None

        name10 = 'data_2D_dark_low_flux_check'
        self.names.append(name10)
        self.kpf_data_levels[name10] = ['2D']
        self.descriptions[name10] = 'Check if flux is low in dark exposure.'
        self.data_types[name10] = 'int'
        self.spectrum_types[name10] = ['Dark', ]
        self.fits_keywords[name10] = 'LOWDARK'
        self.fits_comments[name10] = 'QC: 2D dark low flux check'
        self.db_columns[name10] = None

        name11 = 'data_L1_red_green_check'
        self.names.append(name11)
        self.kpf_data_levels[name11] = ['L1']
        self.data_types[name11] = 'int'
        self.spectrum_types[name11] = ['all', ]
        self.descriptions[name11] = 'Check if red/green data are present in L1 with expected shapes.'
        self.fits_keywords[name11] = 'DATAPRL1'
        self.fits_comments[name11] = 'QC: L1 red and green data present check'
        self.db_columns[name11] = None

        name12 = 'data_L1_CaHK_check'
        self.names.append(name12)
        self.kpf_data_levels[name12] = ['L1']
        self.descriptions[name12] = 'Check if CaHK data is present in L1 with expected shape.'
        self.data_types[name12] = 'int'
        self.spectrum_types[name12] = ['all', ]
        self.fits_keywords[name12] = 'CaHKPRL1'
        self.fits_comments[name12] = 'QC: L1 CaHK present check'
        self.db_columns[name12] = None

        name13 = 'data_L2_check'
        self.names.append(name13)
        self.kpf_data_levels[name13] = ['L2']
        self.descriptions[name13] = 'Check if all data are present in L2.'
        self.data_types[name13] = 'int'
        self.spectrum_types[name13] = ['all', ]
        self.fits_keywords[name13] = 'DATAPRL2'
        self.fits_comments[name13] = 'QC: L2 data present check'
        self.db_columns[name13] = None
        
        name14 = 'data_2D_CaHK_check'
        self.names.append(name14)
        self.kpf_data_levels[name14] = ['2D']
        self.descriptions[name14] = 'Check if CaHK CCD data is present with expected array sizes.'
        self.data_types[name14] = 'int'
        self.spectrum_types[name14] = ['all', ]
        self.fits_keywords[name14] = 'CaHKPR2D'
        self.fits_comments[name14] = 'QC: 2D CaHK data present check'
        self.db_columns[name14] = None

        name15 = 'data_2D_red_green_check'
        self.names.append(name15)
        self.kpf_data_levels[name15] = ['2D']
        self.descriptions[name15] = 'Check if red/green CCD data is present with expected array sizes.'
        self.data_types[name15] = 'int'
        self.spectrum_types[name15] = ['all', ]
        self.fits_keywords[name15] = 'DATAPR2D'
        self.fits_comments[name15] = 'QC: 2D red and green data present check'
        self.db_columns[name15] = None

        name16 = 'add_kpfera'
        self.names.append(name16)
        self.kpf_data_levels[name16] = ['L0', '2D', 'L1', 'L2']
        self.descriptions[name16] = 'Not a QC test.  The QC module is used to add the KPFERA keyword to all files.'
        self.data_types[name16] = 'float'
        self.spectrum_types[name16] = ['all', ]
        self.fits_keywords[name16] = 'KPFERA'
        self.fits_comments[name16] = 'Current era of KPF observations'
        self.db_columns[name16] = None

        # Integrity checks
        if len(self.names) != len(self.kpf_data_levels):
            raise ValueError("Length of kpf_data_levels list does not equal number of entries in descriptions dictionary.")

        if len(self.names) != len(self.descriptions):
            raise ValueError("Length of names list does not equal number of entries in descriptions dictionary.")

        if len(self.names) != len(self.data_types):
            raise ValueError("Length of data_types list does not equal number of entries in data_types dictionary.")

        if len(self.names) != len(self.spectrum_types):
            raise ValueError("Length of spectrum_types list does not equal number of entries in data_types dictionary.")

        if len(self.names) != len(self.fits_keywords):
            raise ValueError("Length of fits_keywords list does not equal number of entries in fits_keywords dictionary.")

        if len(self.names) != len(self.fits_comments):
            raise ValueError("Length of fits_comments list does not equal number of entries in fits_comments dictionary.")

        if len(self.names) != len(self.db_columns):
            raise ValueError("Length of db_columns list does not equal number of entries in db_columns dictionary.")

        keys_list = self.data_types.keys()
        for key in keys_list:
            dt = self.data_types[key]
            if dt not in ['string','int','float']:
                err_str = "Error in data type: " + dt
                raise ValueError(err_str)


    def list_qc_metrics(self):
        """
        Method to print a formatted block of the available QC checks and their
        characteristics, sorted by the data level that the QC check accepts.
        """
        qc_names = self.names
        
        for data_level in ['L0', '2D', 'L1', 'L2']:
            print(f'\033[1mQuality Control tests for {data_level}:\033[0m')
            for qc_name in qc_names:
    
                kpf_data_levels = self.kpf_data_levels[qc_name]
                data_type = self.data_types[qc_name]
                spectrum_type = self.spectrum_types[qc_name]
                keyword = self.fits_keywords[qc_name]
                comment = self.fits_comments[qc_name]
                db_column = self.db_columns[qc_name]
                description = self.descriptions[qc_name]
    
                if data_level in self.kpf_data_levels[qc_name]:
                    print('   \033[1mQC Name:\033[0m ' + qc_name)
                    print('      \033[1mDescription:\033[0m ' + description)
                    print('      \033[1mData levels:\033[0m ' + str(kpf_data_levels))
                    print('      \033[1mData type:\033[0m ' + data_type)
                    print('      \033[1mSpectrum type:\033[0m ' + str(spectrum_type))
                    print('      \033[1mKeyword:\033[0m ' + keyword)
                    print('      \033[1mComment:\033[0m ' + comment)
                    print('      \033[1mDatabase column:\033[0m ' + str(db_column))
                    print()


#####################################################################
#
# Superclass QC is normally not to be called directly (although it is not an abstract class, per se).
#

class QC:

    """
    Description:
        This superclass defines QC functions in general and has common methods across
        subclasses QCL0, QC2D, QCL1, and QCL2.  It also includes QC checks that apply 
        to all data levels.

    Class Attributes:
        kpf_object: Returned from function KPF0.from_fits(fits_filename,data_type),
            which is wrapped by function read_fits in this module.
        qcdefinitions (QCDefinitions object): Returned from constructor of QCDefinitions class.

    """

    def __init__(self, kpf_object, logger=None):
        self.kpf_object = kpf_object
        self.qcdefinitions = QCDefinitions()
        self.logger = logger if logger is not None else DummyLogger()
        

    def add_qc_keyword_to_header(self, qc_name, value, debug=False):

        if str(type(value)) == "<class 'bool'>":
            if value == True:
                value = 1
            else:
                value = 0
        
        keyword = self.qcdefinitions.fits_keywords[qc_name]
        comment = self.qcdefinitions.fits_comments[qc_name]

        self.kpf_object.header['PRIMARY'][keyword] = (value,comment)
        if debug:
            print('---->add_qc_keyword_to_header: qc_name, keyword, value, comment = {}, {}, {}, {}'.format(qc_name,keyword,value,comment))


    def not_junk_check(self, junk_ObsIDs_csv='/data/reference/Junk_Observations_for_KPF.csv', debug=False):
        """
        This Quality Control method can be used in any of the data levels (L0/2D/L1/L2) 
        so it is included in the superclass. 
        It checks if the obsID of the input is in the list of junked files.
    
        Args:
             kpfobs - a KPF L0/2D/L1/L2 object
             junk_ObsIDs_csv - a CSV with ObsIDs in the first column
                               and a column header of 'observation_id'.
                               That is, the first few lines of the file will look like this:
                                   observation_id
                                   KP.20230621.27498.77
                                   KP.20230621.27611.73
                                   KP.20220516.57354.11
    
             debug - an optional flag.  If True, verbose output will be printed.
    
         Returns:
             QC_pass - a boolean signifying that the input(s) are not junk (i.e., = False if junk)
        """
        
        QC_pass = True  # Assume not junk unless explicitly listed in junk_ObsIDs_csv
        
        try:
            filename = self.kpf_object.header['PRIMARY']['OFNAME'] # 'KP.20231129.11266.37.fits' / Filename of output file
        except:
            filename = 'this file'
        obsID = filename[:20]
    
        # read list of junk files
        if os.path.exists(junk_ObsIDs_csv):
            df_junk = pd.read_csv(junk_ObsIDs_csv)
            if debug:
                self.logger.info(f'Read the junk file {junk_ObsIDs_csv}.')
        else:
            self.logger.info(f"The file {junk_ObsIDs_csv} does not exist.")
            return QC_pass
        
        QC_pass = not (df_junk['observation_id'].isin([obsID])).any()
        if debug:
            self.logger.info(f'{filename} is a Junk file: ' + str(not QC_pass[i]))
    
        return QC_pass


    def add_kpfera(self, kfpera_csv='/code/KPF-Pipeline/static/kpfera_definitions.csv', debug=False):
        """
        This is not a Quality Control method.  
        The goal of this method is to add the KPFERA keyword to all KPF files.
        This keyword was created in February 2024, during the first service mission;
        thus, L0 files before then (with KPFERA = 1.0 and 1.5) do not have 
        this defined.  By running a recipe with the L0 checks as the first 
        element in a processing recipe involving L0 files, the KPFERA keyword
        is guaranteed to be in the primary header of every kpf object.
    
        Args:
             kpfobs - a KPF L0/2D/L1/L2 object
             kfpera_csv - a CSV the KPF era definitions    
             debug - an optional flag.  If True, verbose output will be printed.
    
         Returns:
             KPFERA - a string the the KPFERA (e.g., '1.0') for the input file
        """
        
        KPFERA = float('0.0')
        
        try:
            filename = self.kpf_object.header['PRIMARY']['OFNAME'] # 'KP.20231129.11266.37.fits' / Filename of output file
        except:
            filename = 'this file'
        ObsID = filename[:20]
        if len(ObsID.split('.')) != 4:
            if debug:
                self.logger.info(f'ObsID = {kfpera_csv} is not in the correct format.')
            return KPFERA
        datetime_ObsID = get_datetime_obsid(ObsID)
        self.logger.info(f"The datetime of ObsID is {datetime_ObsID}.")

        if os.path.exists(kfpera_csv):
            try:
                df_kpfera = pd.read_csv(kfpera_csv)
                if debug:
                    self.logger.info(f'Read the KPFERA file {kfpera_csv}.')
                nrows = len(df_kpfera)
                for i in np.arange(nrows):
                    starttime = datetime.strptime(df_kpfera.iloc[i][1].strip(), '%Y-%m-%d %H:%M:%S') 
                    stoptime  = datetime.strptime(df_kpfera.iloc[i][2].strip(), '%Y-%m-%d %H:%M:%S')
                    if (datetime_ObsID > starttime) and (datetime_ObsID < stoptime):
                        KPFERA = float(df_kpfera.iloc[i][0])
                        if debug:
                            self.logger.info(f'Setting KPFERA = {KPFERA}')
            except Exception as e:
                self.logger.info(f"Exceptions: {e}")
                return None
        else:
            self.logger.error(f"The file {kfpera_csv} does not exist.")
        
        if debug:
            self.logger.info(f'The KPFERA of {filename} is: ' + str(KPFERA))
    
        return KPFERA

#####################################################################

class QCL0(QC):

    """
    Description:
        This class inherits the QC superclass and defines QC functions for L0 files.
        Since the policy is to not modify an L0 FITS file in the archive location
        /data/kpf/L0/yyyymmdd, the class operates on the FITS object that will
        elevate to a higher data level. The QC info is inherited via the FITS header
        and will be prograted downstream in the data-reduction pipeline, and will
        eventually be written to an output FITS file.

    Class Attributes:
        data_type (string): Data type in terms of project (e.g., KPF).
        kpf_object (astropy.io object): Returned from function KPF0.from_fits(fits_filename,data_type),
            which is wrapped by function read_fits in this module.
        qcdefinitions (QCDefinitions object): Returned from constructor of QCDefinitions class.

    Example python code to illustrate usage of this module in calling program:

        import modules.quality_control.src.quality_control as qc

        qc.what_am_i()

        in_file = '/code/KPF-Pipeline/KP.20230828.40579.55.fits'
        out_file = '/code/KPF-Pipeline/junk.fits'


        kpf_object = from_fits('KPF',in_file)
        qcl0 = qc.QCL0(kpf_object)
        name = 'jarque_bera_test_red_amp1'
        value = 3.14159256
        qcl0.add_qc_keyword_to_header(name,value)
        to_fits(qcl0.kpf_object,out_file)
    """

    # Call superclass.
    def __init__(self,kpf_object):
        super().__init__(kpf_object)


    def L0_data_products_check(self, debug=False):
        """
        This Quality Control function checks if the expected data_products 
        in an L0 file are present and if their data extensions are populated 
        with arrays of non-zero size.
        
        Args:
             L0 - an L0 object
             debug - an optional flag.  If True, missing data products are noted.
    
         Returns:
             QC_pass - a boolean signifying that the QC passed (True) for failed (False)
        """
        
        L0 = self.kpf_object
        
        # Determine which extensions should be in the L0 file.
        # First add the triggrered cameras (Green, Red, CaHK, ExpMeter) to list of data products
        trigtarg = L0.header['PRIMARY']['TRIGTARG']
        if len(trigtarg) > 0:
            data_products = trigtarg.split(',')
        # add Guider
        if hasattr(L0, 'GUIDER_AVG'):
            data_products.append('Guider')
        if hasattr(L0, 'guider_avg'):  # some early files had lower case
            data_products.append('Guider')
        # add Telemetry
        if hasattr(L0, 'TELEMETRY'):
            data_products.append('Telemetry')
        # add Pyrheliometer
        if hasattr(L0, 'SOCAL PYRHELIOMETER'):
            data_products.append('Pyrheliometer')
        if debug:
            self.logger.info('Data products that are supposed to be in this L0 file: ' + str(data_products))
     
        # Use helper funtion to get data products and check their characteristics.
        QC_pass = True
        data_products_present = get_data_products_L0(L0)
        if debug:
            self.logger.info('Data products in L0 file: ' + str(data_products_present))
    
        # Check for specific data products
        possible_data_products = ['Green', 'Red', 'CaHK', 'ExpMeter', 'Guider', 'Telemetry', 'Pyrheliometer']
        for dp in possible_data_products:
            if dp in data_products:
                if not dp in data_products_present:
                    QC_pass = False
                    if debug:
                        self.logger.info(dp + ' not present in L0 file. QC(L0_data_products_check) failed.')
        
        return QC_pass


    def L0_header_keywords_present_check(self, essential_keywords=['auto'], debug=False):
        """
        This Quality Control function checks if a specified set of FITS header keywords are present.
        
        Args:
             L0 - an L0 object
             essential_keywords - an optional list of keywords to check.  If set to ['auto'], 
             then a default list of keywords will be checked. 
             debug - an optional flag.  If True, missing data products are noted.
    
         Returns:
             QC_pass - a boolean signifying that the QC passed (True) for failed (False)
        """
        
        L0 = self.kpf_object

        if essential_keywords == ['auto']:
             essential_keywords = [
                 'DATE-BEG',  # Start of exposure from kpfexpose
                 'DATE-MID',  # Halfway point of the exposure (unweighted)
                 'DATE-END',  # End of exposure
                 'EXPTIME',   # Requested exposure time
                 'ELAPSED',   # Actual exposure time
                 'PROGNAME',  # Program name from kpfexpose
                 'OBJECT',    # Object name
                 'TARGRA',    # Right ascension [hr] from DCS
                 'TARGDEC',   # Declination [deg] from DCS
                 'TARGEPOC',  # Target epoch from DCS
                 'TARGEQUI',  # Target equinox from DCS
                 'TARGPLAX',  # Target parallax [arcsec] from DCS
                 'TARGPMDC',  # Target proper motion [arcsec/yr] in declination from DCS
                 'TARGPMRA',  # Target proper motion [s/yr] in right ascension from DCS
                 'TARGRADV',  # Target radial velocity [km/s]
                 'AIRMASS',   # Airmass from DCS
                 'PARANTEL',  # Parallactic angle of the telescope from DCS
                 'HA',        # Hour angle
                 'EL',        # Elevation [deg]
                 'AZ',        # Azimuth [deg]
                 'LST',       # Local sidereal time
                 'GAIAID',    # GAIA Target name
                 '2MASSID',   # 2MASS Target name
                 'GAIAMAG',   # GAIA G band magnitude
                 '2MASSMAG',  # 2MASS J band magnitude
                 'TARGTEFF',  # Target effective temperature (K)
                 'OCTAGON',   # Selected octagon calibration source (not necessarily powered on)
                 'TRIGTARG',  # Cameras that were sent triggers
                 'IMTYPE',    # Image Type
                 'CAL-OBJ',   # Calibration fiber source
                 'SKY-OBJ',   # Sky fiber source
                 'SCI-OBJ',   # Science fiber source
                 'AGITSTA',   # Agitator status
             ] 
    
        QC_pass = True
        for keyword in essential_keywords:
            if keyword not in L0.header['PRIMARY']:
                QC_pass = False
                if debug:
                    print('The keyword ' + keyword + ' is missing from the primary header.')
        
        return QC_pass


    def L0_datetime_checks(self, debug=False):
        """
        This QC module performs the following checks on datetimes in the L0 primary header
        and in the Exposure Meter table (if present).  The timing checks have precision 
        thresholds to only catch significant timing errors and not trigger on small 
        differences related to machine precision or dead time in the Exposure Meter detector.
        This method returns True only if all checks pass.
        
            Time ordering: 
                DATE-BEG < DATE-MID < DATE-END
            Duration consistency: 
                DATE-END - DATE-BEG = ELAPSED
            Consistency between Green/Red and overall timing:
                DATE-BEG = GRDATE-B
                DATE-BEG = RDDATE-B
                DATE-END = GRDATE-E
                DATE-END = RDDATE-E
            Consistency between Exposure Meter times (Date-Beg, etc.) and overall timing:
                Date-Beg = DATE-BEG
                Date-End = DATE-END
        """
    
        L0 = self.kpf_object
        date_format = "%Y-%m-%dT%H:%M:%S.%f"
        QC_pass = True
    
        time_precision_threshold     = 0.1 # sec - threshold for DATE-BEG, etc.
        time_precision_threshold_exp = 1.0 # sec - threshold for times involving the exposure meter -- account for EM dead time and only catch bad errors
        
        # First check that the appropriate keywords are present
        essential_keywords = ['DATE-BEG', 'DATE-MID', 'DATE-END', 'ELAPSED']
        for keyword in essential_keywords:
            if keyword not in L0.header['PRIMARY']:
                if debug:
                    print(f'Missing keyword: {keyword}')
                QC_pass = False
        if not QC_pass:
            return QC_pass
        
        # Check that dates are ordered correctly
        date_beg = datetime.strptime(L0.header['PRIMARY']['DATE-BEG'], date_format)
        date_mid = datetime.strptime(L0.header['PRIMARY']['DATE-MID'], date_format)
        date_end = datetime.strptime(L0.header['PRIMARY']['DATE-END'], date_format)
        elapsed  = float(L0.header['PRIMARY']['ELAPSED'])
        if (date_end < date_mid) or (date_mid < date_beg):
            QC_pass = False
        
        # Check that DATE-BEG + ELAPSE = DATE-END
        if abs((date_end - date_beg).total_seconds() - elapsed) > time_precision_threshold:
            if debug:
                print(f'(DATE-END - DATE-BEG) - ELASPED = {abs((date_end - date_beg).total_seconds() - elapsed)} sec > {time_precision_threshold} sec')
            QC_pass = False
            
        # Check that GRDATE-B/RDDATE-B are consistent with DATE-BEG, etc.
        data_products = get_data_products_L0(L0)
        if 'Green' in data_products:
            if 'GRDATE-B' not in L0.header['PRIMARY']:
                if debug:
                    print(f'Missing keyword: GRDATE-B')
                QC_pass = False
                return QC_pass
            else:
                grdate_b = datetime.strptime(L0.header['PRIMARY']['GRDATE-B'], date_format)
                if abs((date_beg - grdate_b).total_seconds()) > time_precision_threshold:
                    if debug:
                        print(f'abs(DATE-BEG - GRDATE-B) = {abs((date_beg - grdate_b).total_seconds())} sec > {time_precision_threshold} sec')
                    QC_pass = False
            if 'GRDATE-E' not in L0.header['PRIMARY']:
                if debug:
                    print(f'Missing keyword: GRDATE-E')
                QC_pass = False
                return QC_pass
            else:
                grdate_e = datetime.strptime(L0.header['PRIMARY']['GRDATE-E'], date_format)
                if abs((date_end - grdate_e).total_seconds()) > time_precision_threshold:
                    if debug:
                        print(f'abs(DATE-END - GRDATE-E) = {abs((date_end - grdate_e).total_seconds())} sec > {time_precision_threshold} sec')
                    QC_pass = False
        if 'Red' in data_products:
            if 'RDDATE-B' not in L0.header['PRIMARY']:
                if debug:
                    print(f'Missing keyword: RDDATE-B')
                QC_pass = False
                return QC_pass
            else:
                rddate_b = datetime.strptime(L0.header['PRIMARY']['RDDATE-B'], date_format)
                if abs((date_beg - rddate_b).total_seconds()) > time_precision_threshold:
                    if debug:
                        print(f'abs(DATE-BEG - RDDATE-B) = {abs((date_beg - rddate_b).total_seconds())} sec > {time_precision_threshold} sec')
                    QC_pass = False
            if 'RDDATE-E' not in L0.header['PRIMARY']:
                if debug:
                    print(f'Missing keyword: RDDATE-E')
                QC_pass = False
                return QC_pass
            else:
                rddate_e = datetime.strptime(L0.header['PRIMARY']['RDDATE-E'], date_format)
                if abs((date_end - rddate_e).total_seconds()) > time_precision_threshold:
                    if debug:
                        print(f'abs(DATE-END - RDDATE-E) = {abs((date_end - rddate_e).total_seconds())} sec > {time_precision_threshold} sec')
                    QC_pass = False
        if ('Green' in data_products) and ('Red' in data_products) and QC_pass:
            if abs((grdate_b - rddate_b).total_seconds()) > time_precision_threshold: 
                if debug:
                    print(f'abs(GRDATE-B - RDDATE-B) = {abs((grdate_b - rddate_b).total_seconds())} sec > {time_precision_threshold} sec')
                QC_pass = False
            if abs((grdate_e - rddate_e).total_seconds()) > time_precision_threshold: 
                if debug:
                    print(f'abs(GRDATE-E - RDDATE-E) = {abs((grdate_e - rddate_e).total_seconds())} sec > {time_precision_threshold} sec')
                QC_pass = False
     
        if 'ExpMeter' in data_products:
            if 'Date-Beg-Corr' in L0['EXPMETER_SCI'].columns:
                exp_date_beg = datetime.strptime(L0['EXPMETER_SCI'].iloc[0]['Date-Beg-Corr'], date_format)
                exp_date_end = datetime.strptime(L0['EXPMETER_SCI'].iloc[-1]['Date-End-Corr'], date_format)
            else:
                exp_date_beg = datetime.strptime(L0['EXPMETER_SCI'].iloc[0]['Date-Beg'], date_format)
                exp_date_end = datetime.strptime(L0['EXPMETER_SCI'].iloc[-1]['Date-End'], date_format)
            if 'Green' in data_products:
                if abs((exp_date_beg - grdate_b).total_seconds()) > time_precision_threshold_exp:
                    if debug:
                        print(f"abs(L0['EXPMETER_SCI'].iloc[0]['Date-Beg-Corr'] - GRDATE-B) = {abs((exp_date_beg - grdate_b).total_seconds())} sec > {time_precision_threshold_exp} sec")
                    QC_pass = False
                if abs((exp_date_end - grdate_e).total_seconds()) > time_precision_threshold_exp:
                    if debug:
                        print(f"abs(L0['EXPMETER_SCI'].iloc[-1]['Date-End-Corr'] - GRDATE-E) = {abs((exp_date_end - grdate_e).total_seconds())} sec > {time_precision_threshold_exp} sec")
                    QC_pass = False
            if 'Red' in data_products:
                if abs((exp_date_beg - rddate_b).total_seconds()) > time_precision_threshold_exp:
                    if debug:
                        print(f"abs(L0['EXPMETER_SCI'].iloc[0]['Date-Beg-Corr'] - RDDATE-B) = {abs((exp_date_beg - rddate_b).total_seconds())} sec > {time_precision_threshold_exp} sec")
                    QC_pass = False
                if abs((exp_date_end - rddate_e).total_seconds()) > time_precision_threshold_exp:
                    if debug:
                        print(f"abs(L0['EXPMETER_SCI'].iloc[-1]['Date-End-Corr'] - RDDATE-E) = {abs((exp_date_end - rddate_e).total_seconds())} sec > {time_precision_threshold_exp} sec")
                    QC_pass = False
        
        return QC_pass    


    def exposure_meter_not_saturated_check(self, debug=False):
        """
        This Quality Control function checks if 2 or more reduced pixels in an exposure
        meter spectrum is within 90% of saturated.  The check is applied to the EM-SCI 
        and EM-SKY fibers and returns False if saturation is detected in either.  
        Note that this check only works for L0 files with the EXPMETER_SCI and 
        EXPMETER_SKY extensions present.
        
        Args:
             L0 - an L0 object
             fiber ('SCI' [default value] or 'SKY) - the EM fiber output to be tested
             debug - an optional flag.  If True, missing data products are noted.
    
         Returns:
             QC_pass - a boolean signifying that the QC passed (True) for failed (False)
        """

        saturation_level = 1.93e6 # saturation level in reduced EM spectra (in data frame)
        saturation_fraction = 0.9 
        
        # Read and condition the table of Exposure Meter Data
        L0 = self.kpf_object
        if hasattr(L0, 'EXPMETER_SCI') and hasattr(L0, 'EXPMETER_SKY'):
            if (L0['EXPMETER_SCI'].size > 1) and (L0['EXPMETER_SKY'].size > 1):
                pass
            else:
                return False
        else:
            return True # pass test if no exposure meter data present
        EM_sat_SCI = L0['EXPMETER_SCI'].copy()
        EM_sat_SKY = L0['EXPMETER_SKY'].copy()
        columns_to_drop_SCI = [col for col in EM_sat_SCI.columns if col.startswith('Date')]
        columns_to_drop_SKY = [col for col in EM_sat_SKY.columns if col.startswith('Date')]
        EM_sat_SCI.drop(columns_to_drop_SCI, axis=1, inplace=True)
        EM_sat_SKY.drop(columns_to_drop_SKY, axis=1, inplace=True)
        if len(EM_sat_SCI) >= 3:  # drop first and last rows if nrows >= 3
            EM_sat_SCI = EM_sat_SCI.iloc[1:-1]
            EM_sat_SKY = EM_sat_SKY.iloc[1:-1]
        
        # Determine the saturation fraction
        for col in EM_sat_SCI.columns:
            try: # only apply to columns with wavelengths as headers
                float_col_title = float(col)
                EM_sat_SCI[col] = EM_sat_SCI[col] / saturation_level 
            except ValueError:
                pass 
        for col in EM_sat_SKY.columns:
            try: 
                float_col_title = float(col)
                EM_sat_SKY[col] = EM_sat_SKY[col] / saturation_level 
            except ValueError:
                pass 

        saturated_elements_SCI = (EM_sat_SCI > saturation_fraction).sum().sum()
        saturated_elements_SKY = (EM_sat_SKY > saturation_fraction).sum().sum()
        total_elements = EM_sat_SCI.shape[0] * EM_sat_SCI.shape[1]
        saturated_fraction_threshold = 1.5 / EM_sat_SCI.shape[1]
        
        if saturated_elements_SCI / total_elements > saturated_fraction_threshold:
            QC_pass = False
        elif saturated_elements_SKY / total_elements > saturated_fraction_threshold:
            QC_pass = False
        else: 
            QC_pass = True
            
        return QC_pass


    def exposure_meter_flux_not_negative_check(self, debug=False):
        """
        This Quality Control function checks if 20 or more consecutive elements of the 
        exposure meter spectra are negative.  Negative flux usually indicates 
        over-subtraction of bias from the raw EM images.  The check is applied to the 
        EM-SCI and EM-SKY fibers and returns False if negative flux is detected in 
        either.  Note that this check only works for L0 files with the EXPMETER_SCI and 
        EXPMETER_SKY extensions present.
        
        Args:
             L0 - an L0 object
             fiber ('SCI' [default value] or 'SKY) - the EM fiber output to be tested
             debug - an optional flag.  If True, missing data products are noted.
    
         Returns:
             QC_pass - a boolean signifying that the QC passed (True) for failed (False)
        """

        N_in_a_row = 20 # number of negative flux elements in a row that triggers QC failure
        
        # Read and condition the table of Exposure Meter Data
        L0 = self.kpf_object
        if hasattr(L0, 'EXPMETER_SCI') and hasattr(L0, 'EXPMETER_SKY'):
            if (L0['EXPMETER_SCI'].size > 1) and (L0['EXPMETER_SKY'].size > 1):
                pass
            else:
                return False
        else:
            return True # pass test if no exposure meter data present
        EM_SCI = L0['EXPMETER_SCI'].copy()
        EM_SKY = L0['EXPMETER_SKY'].copy()
        columns_to_drop_SCI = [col for col in EM_SCI.columns if col.startswith('Date')]
        columns_to_drop_SKY = [col for col in EM_SKY.columns if col.startswith('Date')]
        EM_SCI.drop(columns_to_drop_SCI, axis=1, inplace=True)
        EM_SKY.drop(columns_to_drop_SKY, axis=1, inplace=True)
        counts_SCI = EM_SCI.sum(axis=0).values
        counts_SKY = EM_SKY.sum(axis=0).values
        
        # Determine if the spectra have significant negative flux
        negative_mask_SCI = counts_SCI < 0 # spectral elements with negative flux
        negative_mask_SKY = counts_SKY < 0
        window = np.ones(N_in_a_row, dtype=int) # window to convolve with spectra
        conv_result_SCI = convolve1d(negative_mask_SCI.astype(int), window, mode='constant', cval=0)
        conv_result_SKY = convolve1d(negative_mask_SKY.astype(int), window, mode='constant', cval=0)
        has_consec_negs_SCI = np.any(conv_result_SCI == N_in_a_row)
        has_consec_negs_SKY = np.any(conv_result_SKY == N_in_a_row)

        if has_consec_negs_SCI or has_consec_negs_SKY:
            QC_pass = False
        else: 
            QC_pass = True
            
        return QC_pass


#####################################################################

class QC2D(QC):

    """
    Description:
        This class inherits QC superclass and defines QC functions for 2D files.

    Class Attributes:
        kpf_object (astropy.io object): Returned from function KPF0.from_fits(fits_filename,data_type),
            which is wrapped by function read_fits in this module.
        qcdefinitions (QCDefinitions object): Returned from constructor of QCDefinitions class.
    """

    # Call superclass.
    def __init__(self,kpf_object):
        super().__init__(kpf_object)

    def data_2D_red_green_check(self,debug=False):
        """
        This Quality Control function checks if the 2D data exists for both
        the red and green chips and checks that the sizes of the arrays are as expected.
    
        Args:
             debug - an optional flag.  If True, prints shapes of CCD arrays and other comments.
    
         Returns:
             QC_pass - a boolean signifying that all of the data exists as expected
        """
    
        D2 = self.kpf_object


        if debug:
            print(D2.info())
            type_D2 = type(D2)
            print("type_2D = ",type_D2)
            print("D2 = ",D2)

        QC_pass = True
    
        extensions = D2.extensions
    
        if 'GREEN_CCD' in extensions:
        
            if debug:
                print("GREEN_CCD exists")
                print("data_shape =", np.shape(D2["GREEN_CCD"]))
            
            if np.shape(D2["GREEN_CCD"]) != (4080, 4080):  
                QC_pass = False
            
        else:
            if debug:
                print("GREEN_CCD does not exist")
            QC_pass = False       
        
        if 'RED_CCD' in extensions:
        
            if debug:
                print("RED_CCD exists")
                print("data_shape =", np.shape(D2["RED_CCD"]))
            
            if np.shape(D2["RED_CCD"]) != (4080, 4080):  
                QC_pass = False
            
        else:
            if debug:
                print("RED_CCD does not exist")
            QC_pass = False    
        
        return QC_pass

    def data_2D_CaHK_check(self,debug=False):
        """
        This Quality Control function checks if the 2D data exists for the
        Ca H&K chip and checks that the size of the array is as expected.

        Args:
             debug - an optional flag.  If True, prints shape of CaHK CCD array.
    
        Returns:
             QC_pass - a boolean signifying that all of the data exists as expected
        """
    
        D2 = self.kpf_object

        if debug:
            print(D2.info())
            type_D2 = type(D2)
            print("type_2D = ",type_D2)
            print("D2 = ",D2)

        QC_pass = True
    
        extensions = D2.extensions
    
        if 'CA_HK' in extensions:
        
            if debug:
                print("CA_HK exists")
                print("data_shape =", np.shape(D2["CA_HK"]))
            
            if np.shape(D2["CA_HK"]) == (0,):  
                QC_pass = False
            
        else:
            if debug:
                print("CA_HK does not exist")
            QC_pass = False       
        
        return QC_pass

    def data_2D_bias_low_flux_check(self,debug=False):
        """
        This Quality Control function checks if the flux is low
        (mean flux < 10) for a bias exposure.

        Args:
             debug - an optional flag.  If True, prints mean flux in each CCD.

        Returns:
             QC_pass - a boolean signifying that all of the data exists as expected
        """
    
        D2 = self.kpf_object

        if debug:
            print(D2.info())
            type_D2 = type(D2)
            print("type_2D = ",type_D2)
            print("D2 = ",D2)

        QC_pass = True
        extensions = D2.extensions

        mean_GREEN = D2["GREEN_CCD"].flatten().mean()
        mean_RED = D2["RED_CCD"].flatten().mean()

        if debug:
            print("Mean GREEN_CCD flux =", np.round(mean_GREEN, 2))
            print("Mean RED_CCD flux =", np.round(mean_RED, 2))
            print("Max allowed mean flux =", 10)

        if (mean_GREEN > 10) | (mean_RED > 10):
            if debug:
                print("One of the CCDs has a high flux")
            QC_pass = False

        return QC_pass

    def data_2D_dark_low_flux_check(self,debug=False):
        """
        This Quality Control function checks if the flux is low
        (mean flux < 10) for a dark exposure.

        Args:
             debug - an optional flag.  If True, prints mean flux in each CCD.

        Returns:
             QC_pass - a boolean signifying that all of the data exists as expected
        """
    
        D2 = self.kpf_object

        if debug:
            print(D2.info())
            type_D2 = type(D2)
            print("type_2D = ",type_D2)
            print("D2 = ",D2)
    
        QC_pass = True
        extensions = D2.extensions

        mean_GREEN = D2["GREEN_CCD"].flatten().mean()
        mean_RED = D2["RED_CCD"].flatten().mean()

        if debug:
            print("Mean GREEN_CCD flux =", np.round(mean_GREEN, 2))
            print("Mean RED_CCD flux =", np.round(mean_RED, 2))
            print("Max allowed mean flux =", 10)

        if (mean_GREEN > 10) | (mean_RED > 10):
            if debug:
                print("One of the CCDs has a high flux")
            QC_pass = False
        
        return QC_pass

    def D2_lfc_flux_check(self, threshold=4000, debug=False):
        """
        This Quality Control function checks if the flux values in the green and red chips of the
        given 2D file are above a defined threshold at the 98th percentile.
        
        Args:
            debug
        
        Returns:
            QC_Test (bool): True if both green and red channels have 98th percentile values above the
                            threshold, False otherwise.
        """
        
        Two_D = self.kpf_object
        green_counts = Two_D['GREEN_CCD'].data
        red_counts = Two_D['RED_CCD'].data
        
        QC_Test = True
        if debug:
            print("******Green - 98th percentile counts: " + str(np.percentile(green_counts, 98)))
            print("******Red - 98th percentile counts: " + str(np.percentile(red_counts, 98)))
        if np.percentile(green_counts, 98) < threshold or np.percentile(red_counts, 98) < threshold:
            QC_Test = False
           
        return QC_Test

#####################################################################

class QCL1(QC):

    """
    Description:
        This class inherits QC superclass and defines QC functions for L1 files.

    Class Attributes:
        kpf_object (astropy.io object): Returned from function KPF0.from_fits(fits_filename,data_type),
            which is wrapped by function read_fits in this module.
        qcdefinitions (QCDefinitions object): Returned from constructor of QCDefinitions class.
    """

    # Call superclass.
    def __init__(self,kpf_object):
        super().__init__(kpf_object)


    def monotonic_wavelength_solution_check(self,debug=False):
        """
        This Quality Control function checks if a wavelength solution is
        monotonic, specifically if wavelength decreases (or stays constant) with
        increasing array index.

        Args:
             debug - an optional flag.  If True, nonmonotonic orders/orderlets will be noted with
                     print statements and plots.

         Returns:
             QC_pass - a boolean signifying that every order/orderlet is monotonic (or not)
             bad_orders - an array of strings listing the nonmonotonic orders and orderlets
        """

        L1 = self.kpf_object

        if debug:
            print(L1.info())
            type_L1 = type(L1)
            print("type_L1 = ",type_L1)
            print("L1 = ",L1)

        QC_pass = True
        bad_orders = []

        import numpy as np
        if debug:
            import matplotlib.pyplot as plt

        # Define wavelength extensions in L1
        extensions = [p + s for p in ["GREEN_", "RED_"]
                            for s in ["SCI_WAVE1", "SCI_WAVE2", "SCI_WAVE3", "SKY_WAVE", "CAL_WAVE"]]

        # Iterate over extensions (orderlets) and orders to check for monotonicity in each combination.
        for ext in extensions:

            if debug:
                print("ext = ",ext)
            
            extname = ext
            # try:
            #     naxis1 = L1.header[ext]["NAXIS1"]
            #     naxis2 = L1.header[ext]["NAXIS2"]
            # except KeyError:
            #     import pdb; pdb.set_trace()

            # if debug:
            #     print("naxis1,naxis2,extname = ",naxis1,naxis2,extname)

            if ext == extname:  # Check if extension exists (e.g., if RED isn't processed)

                if debug:
                    data_shape = np.shape(L1[ext])
                    print("data_shape = ", data_shape)

                norders = L1[ext].shape[0]
                for o in range(norders):

                    if debug:
                         print("order = ",o)

                    np_obj_ffi = np.array(L1[ext])

                    if debug:
                        print("wls_shape = ", np.shape(np_obj_ffi))

                    WLS = np_obj_ffi[o,:] # wavelength solution of the current order/orderlet

                    isMonotonic = np.all(WLS[:-1] >= WLS[1:]) # this expression determines monotonicity for the orderlet/order
                    if not isMonotonic:
                        QC_pass = False                             # the QC test fails if one order/orderlet is not monotonic
                        bad_orders.append(ext + '(' + str(o)+')') # append the bad order/orderlet to the list
                        if debug:
                            print('L1[' + ext + ']['+ str(o) +']: monotonic = ' + str(isMonotonic))
                            plt.plot(WLS)
                            plt.title('L1[' + ext + '] (order = '+ str(o) +') -- not monotonic')
                            plt.show()
        if debug:
            try:  # using a try/except statement because sometimes OFNAME isn't defined
                print("File: " + L1['PRIMARY'].header['OFNAME'])
            except:
                pass

        return QC_pass #, bad_orders

    def data_L1_red_green_check(self,debug=False):
        """
        This Quality Control function checks if the red and green data
        are present in an L1 file, and that all array sizes are as expected.

        Args:
             debug - an optional flag.  If True prints shapes of arrays.

        Returns:
             QC_pass - a boolean signifying that all of the data exists as expected
        """
    
        L1 = self.kpf_object

        if debug:
            print(L1.info())
            type_L1 = type(L1)
            print("type_L1 = ",type_L1)
            print("L1 = ",L1)

        QC_pass = True
    
        extensions = L1.extensions
    
        GREEN_extensions = [
         'GREEN_SCI_FLUX1',  
         'GREEN_SCI_FLUX2',  
         'GREEN_SCI_FLUX3',  
         'GREEN_SKY_FLUX',   
         'GREEN_CAL_FLUX',   
         'GREEN_SCI_VAR1',  
         'GREEN_SCI_VAR2',  
         'GREEN_SCI_VAR3',  
         'GREEN_SKY_VAR',   
         'GREEN_CAL_VAR',   
         'GREEN_SCI_WAVE1',  
         'GREEN_SCI_WAVE2',  
         'GREEN_SCI_WAVE3',  
         'GREEN_SKY_WAVE',   
         'GREEN_CAL_WAVE' 
        ] 

        RED_extensions = [
         'RED_SCI_FLUX1',  
         'RED_SCI_FLUX2',  
         'RED_SCI_FLUX3',  
         'RED_SKY_FLUX',   
         'RED_CAL_FLUX',   
         'RED_SCI_VAR1',  
         'RED_SCI_VAR2',  
         'RED_SCI_VAR3',  
         'RED_SKY_VAR',   
         'RED_CAL_VAR',   
         'RED_SCI_WAVE1',  
         'RED_SCI_WAVE2',  
         'RED_SCI_WAVE3',  
         'RED_SKY_WAVE',   
         'RED_CAL_WAVE'  
        ] 
    
        QC_pass = True
    
        for ext in GREEN_extensions:
            if ext not in extensions:
                QC_pass = False
                if debug:
                    print('The extension ' + ext + ' is missing from the file.')
            else:
                if np.shape(L1[ext]) != (35, 4080):
                    QC_pass = False
                    if debug:
                        print('Shape of ' + ext + ' array is incorrect.')
                        print("data_shape =", np.shape(L1[ext]))
                    
        for ext in RED_extensions:
            if ext not in extensions:
                QC_pass = False
                if debug:
                    print('The extension ' + ext + ' is missing from the file.')
            else:
                if np.shape(L1[ext]) != (32, 4080):
                    QC_pass = False
                    if debug:
                        print('Shape of ' + ext + ' array is incorrect.')   
                        print("data_shape =", np.shape(L1[ext]))
        
        return QC_pass

    def data_L1_CaHK_check(self,debug=False):
        """
        This Quality Control function checks if the green and red data
        are present in an L1 file, and that all array sizes are as expected.

        Args:
             debug - an optional flag.  If True, prints shapes of arrays.

        Returns:
             QC_pass - a boolean signifying that all of the data exists as expected
        """
    
        L1 = self.kpf_object

        if debug:
            print(L1.info())
            type_L1 = type(L1)
            print("type_L1 = ",type_L1)
            print("L1 = ",L1)

        QC_pass = True
    
        extensions = L1.extensions
    
        CaHK_extensions = [
         'CA_HK_SCI',  
         'CA_HK_SKY',  
         'CA_HK_SCI_WAVE',  
         'CA_HK_SKY_WAVE'  
        ] 
    
        QC_pass = True
    
        for ext in CaHK_extensions:
            if ext not in extensions:
                QC_pass = False
                if debug:
                    print('The extension ' + ext + ' is missing from the file.')
            else:
                if np.shape(L1[ext]) == (0,):
                    QC_pass = False
                    if debug:
                        print('Shape of ' + ext + ' array is zero.')
                        print("data_shape =", np.shape(L1[ext]))
        
        return QC_pass


#####################################################################

class QCL2(QC):

    """
    Description:
        This class inherits QC superclass and defines QC functions for L2 files.

    Class Attributes:
        kpf_object (astropy.io object): Returned from function KPF0.from_fits(fits_filename,data_type),
            which is wrapped by function read_fits in this module.
        qcdefinitions (QCDefinitions object): Returned from constructor of QCDefinitions class.

    """

    # Call superclass.
    def __init__(self,kpf_object):
        super().__init__(kpf_object)

    def data_L2_check(self,debug=False):
        """
        This Quality Control function checks if all of the 
        expected data (telemetry, CCFs, and RVs) are present.

        Args:
             debug - an optional flag.  If True, prints shapes of arrays.

        Returns:
             QC_pass - a boolean signifying that all of the data exists as expected
        """
    
        L2 = self.kpf_object

        if debug:
            print(L2.info())
            type_L2 = type(L2)
            print("type_L2 = ",type_L2)
            print("L2 = ",L2)
    
        extensions = L2.extensions
    
        required_extensions = [
            "TELEMETRY",
            "GREEN_CCF",
            "RED_CCF",
            "GREEN_CCF_RW",
            "RED_CCF_RW",
            "RV"
        ]
    
        QC_pass = True
    
        if "TELEMETRY" not in extensions:
            QC_pass = False
            if debug:
                print('The extension TELEMETRY is missing from the file.')
        else:
            if np.shape(L2["TELEMETRY"]) == (0,):
                QC_pass = False
                if debug:
                    print('Shape of TELEMETRY array is zero.')
                    print("data_shape =", np.shape(L2["TELEMETRY"]))
                
        if "GREEN_CCF" not in extensions:
            QC_pass = False
            if debug:
                print('The extension GREEN_CCF is missing from the file.')
        else:
            if np.shape(L2["GREEN_CCF"]) != (5, 35, 804):
                QC_pass = False
                if debug:
                    print('Shape of GREEN_CCF array is incorrect.')
                    print("data_shape =", np.shape(L2["GREEN_CCF"]))
                
        if "GREEN_CCF_RW" not in extensions:
            QC_pass = False
            if debug:
                print('The extension GREEN_CCF_RW is missing from the file.')
        else:
            if np.shape(L2["GREEN_CCF_RW"]) != (5, 35, 804):
                QC_pass = False
                if debug:
                    print('Shape of GREEN_CCF_RW array is incorrect.')
                    print("data_shape =", np.shape(L2["GREEN_CCF_RW"]))
                
        if "RED_CCF" not in extensions:
            QC_pass = False
            if debug:
                print('The extension RED_CCF is missing from the file.')
        else:
            if np.shape(L2["RED_CCF"]) != (5, 32, 804):
                QC_pass = False
                if debug:
                    print('Shape of RED_CCF_RW array is incorrect.')
                    print("data_shape =", np.shape(L2["RED_CCF"]))
                
        if "RED_CCF_RW" not in extensions:
            QC_pass = False
            if debug:
                print('The extension RED_CCF_RW is missing from the file.')
        else:
            if np.shape(L2["RED_CCF_RW"]) != (5, 32, 804):
                QC_pass = False
                if debug:
                    print('Shape of RED_CCF_RW array is incorrect.')
                    print("data_shape =", np.shape(L2["RED_CCF_RW"]))
                
        if "RV" not in extensions:
            QC_pass = False
            if debug:
                print('The extension RV is missing from the file.')
        else:
            if np.shape(L2["RV"]) == (0,):
                QC_pass = False
                if debug:
                    print('Shape of RV array is zero.')
                    print("data_shape =", np.shape(L2["RV"]))
        
        return QC_pass

    def L2_datetime_checks(self, debug=False):
        """
        This QC module performs the following checks on datetimes in the header 
        to the RV extension in an L2 object. The timing checks have precision 
        thresholds to only catch significant timing errors and not trigger on 
        small differences related to machine precision or dead time in the 
        Exposure Meter detector.  This method returns True only if all checks 
        pass.
        
            Time ordering: 
                DATE-BEG < DATE-MID < DATE-END
            Duration consistency: 
                DATE-END - DATE-BEG = ELAPSED
            Consistency between Green/Red and overall timing:
                DATE-BEG = GRDATE-B
                DATE-BEG = RDDATE-B
                DATE-END = GRDATE-E
                DATE-END = RDDATE-E
        
        To-do: 
          * Add checks for the times in the RV table.  These are currently in BJD.
            We will also need to record the UT times.
        """
    
        L2 = self.kpf_object
        date_format = "%Y-%m-%dT%H:%M:%S.%f"
        QC_pass = True
    
        time_precision_threshold     = 0.1 # sec - threshold for DATE-BEG, etc.
        time_precision_threshold_exp = 1.0 # sec - threshold for times involving the exposure meter -- account for EM dead time and only catch bad errors
        
        # First check that the appropriate headers and keywords are present
        if not 'PRIMARY' in L2.header:
            QC_pass = False
            return QC_pass
        if not 'RV' in L2.header:
            QC_pass = False
            return QC_pass
        essential_keywords = ['DATE-BEG', 'DATE-MID', 'DATE-END', 'ELAPSED']
        for keyword in essential_keywords:
            if keyword not in L2.header['PRIMARY']:
                if debug:
                    print(f'Missing keyword: {keyword}')
                QC_pass = False
        if not QC_pass:
            return QC_pass
        
        # Check that dates are ordered correctly
        date_beg = datetime.strptime(L2.header['PRIMARY']['DATE-BEG'], date_format)
        date_mid = datetime.strptime(L2.header['PRIMARY']['DATE-MID'], date_format)
        date_end = datetime.strptime(L2.header['PRIMARY']['DATE-END'], date_format)
        elapsed  = float(L2.header['PRIMARY']['ELAPSED'])
        if (date_end < date_mid) or (date_mid < date_beg):
            QC_pass = False
        
        # Check that DATE-BEG + ELAPSE = DATE-END
        if abs((date_end - date_beg).total_seconds() - elapsed) > time_precision_threshold:
            if debug:
                print(f'(DATE-END - DATE-BEG) - ELASPED = {abs((date_end - date_beg).total_seconds() - elapsed)} sec > {time_precision_threshold} sec')
            QC_pass = False
            
        return QC_pass    
