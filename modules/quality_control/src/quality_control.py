import os
import re
import yaml
import numpy as np
import numpy.ma as ma
import pandas as pd
from datetime import datetime
from scipy.ndimage import convolve1d
from kpfpipe.models.level1 import KPF1
from modules.Utils.utils import DummyLogger, styled_text
from modules.Utils.kpf_parse import HeaderParse, get_datetime_obsid, get_kpf_level, get_data_products_expected, get_ObsID
from modules.Utils.kpf_parse import get_data_products_L0, get_data_products_2D, get_data_products_L1, get_data_products_L2
from modules.quicklook.src.analyze_guider import AnalyzeGuider
from modules.quicklook.src.analyze_2d import Analyze2D
from modules.quicklook.src.analyze_l1 import AnalyzeL1
from modules.quicklook.src.analyze_l2 import AnalyzeL2
from modules.calibration_lookup.src.alg import GetCalibrations

DEFAULT_CALIBRATION_CFG_PATH = os.path.join(os.path.dirname(__file__), '../../calibration_lookup/configs/default.cfg')
DEFAULT_CALIBRATION_CFG_PATH = os.path.normpath(DEFAULT_CALIBRATION_CFG_PATH)

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

def compute_clip_corr(n_sigma):

    """
    Compute a correction factor to properly reinflate the variance after it is
    naturally diminished via data-clipping.  Employ a simple Monte Carlo method
    and standard normal deviates to simulate the data-clipping and obtain the
    correction factor.
    """

    var_trials = []
    for x in range(0,10):
        a = np.random.normal(0.0, 1.0, 1000000)
        med = np.median(a, axis=0)
        p16 = np.percentile(a, 16, axis=0)
        p84 = np.percentile(a, 84, axis=0)
        sigma = 0.5 * (p84 - p16)
        mdmsg = med - n_sigma * sigma
        b = np.less(a,mdmsg)
        mdpsg = med + n_sigma * sigma
        c = np.greater(a,mdpsg)
        mask = np.any([b,c],axis=0)
        mx = ma.masked_array(a, mask)
        var = ma.getdata(mx.var(axis=0))
        var_trials.append(var)

    np_var_trials = np.array(var_trials)
    avg_var_trials = np.mean(np_var_trials)
    std_var_trials = np.std(np_var_trials)
    corr_fact = 1.0 / avg_var_trials

    return corr_fact

def avg_data_with_clipping(data_array,n_sigma = 3.0):

    """
    Statistics with outlier rejection (n-sigma data-trimming), ignoring NaNs, across all data array dimensions.
    """

    cf = compute_clip_corr(n_sigma)
    sqrtcf = np.sqrt(cf)

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
    std = ma.getdata(mx.std()) * sqrtcf
    cnt = ma.getdata(mx.count())

    return avg,std,cnt


def check_all_qc_keywords(kpf_object,fname,input_master_type='all',logger=None):

    """
    Method to check all QC keywords in PRIMARY header of FITS object.

    Agnostic of data level; checks all QC keywords in PRIMARY header
    that have assigned value for qc_definitions.fits_keyword_fail_value[dict_key]
    (which are not None).  Failure is declared only for the relevant master type.
    Currently only integer fail_values are handled.

    Returns:
        qc_fail - a boolean signifying that the QC failed (True) for at least one of the QC keywords or not (False).
    """

    logger = logger if logger is not None else DummyLogger()

    qc_fail = False

    qc_definitions = QCDefinitions()

    dict_keys_list = qc_definitions.fits_keywords.keys()

    for dict_key in dict_keys_list:

        kw = qc_definitions.fits_keywords[dict_key]
        master_types = qc_definitions.master_types[dict_key]

        try:
            fail_value = qc_definitions.fits_keyword_fail_value[dict_key]
        except:
            continue

        if fail_value is None:
            continue

        try:
            kw_value = kpf_object.header['PRIMARY'][kw]
            if kw_value == fail_value:
                logger.debug('--------->quality_control: check_all_qc_keywords: fname,kw,kw_value,fail_value = {},{},{},{}'.format(fname,kw,kw_value,fail_value))
                for master_type in master_types:
                    if input_master_type.lower() == master_type.lower() or master_type.lower() == 'all' or input_master_type.lower() == 'all':
                        qc_fail = True
                        break

                return qc_fail

        except KeyError as err:
            continue

    return qc_fail


def execute_all_QCs(kpf_object, data_level, logger=None):
    """
    Method to loop over all QC tests for the data level of the input KPF object
    (an L0, 2D, L1, or L2 object).  This method is useful for testing (e.g.,
    in a Jupyter Notebook).  To run the QCs in a recipe, use methods in
    quality_control_framework.py

    Args:
        kpf_object - a KPF object (L0, 2D, L1, or L2)
        data_type -

    Attributes:
        None

    Returns:
        kpf_object - the input kpf_object with QC keywords added
    """

    logger = logger if logger is not None else DummyLogger()

    #data_level = get_kpf_level(kpf_object)

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

        # Run the QC tests and add result keyword to header
        primary_header = HeaderParse(kpf_object, 'PRIMARY')
        is_good = 1
        this_spectrum_type = primary_header.get_name(use_star_names=False)
        logger.info(f'Spectrum type: {this_spectrum_type}')
        for qc_name in qc_names:
            try:
                spectrum_types = qc_obj.qcdefinitions.spectrum_types[qc_name]
                if (this_spectrum_type in spectrum_types) or ('all' in spectrum_types):
                    if len(qc_obj.qcdefinitions.required_data_products[qc_name]) == 0:
                        all_required_data_products_present = True
                    else:
                        data_products_expected = get_data_products_expected(kpf_object, data_level)
                        data_products_required = qc_obj.qcdefinitions.required_data_products[qc_name]
                        all_required_data_products_present = all(element in data_products_expected for element in data_products_required)
                    if all_required_data_products_present:
                        text_running_qc = styled_text('QC', style="Bold", color="Magenta")
                        text_qc_name = styled_text(qc_name, style="Bold", color="Blue")
                        text_qc_keyword = styled_text(qc_obj.qcdefinitions.fits_keywords[qc_name], style="Bold", color="Blue")
                        logger.info(f'{text_running_qc}: {text_qc_name} ({text_qc_keyword}; {qc_obj.qcdefinitions.descriptions[qc_name]})')
                        method = getattr(qc_obj, qc_name) # get method with the name 'qc_name'
                        qc_value = method() # evaluate method
                        if qc_value == True:
                            text_qc_value = styled_text(qc_value, style="Bold", color="Green")
                        elif qc_value == False:
                            text_qc_value = styled_text(qc_value, style="Bold", color="Red")
                            is_good = 0
                        if qc_obj.qcdefinitions.fits_keywords[qc_name] == 'KPFERA':
                            logger.info(f'Result: {styled_text("KPFERA", style="Bold", color="Blue")}={styled_text(qc_value, style="Bold")}')
                        else:
                            logger.info(f'QC result: {text_qc_value} (True = pass)')
                        qc_obj.add_qc_keyword_to_header(qc_name, qc_value)
                    else:
                        logger.info(f'Not running QC: {qc_name} ({qc_obj.qcdefinitions.descriptions[qc_name]}) because {data_products_required} not in list of expected data products({data_products_expected})')
                else:
                    logger.info(f'Not running QC: {qc_name} ({qc_obj.qcdefinitions.descriptions[qc_name]}) because {this_spectrum_type} not in list of spectrum types: {spectrum_types}')

            except KeyError as e:
                logger.info(f"KeyError: {e}")
                pass

            except AttributeError as e:
                logger.info(f'Method {qc_name} does not exist in qc_obj or another AttributeError occurred: {e}')
                pass

            except Exception as e:
                logger.info(f'An error occurred when executing {qc_name}:', str(e))
                pass

        kpf_object.header['PRIMARY']['ISGOOD'] = (is_good, "QC: all other QC tests passed")

    return kpf_object


def QC_report(kpf_object, return_keywords=True, print_output=False, yaml_path=None, logger=None):
    """
    Method to determine if all QC tests have been run on the input kpf_object
    by examining its keywords. The method determines the data_level for
    kpf_object and checks for keywords of that level and lower, e.g., for
    data_level = 'L1', the method checks for keywords in levels 'L0', '2D',
    and 'L1'.

    Args:
        kpf_object - a KPF object (L0, 2D, L1, or L2)
        return_keywords (boolean) - if true, keywords are returned (e.g., 'OLDBIAS') instead of method names (e.g., 'D2_master_bias_age')
        print_output (boolean) - if true, print the output instead of returning anything
        yaml_path (str or None) - if not None, output the QC results to the specified YAML file path
        logger - Python logger object; if None, the DummyLogger is used

    Returns:
        tuple of (qc_names_missing, qc_names_present, qc_names_present_pass, qc_names_present_fail), where
            qc_names_missing = names of QC tests with keywords missing from kpf_object that should be
            qc_names_present = names of QC tests with keywords in kpf_object that should be
            qc_names_present_pass = names of QC tests with keywords in kpf_object that should be and QC = True (passed)
            qc_names_present_fail = names of QC tests with keywords in kpf_object that should be and QC = False (failed)
    """

    def unique_preserve_order(mylist):
        from itertools import chain
        flattened = list(chain.from_iterable(mylist))
        seen = set()
        return [x for x in flattened if not (x in seen or seen.add(x))]

    def get_appropriate_qcs(data_level, spectrum_type):
        '''
        Get a list of QC method names appropriate for the data level
        
        Args:
            data_level - 'L0', '2D', 'L1', or 'L2'
            spectrum_type - types of spectra that a QC is applied to
                One of: 'all', 'Bias', 'Dark', 'Flat', 'Wide Flat', 'LFC', 'Etalon', 'ThAr', 'UNe', 'Sun', 'Star', <starname>

        Returns:
            qc_names - list of QC method names to be run on level object
        '''
        if data_level == 'L0':
            qc_obj = QCL0(kpf_object)
            data_products_present = get_data_products_L0(kpf_object)
        elif data_level == '2D':
            qc_obj = QC2D(kpf_object)
            data_products_present = get_data_products_2D(kpf_object)
        elif data_level == 'L1':
            qc_obj = QCL1(kpf_object)
            data_products_present = get_data_products_L1(kpf_object)
        elif data_level == 'L2':
            qc_obj = QCL2(kpf_object)
            data_products_present = get_data_products_L2(kpf_object)

        qc_names = []
        for qc_name in qc_obj.qcdefinitions.names:
            if data_level in qc_obj.qcdefinitions.kpf_data_levels[qc_name]:
                required_data_products_present = all(elem in data_products_present for elem in qc_obj.qcdefinitions.required_data_products[qc_name])
                if required_data_products_present:
                    if ('all' in qc_obj.qcdefinitions.spectrum_types[qc_name]) or (spectrum_type in qc_obj.qcdefinitions.spectrum_types[qc_name]):
                        qc_names.append(qc_name)
        
        return qc_names

    logger = logger if logger is not None else DummyLogger()
    this_data_level = get_kpf_level(kpf_object)
    primary_header = HeaderParse(kpf_object, 'PRIMARY')
    this_spectrum_type = primary_header.get_name(use_star_names=False)

    try:
        ObsID = kpf_object.header['PRIMARY']['OFNAME']
    except:
        ObsID = 'ObsID not available'

    data_levels_map = {'L0': ['L0'], '2D': ['L0', '2D'], 'L1': ['L0', '2D', 'L1'], 'L2': ['L0', '2D', 'L1', 'L2']}
    data_levels = data_levels_map.get(this_data_level, [])

    expected_qc_names = unique_preserve_order([get_appropriate_qcs(dl, this_spectrum_type) for dl in data_levels])

    if 'add_kpfera' in expected_qc_names:
        expected_qc_names.remove('add_kpfera')

    qc_names_missing, qc_names_present, qc_names_present_pass, qc_names_present_fail = [], [], [], []
    qcd = QCDefinitions()

    report_rows = []

    for qc_name in expected_qc_names:
        kwd = qcd.fits_keywords[qc_name]
        lvl = qcd.kpf_data_levels[qc_name][0]
        desc = qcd.descriptions[qc_name]
        present = kwd in kpf_object.header['PRIMARY']
        passed = present and bool(kpf_object.header['PRIMARY'][kwd])
        master_types = qcd.master_types[qc_name]
        drift_types = qcd.drift_types[qc_name]

        name_entry = kwd if return_keywords else qc_name

        if present:
            qc_names_present.append(name_entry)
            if passed:
                qc_names_present_pass.append(name_entry)
            else:
                qc_names_present_fail.append(name_entry)
        else:
            qc_names_missing.append(name_entry)

        report_rows.append({
            "Keyword": kwd,
            "Level": lvl,
            "Description": desc,
            "Present": present,
            "Pass": passed,
            "master_types": master_types,
            "drift_types": drift_types,
        })

    if yaml_path is not None:
        with open(yaml_path, 'w') as yaml_file:
            yaml.dump({"QC_Report": report_rows}, yaml_file, sort_keys=False)
        if logger:
            logger.info(f"YAML QC report saved to {yaml_path}")

    if print_output:
        # All QCs
        print(f'{styled_text(f"Quality Control Report for {ObsID} ({this_spectrum_type})", style="Bold", color="Black")}')
        print()
        print(f'{styled_text(f"All QC Keywords:", style="Bold", color="Black")}')
        print(f'{styled_text("    Keyword      Level  QC Test Description", style="Bold", color="Black")}')  
        for row in report_rows:
            checkmark = '✓' if row["Present"] else '✗'
            col = 'Green' if row["Pass"] else 'Red' if row["Present"] else 'Black'
            print(f'    {styled_text(row["Keyword"], style="Bold", color=col):<9} {checkmark} {" " * (8 - len(row["Keyword"]))}  {row["Level"]:<6} {row["Description"]}')
        print()
        print(f"    {styled_text('Pass', style='Bold', color='Green')}/{styled_text('Fail', style='Bold', color='Red')}, ✓ - keyword present, ✗ - keyword missing")
        print()

        # QCs for Masters
        if this_spectrum_type in ['Bias', 'Dark', 'Flat', 'Wide Flat', 'LFC', 'Etalon', 'ThAr', 'UNe']:
            print(f'{styled_text(f"QC Keywords for observation to be included in master {this_spectrum_type} stack:", style="Bold", color="Black")}')
            if not all(row['master_types'] == [] for row in report_rows):
                print(f'{styled_text("    Keyword      Level  QC Test Description", style="Bold", color="Black")}')  
                for row in report_rows:
                    if any(x in row['master_types'] for x in [this_spectrum_type, 'all']):
                        checkmark = '✓' if row["Present"] else '✗'
                        col = 'Green' if row["Pass"] else 'Red' if row["Present"] else 'Black'
                        print(f'    {styled_text(row["Keyword"], style="Bold", color=col):<9} {checkmark} {" " * (8 - len(row["Keyword"]))}  {row["Level"]:<6} {row["Description"]}')
                print()
                print(f"    {styled_text('Pass', style='Bold', color='Green')}/{styled_text('Fail', style='Bold', color='Red')}, ✓ - keyword present, ✗ - keyword missing")
            else:
                print(f'{styled_text("    None", style="Bold", color="Black")}')  
            print()

        # QCs for Drift Measurements
        if this_spectrum_type in ['Etalon', 'LFC', 'ThAr', 'UNe']:
            print(f'{styled_text(f"QC Keywords for observation to be included in {this_spectrum_type} drift measurements:", style="Bold", color="Black")}')
            if not all(row['drift_types'] == [] for row in report_rows):
                print(f'{styled_text("    Keyword      Level  QC Test Description", style="Bold", color="Black")}')  
                for row in report_rows:
                    if any(x in row['drift_types'] for x in [this_spectrum_type, 'all']):
                        checkmark = '✓' if row["Present"] else '✗'
                        col = 'Green' if row["Pass"] else 'Red' if row["Present"] else 'Black'
                        print(f'    {styled_text(row["Keyword"], style="Bold", color=col):<9} {checkmark} {" " * (8 - len(row["Keyword"]))}  {row["Level"]:<6} {row["Description"]}')
                print()
                print(f"    {styled_text('Pass', style='Bold', color='Green')}/{styled_text('Fail', style='Bold', color='Red')}, ✓ - keyword present, ✗ - keyword missing")
            else:
                print(f'{styled_text("    None", style="Bold", color="Black")}')  

    if not (yaml_path or print_output):
        return (qc_names_missing, qc_names_present, qc_names_present_pass, qc_names_present_fail)


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
        spectrum_types (dictionary of arrays of strings): Each entry specifies the types of spectra that the metric will be applied to.
            Possible strings in array: 'all', 'Bias', 'Dark', 'Flat', 'Wide Flat', 'LFC', 'Etalon', 'ThAr', 'UNe', 'Sun', 'Star', <starname>
        master_types (dictionary of arrays of strings): Each entry specifies the types of masters where the QC check is relevant.  If the QC fails for an exposure, it is not added to the master stack.
            Possible strings in array: 'Bias', 'Dark', 'Flat', 'Wide Flat', 'LFC', 'Etalon', 'ThAr', 'UNe'
        drift_types (dictionary of arrays of strings): Each entry specifies the types of observations where the QC check is relevant.  If the QC fails for an exposure, it is not used in wavelength drift measurements.
            Possible strings in array: 'all', 'LFC', 'Etalon', 'ThAr', 'UNe'
        required_data_products (dictionary of arrays of strings): specifies if data products are needed to perform check
            if = [], then no required data products; other possible values are from get_data_products_L0, etc.
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
        self.data_types = {}
        self.spectrum_types = {}
        self.master_types = {} # if = [], then the QC test is not relevant for the construction of any masters
        self.drift_types = {} # if = [], then the QC test is not relevant for drift measurements with any calibration type
        self.required_data_products = {} # if = [], then no required data products; other possible values: Green, Red, CaHK, ExpMeter, Guider, Telemetry, Config, Receipt, Pyrheliometer
        self.fits_keywords = {}
        self.fits_comments = {}
        self.db_columns = {}
        self.fits_keyword_fail_value = {}

        # Define QC metrics
        name1 = 'not_junk'
        self.names.append(name1)
        self.descriptions[name1] = 'File is not in list of junk files'
        self.kpf_data_levels[name1] = ['L0', '2D', 'L1', 'L2']
        self.data_types[name1] = 'int'
        self.spectrum_types[name1] = ['all', ] # Need trailing comma to make list hashable
        self.master_types[name1] = ['all', ]
        self.drift_types[name1] = ['all', ]
        self.required_data_products[name1] = [] # no required data products
        self.fits_keywords[name1] = 'NOTJUNK'
        self.fits_comments[name1] = 'QC: Not in list of junk files'
        self.db_columns[name1] = None
        self.fits_keyword_fail_value[name1] = 0

        name2 = 'monotonic_wavelength_solution'
        self.names.append(name2)
        self.descriptions[name2] = 'Wavelength solution is monotonic'
        self.kpf_data_levels[name2] = ['L1']
        self.data_types[name2] = 'int'
        self.spectrum_types[name2] = ['all', ]
        self.master_types[name2] = []
        self.drift_types[name2] = []
        self.required_data_products[name2] = [] # no required data products
        self.fits_keywords[name2] = 'MONOTWLS'
        self.fits_comments[name2] = 'QC: Monotonic wavelength solution'
        self.db_columns[name2] = None
        self.fits_keyword_fail_value[name2] = 0

        name3 = 'L0_data_products'
        self.names.append(name3)
        self.kpf_data_levels[name3] = ['L0']
        self.descriptions[name3] = 'Expected L0 data products present with non-zero array sizes'
        self.data_types[name3] = 'int'
        self.spectrum_types[name3] = ['all', ]
        self.master_types[name3] = ['all', ]
        self.drift_types[name3] = ['all',]
        self.required_data_products[name3] = [] # no required data products
        self.fits_keywords[name3] = 'DATAPRL0'
        self.fits_comments[name3] = 'QC: L0 data present'
        self.db_columns[name3] = None
        self.fits_keyword_fail_value[name3] = 0

        name4 = 'L0_header_keywords_present'
        self.names.append(name4)
        self.kpf_data_levels[name4] = ['L0']
        self.descriptions[name4] = 'Expected L0 header keywords present'
        self.data_types[name4] = 'int'
        self.spectrum_types[name4] = ['all', ]
        self.master_types[name4] = ['all', ]
        self.drift_types[name4] = ['all',]
        self.required_data_products[name4] = [] # no required data products
        self.fits_keywords[name4] = 'KWRDPRL0'
        self.fits_comments[name4] = 'QC: L0 keywords present'
        self.db_columns[name4] = None
        self.fits_keyword_fail_value[name4] = 0

        name5 = 'L0_datetime'
        self.names.append(name5)
        self.kpf_data_levels[name5] = ['L0']
        self.descriptions[name5] = 'Timing consistency in L0 header keywords and ExpMeter table'
        self.data_types[name5] = 'int'
        self.spectrum_types[name5] = ['all', ]
        self.master_types[name5] = ['all', ]
        self.drift_types[name5] = []
        self.required_data_products[name5] = [] # no required data products
        self.fits_keywords[name5] = 'TIMCHKL0'
        self.fits_comments[name5] = 'QC: L0 times consistent'
        self.db_columns[name5] = None
        self.fits_keyword_fail_value[name5] = 0

        name5b = 'L2_datetime'
        self.names.append(name5b)
        self.kpf_data_levels[name5b] = ['L2']
        self.descriptions[name5b] = 'Timing consistency in L2 files'
        self.data_types[name5b] = 'int'
        self.spectrum_types[name5b] = ['all', ]
        self.master_types[name5b] = []
        self.drift_types[name5b] = []
        self.required_data_products[name5b] = [] # no required data products
        self.fits_keywords[name5b] = 'TIMCHKL2'
        self.fits_comments[name5b] = 'QC: L2 times consistent'
        self.db_columns[name5b] = None
        self.fits_keyword_fail_value[name5b] = 0

        name6 = 'EM_not_saturated'
        self.names.append(name6)
        self.kpf_data_levels[name6] = ['L0']
        self.descriptions[name6] = '2+ reduced EM pixels within 90% of saturation in EM-SCI or EM-SKY'
        self.data_types[name6] = 'int'
        self.spectrum_types[name6] = ['all', ]
        self.master_types[name6] = []
        self.drift_types[name6] = []
        self.required_data_products[name6] = ['ExpMeter']
        self.fits_keywords[name6] = 'EMSAT'
        self.fits_comments[name6] = 'QC: EM not saturated'
        self.db_columns[name6] = None
        self.fits_keyword_fail_value[name6] = 0

        name7 = 'EM_flux_not_negative'
        self.names.append(name7)
        self.kpf_data_levels[name7] = ['L0']
        self.descriptions[name7] = 'Negative flux in the EM-SCI and EM-SKY by looking for 20 consecuitive pixels in the summed spectra with negative flux.'
        self.data_types[name7] = 'int'
        self.spectrum_types[name7] = ['all', ]
        self.master_types[name7] = []
        self.drift_types[name7] = []
        self.required_data_products[name7] = ['ExpMeter']
        self.fits_keywords[name7] = 'EMNEG'
        self.fits_comments[name7] = 'QC: EM not negative flux'
        self.db_columns[name7] = None
        self.fits_keyword_fail_value[name7] = 0

        name8 = 'D2_lfc_flux'
        self.names.append(name8)
        self.kpf_data_levels[name8] = ['2D']
        self.descriptions[name8] = 'LFC frame that goes into a master has sufficient flux'
        self.data_types[name8] = 'int'
        self.spectrum_types[name8] = ['LFC', ]
        self.master_types[name8] = ['LFC', ]
        self.drift_types[name8] = ['LFC', ]
        self.required_data_products[name8] = [] # no required data products
        self.fits_keywords[name8] = 'LFC2DFOK'
        self.fits_comments[name8] = 'QC: LFC flux meets threshold of 4000 counts'
        self.db_columns[name8] = None
        self.fits_keyword_fail_value[name8] = 0

        name9 = 'data_2D_bias_low_flux'
        self.names.append(name9)
        self.kpf_data_levels[name9] = ['2D']
        self.descriptions[name9] = 'Flux is low in bias exposure'
        self.data_types[name9] = 'int'
        self.spectrum_types[name9] = ['Bias', ]
        self.master_types[name9] = ['Bias', ]
        self.drift_types[name9] = []
        self.required_data_products[name9] = [] # no required data products
        self.fits_keywords[name9] = 'LOWBIAS'
        self.fits_comments[name9] = 'QC: 2D bias low flux check'
        self.db_columns[name9] = None
        self.fits_keyword_fail_value[name9] = 0

        name10 = 'data_2D_dark_low_flux'
        self.names.append(name10)
        self.kpf_data_levels[name10] = ['2D']
        self.descriptions[name10] = 'Flux is low in dark exposure'
        self.data_types[name10] = 'int'
        self.spectrum_types[name10] = ['Dark', ]
        self.master_types[name10] = ['Dark', ]
        self.drift_types[name10] = []
        self.required_data_products[name10] = [] # no required data products
        self.fits_keywords[name10] = 'LOWDARK'
        self.fits_comments[name10] = 'QC: 2D dark low flux check'
        self.db_columns[name10] = None
        self.fits_keyword_fail_value[name10] = 0

        name11 = 'data_L1_red_green'
        self.names.append(name11)
        self.kpf_data_levels[name11] = ['L1']
        self.data_types[name11] = 'int'
        self.spectrum_types[name11] = ['all', ]
        self.master_types[name11] = []
        self.drift_types[name11] = ['all', ]
        self.required_data_products[name11] = [] # no required data products
        self.descriptions[name11] = 'Green and Red data present in L1 with expected shapes'
        self.fits_keywords[name11] = 'DATAPRL1'
        self.fits_comments[name11] = 'QC: L1 red and green data present check'
        self.db_columns[name11] = None
        self.fits_keyword_fail_value[name11] = 0

        name12 = 'data_L1_CaHK'
        self.names.append(name12)
        self.kpf_data_levels[name12] = ['L1']
        self.descriptions[name12] = 'CaHK data present in L1 with expected shape'
        self.data_types[name12] = 'int'
        self.spectrum_types[name12] = ['all', ]
        self.master_types[name12] = []
        self.drift_types[name12] = []
        self.required_data_products[name12] = ['CaHK']
        self.fits_keywords[name12] = 'CAHKPRL1'
        self.fits_comments[name12] = 'QC: L1 CaHK present check'
        self.db_columns[name12] = None
        self.fits_keyword_fail_value[name12] = 0

        name13 = 'data_L2'
        self.names.append(name13)
        self.kpf_data_levels[name13] = ['L2']
        self.descriptions[name13] = 'All data present in L2'
        self.data_types[name13] = 'int'
        self.spectrum_types[name13] = ['all', ]
        self.master_types[name13] = []
        self.drift_types[name13] = []
        self.required_data_products[name13] = [] # no required data products
        self.fits_keywords[name13] = 'DATAPRL2'
        self.fits_comments[name13] = 'QC: L2 data present check'
        self.db_columns[name13] = None
        self.fits_keyword_fail_value[name13] = 0

        name14 = 'data_2D_CaHK'
        self.names.append(name14)
        self.kpf_data_levels[name14] = ['2D']
        self.descriptions[name14] = 'CaHK CCD data present with expected array sizes'
        self.data_types[name14] = 'int'
        self.spectrum_types[name14] = ['all', ]
        self.master_types[name14] = []
        self.drift_types[name14] = []
        self.required_data_products[name14] = ['CaHK']
        self.fits_keywords[name14] = 'CAHKPR2D'
        self.fits_comments[name14] = 'QC: 2D CaHK data present check'
        self.db_columns[name14] = None
        self.fits_keyword_fail_value[name14] = 0

        name15 = 'data_2D_red_green'
        self.names.append(name15)
        self.kpf_data_levels[name15] = ['2D']
        self.descriptions[name15] = 'Green and Red CCD data present with expected array sizes'
        self.data_types[name15] = 'int'
        self.spectrum_types[name15] = ['all', ]
        self.master_types[name15] = ['all', ]
        self.drift_types[name15] = ['all', ]
        self.required_data_products[name15] = [] # no required data products
        self.fits_keywords[name15] = 'DATAPR2D'
        self.fits_comments[name15] = 'QC: 2D red and green data present check'
        self.db_columns[name15] = None
        self.fits_keyword_fail_value[name15] = 0

        name16 = 'positive_2D_SNR'
        self.names.append(name16)
        self.kpf_data_levels[name16] = ['2D']
        self.descriptions[name16] = 'Green and Red CCD data/variance^0.5 not significantly negative'
        self.data_types[name16] = 'int'
        self.spectrum_types[name16] = ['all', ]
        self.master_types[name16] = []
        self.drift_types[name16] = []
        self.required_data_products[name16] = [] # no required data products
        self.fits_keywords[name16] = 'POS2DSNR'
        self.fits_comments[name16] = 'QC: 2D check for > 10% data 5-sigma below zero'
        self.db_columns[name16] = None
        self.fits_keyword_fail_value[name16] = 0

        name17 = 'add_kpfera'
        self.names.append(name17)
        self.kpf_data_levels[name17] = ['L0', '2D', 'L1', 'L2']
        self.descriptions[name17] = 'Not a QC test; KPFERA keyword added to header'
        self.data_types[name17] = 'float'
        self.spectrum_types[name17] = ['all', ]
        self.master_types[name17] = []
        self.drift_types[name17] = []
        self.required_data_products[name17] = [] # no required data products
        self.fits_keywords[name17] = 'KPFERA'
        self.fits_comments[name17] = 'Current era of KPF observations'
        self.db_columns[name17] = None
        self.fits_keyword_fail_value[name17] = -1

        name19 = 'L1_check_snr_lfc'
        self.names.append(name19)
        self.kpf_data_levels[name19] = ['L1']
        self.descriptions[name19] = 'LFC not saturated'
        self.data_types[name19] = 'int'
        self.spectrum_types[name19] = ['LFC', ]
        self.master_types[name19] = ['LFC', ]
        self.drift_types[name19] = ['LFC', ]
        self.required_data_products[name19] = [] # no required data products
        self.fits_keywords[name19] = 'LFCSAT'
        self.fits_comments[name19] = 'QC: LFC not saturated'
        self.db_columns[name19] = None
        self.fits_keyword_fail_value[name19] = 0

        name18 = 'L0_bad_readout_check'
        self.names.append(name18)
        self.kpf_data_levels[name18] = ['L0']
        self.descriptions[name18] = 'CCD read properly (Texp !≈ 6 sec and Texp_desired > 7 sec)'
        self.data_types[name18] = 'int'
        self.spectrum_types[name18] = ['all', ]
        self.master_types[name18] = ['all', ]
        self.drift_types[name18] = ['all', ]
        self.required_data_products[name18] = [] # no required data products
        self.fits_keywords[name18] = 'GOODREAD'
        self.fits_comments[name18] = 'QC: CCD read properly'
        self.db_columns[name18] = None
        self.fits_keyword_fail_value[name18] = 0

        name20 = 'L1_correct_wls_check'
        self.names.append(name20)
        self.kpf_data_levels[name20] = ['L1']
        self.descriptions[name20] = 'WLS files exist, are not the same, and bracket the observation'
        self.data_types[name20] = 'int'
        self.spectrum_types[name20] = ['all', ]
        self.master_types[name20] = []
        self.drift_types[name20] = []
        self.required_data_products[name20] = [] # no required data products
        self.fits_keywords[name20] = 'WLSL1'
        self.fits_comments[name20] = 'QC: WLS files are correct in L1'
        self.db_columns[name20] = None
        self.fits_keyword_fail_value[name20] = 0

        name21 = 'D2_master_bias_age'
        self.names.append(name21)
        self.kpf_data_levels[name21] = ['2D']
        self.descriptions[name21] = 'Master bias from within 5 days of this observation'
        self.data_types[name21] = 'int'
        self.spectrum_types[name21] = ['Dark', 'Flat', 'Wide Flat', 'LFC', 'Etalon', 'ThAr', 'UNe', 'Sun', 'Star']
        self.master_types[name21] = []
        self.drift_types[name21] = []
        self.required_data_products[name21] = [] # no required data products
        self.fits_keywords[name21] = 'OLDBIAS'
        self.fits_comments[name21] = 'QC: Master bias within 5 days of this obs'
        self.db_columns[name21] = None
        self.fits_keyword_fail_value[name21] = 0

        name23 = 'D2_master_dark_age'
        self.names.append(name23)
        self.kpf_data_levels[name23] = ['2D']
        self.descriptions[name23] = 'Master dark from within 5 days of this observation'
        self.data_types[name23] = 'int'
        self.spectrum_types[name23] = ['Bias', 'Flat', 'Wide Flat', 'LFC', 'Etalon', 'ThAr', 'UNe', 'Sun', 'Star']
        self.master_types[name23] = []
        self.drift_types[name23] = []
        self.required_data_products[name23] = [] # no required data products
        self.fits_keywords[name23] = 'OLDDARK'
        self.fits_comments[name23] = 'QC: Master dark within 5 days of this obs'
        self.db_columns[name23] = None
        self.fits_keyword_fail_value[name23] = 0

        name24 = 'D2_master_flat_age'
        self.names.append(name24)
        self.kpf_data_levels[name24] = ['2D']
        self.descriptions[name24] = 'Master flat from within 5 days of this observation'
        self.data_types[name24] = 'int'
        self.spectrum_types[name24] = ['Bias', 'Dark', 'Wide Flat', 'LFC', 'Etalon', 'ThAr', 'UNe', 'Sun', 'Star']
        self.master_types[name24] = []
        self.drift_types[name24] = []
        self.required_data_products[name24] = [] # no required data products
        self.fits_keywords[name24] = 'OLDFLAT'
        self.fits_comments[name24] = 'QC: Master flat within 5 days of this obs'
        self.db_columns[name24] = None
        self.fits_keyword_fail_value[name24] = 0

        name25 = 'L1_WLSFILE_age'
        self.names.append(name25)
        self.kpf_data_levels[name25] = ['L1']
        self.descriptions[name25] = 'WLSFILE from within 2 days of this observation'
        self.data_types[name25] = 'int'
        self.spectrum_types[name25] = ['all', ]
        self.master_types[name25] = []
        self.drift_types[name25] = []
        self.required_data_products[name25] = [] # no required data products
        self.fits_keywords[name25] = 'OLDWLS'
        self.fits_comments[name25] = 'QC: WLSFILE within 2 days of this obs'
        self.db_columns[name25] = None
        self.fits_keyword_fail_value[name25] = 0

        name26 = 'L1_WLSFILE2_age'
        self.names.append(name26)
        self.kpf_data_levels[name26] = ['L1']
        self.descriptions[name26] = 'WLSFILE2 from within 2 days of this observation'
        self.data_types[name26] = 'int'
        self.spectrum_types[name26] = ['all', ]
        self.master_types[name26] = []
        self.drift_types[name26] = []
        self.required_data_products[name26] = [] # no required data products
        self.fits_keywords[name26] = 'OLDWLS2'
        self.fits_comments[name26] = 'QC: WLSFILE2 within 2 days of this obs'
        self.db_columns[name26] = None
        self.fits_keyword_fail_value[name26] = 0

        name27 = 'L1_FLAT_SNR'
        self.names.append(name27)
        self.kpf_data_levels[name27] = ['L1']
        self.descriptions[name27] = 'Flat SNR sufficient, all orders/orderlets'
        self.data_types[name27] = 'int'
        self.spectrum_types[name27] = ['Flat', ]
        self.master_types[name27] = []
        self.drift_types[name27] = []
        self.required_data_products[name27] = [] # no required data products
        self.fits_keywords[name27] = 'FLATSNR'
        self.fits_comments[name27] = 'QC: Flat SNR sufficient, all orders/orderlets'
        self.db_columns[name27] = None
        self.fits_keyword_fail_value[name27] = 0

        name28 = 'L1_LFC_lines'
        self.names.append(name28)
        self.kpf_data_levels[name28] = ['L1']
        self.descriptions[name28] = 'Number and distribution of LFC lines sufficient'
        self.data_types[name28] = 'int'
        self.spectrum_types[name28] = ['LFC', ]
        self.master_types[name28] = []
        self.drift_types[name28] = ['LFC', ]
        self.required_data_products[name28] = [] # no required data products
        self.fits_keywords[name28] = 'LFCLINES'
        self.fits_comments[name28] = 'QC: Number and dist of LFC lines sufficient'
        self.db_columns[name28] = None
        self.fits_keyword_fail_value[name28] = 0

        name29 = 'L1_Etalon_lines'
        self.names.append(name29)
        self.kpf_data_levels[name29] = ['L1']
        self.descriptions[name29] = 'Number and distribution of Etalon lines sufficient'
        self.data_types[name29] = 'int'
        self.spectrum_types[name29] = ['Etalon', ]
        self.master_types[name29] = []
        self.drift_types[name29] = ['Etalon', ]
        self.required_data_products[name29] = [] # no required data products
        self.fits_keywords[name29] = 'ETALINES'
        self.fits_comments[name29] = 'QC: Number and dist of Etalon lines sufficient'
        self.db_columns[name29] = None
        self.fits_keyword_fail_value[name29] = 0

        name30 = 'L1_wild_WLS_SCI'
        self.names.append(name30)
        self.kpf_data_levels[name30] = ['L1']
        self.descriptions[name30] = 'Not wild SCI WLS (stdev < 5 pix in all orders compared to ref)'
        self.data_types[name30] = 'int'
        self.spectrum_types[name30] = ['all', ]
        self.master_types[name30] = []
        self.drift_types[name30] = []
        self.required_data_products[name30] = [] # no required data products
        self.fits_keywords[name30] = 'WILDWSCI'
        self.fits_comments[name30] = 'QC: SCI wavelength solution not wild'
        self.db_columns[name30] = None
        self.fits_keyword_fail_value[name30] = 0

        name31 = 'L1_wild_WLS_SKY'
        self.names.append(name31)
        self.kpf_data_levels[name31] = ['L1']
        self.descriptions[name31] = 'Not wild SKY WLS (stdev < 5 pix in all orders compared to ref)'
        self.data_types[name31] = 'int'
        self.spectrum_types[name31] = ['all', ]
        self.master_types[name31] = []
        self.drift_types[name31] = []
        self.required_data_products[name31] = [] # no required data products
        self.fits_keywords[name31] = 'WILDWSKY'
        self.fits_comments[name31] = 'QC: SKY wavelength solution not wild'
        self.db_columns[name31] = None
        self.fits_keyword_fail_value[name31] = 0

        name32 = 'L1_wild_WLS_CAL'
        self.names.append(name32)
        self.kpf_data_levels[name32] = ['L1']
        self.descriptions[name32] = 'Not wild CAL WLS (stdev < 5 pix in all orders compared to ref)'
        self.data_types[name32] = 'int'
        self.spectrum_types[name32] = ['all', ]
        self.master_types[name32] = []
        self.drift_types[name32] = []
        self.required_data_products[name32] = [] # no required data products
        self.fits_keywords[name32] = 'WILDWCAL'
        self.fits_comments[name32] = 'QC: CAL wavelength solution not wild'
        self.db_columns[name32] = None
        self.fits_keyword_fail_value[name32] = 0

        name33 = 'NTP_timing'
        self.names.append(name33)
        self.kpf_data_levels[name33] = ['L0']
        self.descriptions[name33] = 'NTP time accurate to within 100 ms'
        self.data_types[name33] = 'int'
        self.spectrum_types[name33] = ['all', ]
        self.master_types[name33] = []
        self.drift_types[name33] = []
        self.required_data_products[name33] = [] # no required data products
        self.fits_keywords[name33] = 'NTPGOOD'
        self.fits_comments[name33] = 'QC: NTP time accurate to within 100 ms'
        self.db_columns[name33] = None
        self.fits_keyword_fail_value[name33] = 0

        name34 = 'good_guiding'
        self.names.append(name34)
        self.kpf_data_levels[name34] = ['L0']
        self.descriptions[name34] = 'Guiding meets specs'
        self.data_types[name34] = 'int'
        self.spectrum_types[name34] = ['Star', ]
        self.master_types[name34] = []
        self.drift_types[name34] = []
        self.required_data_products[name34] = [] # no required data products
        self.fits_keywords[name34] = 'GUIDGOOD'
        self.fits_comments[name34] = 'QC: Guider RMS and bias within 50 mas RMS'
        self.db_columns[name34] = None
        self.fits_keyword_fail_value[name34] = 0

        name35 = 'good_TARG_headers'
        self.names.append(name35)
        self.kpf_data_levels[name35] = ['L0']
        self.descriptions[name35] = 'TARG headers have plausible values'
        self.data_types[name35] = 'int'
        self.spectrum_types[name35] = ['Star', ]
        self.master_types[name35] = []
        self.drift_types[name35] = []
        self.required_data_products[name35] = [] # no required data products
        self.fits_keywords[name35] = 'TARGPLAU'
        self.fits_comments[name35] = 'QC: TARG kwds present with plausible values'
        self.db_columns[name35] = None
        self.fits_keyword_fail_value[name35] = 0

        name36 = 'L2_barycentric_rv_percent_change'
        self.names.append(name36)
        self.kpf_data_levels[name36] = ['L2']
        self.descriptions[name36] = 'Check non-zero-weight orders PCBCV values are within an acceptable range.'
        self.data_types[name36] = 'int'
        self.spectrum_types[name36] = ['Star', ]
        self.master_types[name36] = []
        self.required_data_products[name36] = ['Green', 'Red']
        self.fits_keywords[name36] = 'QCPCBCV'
        self.fits_comments[name36] = 'QC: PCBCV values within acceptable range'
        self.db_columns[name36] = None
        self.fits_keyword_fail_value[name36] = 0

#        name36 = 'DRP_version_equal_2D_L1'
#        self.names.append(name36)
#        self.kpf_data_levels[name36] = ['L1']
#        self.descriptions[name36] = 'Same DRP version for 2D and L1'
#        self.data_types[name36] = 'int'
#        self.spectrum_types[name36] = ['all',]
#        self.master_types[name36] = []
#        self.required_data_products[name36] = [] # no required data products
#        self.fits_keywords[name36] = 'DRPS2DL1'
#        self.fits_comments[name36] = 'QC: Same DRP version for 2D and L1'
#        self.db_columns[name36] = None
#        self.fits_keyword_fail_value[name36] = 0
#
#        name37 = 'DRP_version_equal_L1_L2'
#        self.names.append(name37)
#        self.kpf_data_levels[name37] = ['L2']
#        self.descriptions[name37] = 'Same DRP version for L1 and L2'
#        self.data_types[name37] = 'int'
#        self.spectrum_types[name37] = ['all',]
#        self.master_types[name37] = []
#        self.required_data_products[name37] = [] # no required data products
#        self.fits_keywords[name37] = 'DRPS2DL1'
#        self.fits_comments[name37] = 'QC: Same DRP version for L1 and L2'
#        self.db_columns[name37] = None
#        self.fits_keyword_fail_value[name37] = 0

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
            print(styled_text(f"Quality Control tests for {data_level}:", style="Bold"))
            for qc_name in qc_names:

                kpf_data_levels = self.kpf_data_levels[qc_name]
                data_type = self.data_types[qc_name]
                spectrum_types = self.spectrum_types[qc_name]
                master_types = self.master_types[qc_name]
                drift_types = self.drift_types[qc_name]
                required_data_products = self.required_data_products[qc_name]
                keyword = self.fits_keywords[qc_name]
                keyword_fail_value = self.fits_keyword_fail_value[qc_name]
                comment = self.fits_comments[qc_name]
                db_column = self.db_columns[qc_name]
                description = self.descriptions[qc_name]

                if data_level in self.kpf_data_levels[qc_name]:
                    print('   ' + styled_text("Name: ", style="Bold") + styled_text(qc_name, style="Bold", color="Blue"))
                    print('      ' + styled_text("Description: ", style="Bold") + description)
                    print('      ' + styled_text("Date levels: ", style="Bold") + str(kpf_data_levels))
                    print('      ' + styled_text("Date type: ", style="Bold") + data_type)
                    print('      ' + styled_text("Required data products: ", style="Bold") + str(required_data_products))
                    print('      ' + styled_text("Spectrum types (applied to): ", style="Bold") + str(spectrum_types))
                    print('      ' + styled_text("Master types (required for): ", style="Bold") + str(master_types))
                    print('      ' + styled_text("Drift types (required for): ", style="Bold") + str(drift_types))
                    print('      ' + styled_text("Keyword: ", style="Bold") + styled_text(keyword, style="Bold", color='Blue'))
                    print('      ' + styled_text("Keyword fail value: ", style="Bold") + str(keyword_fail_value))
                    print('      ' + styled_text("Comment: ", style="Bold") + comment)
                    print('      ' + styled_text("Database column: ", style="Bold") + str(db_column))
                    print()

    def search_for_QC_keywords_in_files(self):
        """
        This method checks if each QC keyword is listed in two places and
        prints the results with green and red highlighting.  The two places
        are: 1) .yaml plot configuration files for the time series database,
        2) .csv files that define the time series database structure, and xxx.
        It is best used in an interactive environment, e.g., in a Jupyter
        notebook.
        """

        cases = ['plots', 'database']

        for case in cases:

            if case == 'plots':
                search_directory = '/code/KPF-Pipeline/static/tsdb_plot_configs/'
                file_ext = '.yaml'
            if case == 'database':
                search_directory = '/code/KPF-Pipeline/static/tsdb_keywords/'
                file_ext = '.csv'

            print(styled_text(f"Searching for *{file_ext} files in {search_directory} for QC keywords.", style="Bold"))
            for name in self.names:
                fits_kwd = self.fits_keywords.get(name, "")
                if not fits_kwd:
                    print(f"Warning: No search string found for '{name}'")
                    continue
                found_occurrence = False
                for root, dirs, files in os.walk(search_directory):
                    for file_name in files:
                        if file_name.endswith(file_ext):
                            full_path = os.path.join(root, file_name)
                            # Read the file contents and check for the string
                            with open(full_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if fits_kwd in content:
                                    found_occurrence = True
                                    print(styled_text(f"Found ", color="Green") + styled_text(f"'{fits_kwd}' from '{name}'", style="Bold", color="Green") + styled_text(f" in: {full_path}", color="Green"))
                if not found_occurrence:
                    print(styled_text(f"No occurrence of ", color="Red") + styled_text(f"'{name}' => '{fits_kwd}'", style="Bold", color="Red") + styled_text(f" found in any {file_ext} file.", color="Red"))
            print()


    def get_required_QCs(self, data_product=None, source=None):
        """
        This method returns a list of QC keywords (or resulting keywords) that 
        are required for particular category of data product (master, drift) 
        and of a particular source (e.g., etalon, LFC, bias)
        """

        if data_product == None or source == None:
            self.logger.error("data_product or source not specified in get_required_QCs()")
            return None

        QC_keywords = []
        for qc_name in self.names:
            if data_product.lower() == 'master':
                 if any(x in self.master_types[qc_name] for x in [source, 'all']):
                      QC_keywords.append(self.fits_keywords[qc_name])
            elif data_product.lower() == 'drift':
                 if any(x in self.drift_types[qc_name] for x in [source, 'all']):
                      QC_keywords.append(self.fits_keywords[qc_name])
        
        return QC_keywords

            

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

        if (str(type(value)) == "<class 'bool'>") or (str(type(value)) == "<class 'numpy.bool'>"):
            if value == True:
                value = 1
            else:
                value = 0

        keyword = self.qcdefinitions.fits_keywords[qc_name]
        comment = self.qcdefinitions.fits_comments[qc_name]

        self.kpf_object.header['PRIMARY'][keyword] = (value,comment)
        if debug:
            print('---->add_qc_keyword_to_header: qc_name, keyword, value, comment = {}, {}, {}, {}'.format(qc_name,keyword,value,comment))


    def not_junk(self, junk_ObsIDs_csv='/data/reference/Junk_Observations_for_KPF.csv', debug=False):
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
        if debug:
            self.logger.info(f"The datetime of ObsID is {datetime_ObsID}.")

        if os.path.exists(kfpera_csv):
            try:
                df_kpfera = pd.read_csv(kfpera_csv)
                if debug:
                    self.logger.info(f'Read the KPFERA file {kfpera_csv}.')
                nrows = len(df_kpfera)
                for i in np.arange(nrows):
                    starttime = datetime.strptime(df_kpfera.iloc[i].iloc[1].strip(), '%Y-%m-%d %H:%M:%S')
                    stoptime  = datetime.strptime(df_kpfera.iloc[i].iloc[2].strip(), '%Y-%m-%d %H:%M:%S')
                    if (datetime_ObsID > starttime) and (datetime_ObsID < stoptime):
                        KPFERA = float(df_kpfera.iloc[i].iloc[0])
                        if debug:
                            self.logger.info(f'Setting KPFERA = {KPFERA}')
            except Exception as e:
                self.logger.info(f"Exception: {e}")
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


    def L0_data_products(self, debug=False):
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

        try:
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
                self.logger.info('Data products expected in this L0 file: ' + str(data_products))

            # Use helper funtion to get data products and check their characteristics.
            QC_pass = True
            data_products_present = get_data_products_L0(L0)
            if debug:
                self.logger.info('Data products in L0 file: ' + str(data_products_present))

            # Check for specific data products
            possible_data_products = ['Green', 'Red', 'CaHK', 'ExpMeter', 'Guider', 'Telemetry', 'Pyrheliometer']
            if debug:
                self.logger.info('Possible data products in L0 file: ' + str(possible_data_products))
            for dp in possible_data_products:
                if dp in data_products:
                    if not dp in data_products_present:
                        QC_pass = False
                        if debug:
                            self.logger.info(dp + ' not present in L0 file. QC(L0_data_products) failed.')

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def L0_header_keywords_present(self, essential_keywords=['auto'], debug=False):
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

        try:
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

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def L0_datetime(self, debug=False):
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
        debug=True
        try:
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
                        self.logger.info(f'Missing keyword: {keyword}')
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
                    self.logger.info(f'(DATE-END - DATE-BEG) - ELASPED = {abs((date_end - date_beg).total_seconds() - elapsed)} sec > {time_precision_threshold} sec')
                QC_pass = False

            # Check that GRDATE-B/RDDATE-B are consistent with DATE-BEG, etc.
            data_products = get_data_products_L0(L0)
            if 'Green' in data_products:
                if 'GRDATE-B' not in L0.header['PRIMARY']:
                    if debug:
                        self.logger.info(f'Missing keyword: GRDATE-B')
                    QC_pass = False
                    return QC_pass
                else:
                    grdate_b = datetime.strptime(L0.header['PRIMARY']['GRDATE-B'], date_format)
                    if abs((date_beg - grdate_b).total_seconds()) > time_precision_threshold:
                        if debug:
                            self.logger.info(f'abs(DATE-BEG - GRDATE-B) = {abs((date_beg - grdate_b).total_seconds())} sec > {time_precision_threshold} sec')
                        QC_pass = False
                if 'GRDATE-E' not in L0.header['PRIMARY']:
                    if debug:
                        self.logger.info(f'Missing keyword: GRDATE-E')
                    QC_pass = False
                    return QC_pass
                else:
                    grdate_e = datetime.strptime(L0.header['PRIMARY']['GRDATE-E'], date_format)
                    if abs((date_end - grdate_e).total_seconds()) > time_precision_threshold:
                        if debug:
                            self.logger.info(f'abs(DATE-END - GRDATE-E) = {abs((date_end - grdate_e).total_seconds())} sec > {time_precision_threshold} sec')
                        QC_pass = False
            if 'Red' in data_products:
                if 'RDDATE-B' not in L0.header['PRIMARY']:
                    if debug:
                        self.logger.info(f'Missing keyword: RDDATE-B')
                    QC_pass = False
                    return QC_pass
                else:
                    rddate_b = datetime.strptime(L0.header['PRIMARY']['RDDATE-B'], date_format)
                    if abs((date_beg - rddate_b).total_seconds()) > time_precision_threshold:
                        if debug:
                            self.logger.info(f'abs(DATE-BEG - RDDATE-B) = {abs((date_beg - rddate_b).total_seconds())} sec > {time_precision_threshold} sec')
                        QC_pass = False
                if 'RDDATE-E' not in L0.header['PRIMARY']:
                    if debug:
                        self.logger.info(f'Missing keyword: RDDATE-E')
                    QC_pass = False
                    return QC_pass
                else:
                    rddate_e = datetime.strptime(L0.header['PRIMARY']['RDDATE-E'], date_format)
                    if abs((date_end - rddate_e).total_seconds()) > time_precision_threshold:
                        if debug:
                            self.logger.info(f'abs(DATE-END - RDDATE-E) = {abs((date_end - rddate_e).total_seconds())} sec > {time_precision_threshold} sec')
                        QC_pass = False
            if ('Green' in data_products) and ('Red' in data_products) and QC_pass:
                if abs((grdate_b - rddate_b).total_seconds()) > time_precision_threshold:
                    if debug:
                        self.logger.info(f'abs(GRDATE-B - RDDATE-B) = {abs((grdate_b - rddate_b).total_seconds())} sec > {time_precision_threshold} sec')
                    QC_pass = False
                if abs((grdate_e - rddate_e).total_seconds()) > time_precision_threshold:
                    if debug:
                        self.logger.info(f'abs(GRDATE-E - RDDATE-E) = {abs((grdate_e - rddate_e).total_seconds())} sec > {time_precision_threshold} sec')
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
                            self.logger.info(f"abs(L0['EXPMETER_SCI'].iloc[0]['Date-Beg-Corr'] - GRDATE-B) = {abs((exp_date_beg - grdate_b).total_seconds())} sec > {time_precision_threshold_exp} sec")
                        QC_pass = False
                    if abs((exp_date_end - grdate_e).total_seconds()) > time_precision_threshold_exp:
                        if debug:
                            self.logger.info(f"abs(L0['EXPMETER_SCI'].iloc[-1]['Date-End-Corr'] - GRDATE-E) = {abs((exp_date_end - grdate_e).total_seconds())} sec > {time_precision_threshold_exp} sec")
                        QC_pass = False
                if 'Red' in data_products:
                    if abs((exp_date_beg - rddate_b).total_seconds()) > time_precision_threshold_exp:
                        if debug:
                            self.logger.info(f"abs(L0['EXPMETER_SCI'].iloc[0]['Date-Beg-Corr'] - RDDATE-B) = {abs((exp_date_beg - rddate_b).total_seconds())} sec > {time_precision_threshold_exp} sec")
                        QC_pass = False
                    if abs((exp_date_end - rddate_e).total_seconds()) > time_precision_threshold_exp:
                        if debug:
                            self.logger.info(f"abs(L0['EXPMETER_SCI'].iloc[-1]['Date-End-Corr'] - RDDATE-E) = {abs((exp_date_end - rddate_e).total_seconds())} sec > {time_precision_threshold_exp} sec")
                        QC_pass = False

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def EM_not_saturated(self, debug=False):
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

        try:
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

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def EM_flux_not_negative(self, debug=False):
        """
        This Quality Control function checks if 20 or more consecutive elements 
        of the SCI and SKY exposure meter spectra are negative.  Negative flux 
        usually indicates over-subtraction of bias from the raw EM images.  
        The check is applied to the EM-SCI and EM-SKY fibers and returns False 
        if negative flux is detected in either.  Note that this check only works 
        for L0 files with the EXPMETER_SCI and EXPMETER_SKY extensions present.

        Args:
             debug - an optional flag.  If True, missing data products are noted.

         Returns:
             QC_pass - a boolean signifying that the QC passed (True) for failed (False)
        """

        N_in_a_row = 20 # number of negative flux elements in a row that triggers QC failure

        try:
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

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def L0_bad_readout_check(self, debug=False):
        """
        This Quality Control function checks if the desired readout time
        matches the expected readout time (within some limit). This
        mismatch idetifies a 'smeared' readout scenario that we want to junk.
        Bad readout states can also have no value for Greed/Red elapsed time.
        Bad readouts have elapsed time between 6 and 7 seconds.
        This occurs a few times per day on both cals and stars.

        Edge case: If a star has a desired exposure time larger than 7 seconds
        but the exposure meter properly terminates the exposure between
        6.0 and 6.7 seconds, the star will be improperly failed. (very rare)

        Args:
            debug - an optional flag.  If True, missing data products are noted.

            Example that should fail this QC test: KP.20241008.31459.57

        Returns:
            QC_pass - a boolean signifying that the QC passed for failed
        """

        try:
            L0 = self.kpf_object

            Texp_desired = L0.header['PRIMARY']['EXPTIME'] # desired exptime
            Texp_actual  = L0.header['PRIMARY']['ELAPSED'] # actual exposure time

            if (Texp_desired >= 7) and (6.0 < Texp_actual <= 6.6):
                QC_pass = False
            else:
                QC_pass = True

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def NTP_timing(self, max_timing_error_ms=100, debug=False):
        """
        This Quality Control function checks if the status message from 
        the Network Time Protocol (NTP) indicates that the timing uncertainty
        is less than max_timing_error_ms.

        Args:
            debug - an optional flag.  If True, missing data products are noted.

        Returns:
            QC_pass - a boolean signifying that the QC passed for failed
        """

        QC_pass = False
        try:
            L0 = self.kpf_object
            if 'TIMEERR' in L0.header['PRIMARY']:
                NTP_string = L0.header['PRIMARY']['TIMEERR'] # NTP status message
                match = re.search(r'NTP time correct to within ([\d.]+) ms', NTP_string)
                if match:
                    timing_error_ms = float(match.group(1))
                    if timing_error_ms < max_timing_error_ms:
                        QC_pass = True

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass

        return QC_pass


    def good_guiding(self, max_guider_rms_mas=50, max_guider_offset_mas=50, debug=False):
        """
        This Quality Control function checks if the Guider has an RMS guiding
        performance of 50 mas (settable with max_guider_rms_mas) or better
        in the X and Y directions.  The same test is applied for guiding offsets.

        Args:
            debug - an optional flag.  If True, missing data products are noted.

        Returns:
            QC_pass - a boolean signifying that the QC passed for failed
        """

        QC_pass = True
        try:
            L0 = self.kpf_object
            myGuider = AnalyzeGuider(L0)

            # Check guiding RMS
            if hasattr(myGuider, 'x_rms'):
                if myGuider.x_rms > max_guider_rms_mas:
                    QC_pass = False
            else:
                QC_pass = False
            if hasattr(myGuider, 'y_rms'):
                if myGuider.y_rms > max_guider_rms_mas:
                    QC_pass = False
            else:
                QC_pass = False

            # Check guiding offsets
            if hasattr(myGuider, 'x_bias'):
                if myGuider.x_bias > max_guider_offset_mas:
                    QC_pass = False
            else:
                QC_pass = False
            if hasattr(myGuider, 'y_bias'):
                if myGuider.y_bias > max_guider_offset_mas:
                    QC_pass = False
            else:
                QC_pass = False

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass



    def good_TARG_headers(self):
        """
        This Quality Control function checks that a set of "TARG" keywords exist
        in an L0 file and that a subset of those for reasonable values.
        
        keywords that are checked for existence: 
            TARGNAME, TARGRA, TARGDEC, TARGEPOC, TARGEQUI, TARGPLAX, 
            TARGPMDC, TARGPMRA, TARGRADV, TARGFRAM, TARGTEFF
        
        keywords that are verified to be numbers: 
            TARGEPOC, TARGEQUI, TARGPLAX, 
            TARGPMDC, TARGPMRA, TARGRADV, TARGTEFF

        Conditions checked:
            radial velocity: TARGRADV < 100 km/s
            parallax: TARGPLAX < 1 arcsec
            epoch, equinox: TARGEPOC, TARGEQUI >= 1950 and < 2050
            proper motion: TARGPMDC, TARGPMRA < 15 arcsec/yr (NOT CURRENTLY CHECKED - units uncertain)
            effective temperature: TARGTEFF < 10000 K and > 2000 K
        
        Returns:
            QC_pass - a boolean signifying that the QC passed for failed
        """

        def is_number(maybe_num):
            '''
            Return True if the input is a number (int or float) and False otherwise.
            '''
            if maybe_num == None:
                return False
            if isinstance(maybe_num, bool):
                return False
            if isinstance(maybe_num, (int, float)):
                if not np.isnan(maybe_num):
                    return True
            return False

        QC_pass = True
        try:
            L0 = self.kpf_object
            header = L0.header['PRIMARY']

            # Check that certain TARGxxxx keywords exist
            TARG_keywords = ['TARGNAME', 'TARGRA', 'TARGDEC', 'TARGEPOC', 'TARGEQUI', 'TARGPLAX', 'TARGPMDC', 'TARGPMRA', 'TARGRADV', 'TARGFRAM', 'TARGTEFF']            
            for kwd in TARG_keywords:
                if not kwd in header:
                    QC_pass = False
                    self.logger.info(f'Missing L0 keyword: {kwd}')
                    return QC_pass
            
            # Check that certain TARGxxxx keywords are numbers
            TARG_keywords = ['TARGEPOC', 'TARGEQUI', 'TARGPLAX', 'TARGPMDC', 'TARGPMRA', 'TARGRADV', 'TARGTEFF']            
            for kwd in TARG_keywords:
                if not is_number(header[kwd]):
                    QC_pass = False
                    self.logger.info(f'Keyword {kwd} = {header[kwd]} is not a number')
                    return QC_pass
            
            # Check that TARGRADV < 350 km/s (see Fig. 8 of Chubak et al. 2012 - arXiv:1207.6212)
            if 'TARGRADV' in header:
                 if abs(header['TARGRADV']) > 350:
                     QC_pass = False
                     self.logger.info(f'L0 keyword problem: abs(TARGRADV) = abs({str(header["TARGRADV"])}) > 100 km/s')
            
            # Check that TARGPLAX < 1 arcsec
            if 'TARGPLAX' in header:
                 if float(header['TARGPLAX']) >= 1 * 1000:
                     QC_pass = False
                     self.logger.info(f'L0 keyword problem: TARGPLAX = {str(header["TARGPLAX"])} > 1000 mas')
            
            # Check that TARGEPOC, TARGEQUI >= 1950 and < 2050
            kwds = ['TARGEPOC', 'TARGEQUI']
            for kwd in kwds:
                if kwd in header:
                     if (float(header[kwd]) <= 1950) or (float(header[kwd]) > 2050):
                         QC_pass = False
                         self.logger.info(f'L0 keyword problem: {kwd} = {str(header[kwd])} <= 1950 or > 2050')
            
#            # Check that TARGPMDC, TARGPMRA < 15 arcsec/yr
#            kwds = ['TARGPMDC', 'TARGPMRA']
#            for kwd in kwds:
#                if kwd in header:
#                     if float(header[kwd]) > 15:
#                          QC_pass = False
            
            # Check that TARGTEFF > 2000 K and < 10000 K
            if 'TARGTEFF' in header:
                 if (float(header['TARGTEFF']) <= 2000) or (float(header['TARGTEFF']) >= 10000):
                     QC_pass = False
            
        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

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

    def data_2D_red_green(self, debug=False):
        """
        This Quality Control function checks if the 2D data exists for both
        the red and green chips and checks that the sizes of the arrays are as expected.

        Args:
             debug - an optional flag.  If True, prints shapes of CCD arrays and other comments.

         Returns:
             QC_pass - a boolean signifying that all of the data exists as expected
        """

        try:
            D2 = self.kpf_object

            if debug:
                self.logger.info(D2.info())
                type_D2 = type(D2)
                self.logger.info("type_2D = ",type_D2)
                self.logger.info("D2 = ",D2)

            QC_pass = True

            extensions = D2.extensions

            if 'GREEN_CCD' in extensions:

                if debug:
                    self.logger.info("GREEN_CCD exists")
                    self.logger.info("data_shape =", np.shape(D2["GREEN_CCD"]))

                if np.shape(D2["GREEN_CCD"]) != (4080, 4080):
                    QC_pass = False

            else:
                if debug:
                    self.logger.info("GREEN_CCD does not exist")
                QC_pass = False

            if 'RED_CCD' in extensions:

                if debug:
                    self.logger.info("RED_CCD exists")
                    self.logger.info("data_shape =", np.shape(D2["RED_CCD"]))

                if np.shape(D2["RED_CCD"]) != (4080, 4080):
                    QC_pass = False

            else:
                if debug:
                    self.logger.info("RED_CCD does not exist")
                QC_pass = False

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def data_2D_CaHK(self,debug=False):
        """
        This Quality Control function checks if the 2D data exists for the
        Ca H&K chip and checks that the size of the array is as expected.

        Args:
             debug - an optional flag.  If True, prints shape of CaHK CCD array.

        Returns:
             QC_pass - a boolean signifying that all of the data exists as expected
        """

        try:
            D2 = self.kpf_object

            if debug:
                self.logger.info(D2.info())
                type_D2 = type(D2)
                self.logger.info("type_2D = ",type_D2)
                self.logger.info("D2 = ",D2)

            QC_pass = True

            extensions = D2.extensions

            if 'CA_HK' in extensions:

                if debug:
                    self.logger.info("CA_HK exists")
                    self.logger.info("data_shape =", np.shape(D2["CA_HK"]))

                if np.shape(D2["CA_HK"]) == (0,):
                    QC_pass = False

            else:
                if debug:
                    self.logger.info("CA_HK does not exist")
                QC_pass = False

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def data_2D_bias_low_flux(self,debug=False):
        """
        This Quality Control function checks if the flux is low
        (mean flux < 10) for a bias exposure.

        Args:
             debug - an optional flag.  If True, prints mean flux in each CCD.

        Returns:
             QC_pass - a boolean signifying that all of the data exists as expected
        """

        try:
            D2 = self.kpf_object

            if debug:
                self.logger.info(D2.info())
                type_D2 = type(D2)
                self.logger.info("type_2D = ",type_D2)
                self.logger.info("D2 = ",D2)

            QC_pass = True
            extensions = D2.extensions

            mean_GREEN = D2["GREEN_CCD"].flatten().mean()
            mean_RED = D2["RED_CCD"].flatten().mean()

            if debug:
                self.logger.info("Mean GREEN_CCD flux =", np.round(mean_GREEN, 2))
                self.logger.info("Mean RED_CCD flux =", np.round(mean_RED, 2))
                self.logger.info("Max allowed mean flux =", 10)

            if (mean_GREEN > 10) | (mean_RED > 10):
                if debug:
                    self.logger.info("One of the CCDs has a high flux")
                QC_pass = False


        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass

    def data_2D_dark_low_flux(self,debug=False):
        """
        This Quality Control function checks if the flux is low
        (mean flux < 10) for a dark exposure.

        Args:
             debug - an optional flag.  If True, prints mean flux in each CCD.

        Returns:
             QC_pass - a boolean signifying that all of the data exists as expected
        """

        try:
            D2 = self.kpf_object

            if debug:
                self.logger.info(D2.info())
                type_D2 = type(D2)
                self.logger.info("type_2D = ",type_D2)
                self.logger.info("D2 = ",D2)

            QC_pass = True
            extensions = D2.extensions

            mean_GREEN = D2["GREEN_CCD"].flatten().mean()
            mean_RED = D2["RED_CCD"].flatten().mean()

            max_allowed_mean_flux_green = 11
            max_allowed_mean_flux_red = 13

            if debug:
                self.logger.info("Mean GREEN_CCD flux =", np.round(mean_GREEN, 2))
                self.logger.info("Mean RED_CCD flux =", np.round(mean_RED, 2))
                self.logger.info("Max allowed mean flux for GREEN =", max_allowed_mean_flux_green)
                self.logger.info("Max allowed mean flux for RED =", max_allowed_mean_flux_red)

            if (mean_GREEN > max_allowed_mean_flux_green) | (mean_RED > max_allowed_mean_flux_red):
                if debug:
                    self.logger.info("One of the CCDs has a high flux")
                QC_pass = False

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass

    def positive_2D_SNR(self, neg_threshold=-5, debug=False):
        """
        This Quality Control function checks a 2D image to see if more than 1%
        of the pixel values of 'SNR' = counts / sqrt(variance) < -5.
        The value of -5 was chosen because for a definition of SNR that is
        normally distributed, this is 5-sigma low.

        Args:
             neg_threshold - the low flux threshold (default: -5, i.e., 5-sigma)
             debug - an optional flag.  If True, prints mean flux in each CCD.

        Returns:
             QC_pass - a boolean signifying that < 1% of the 2D pixels have
                       values below the threshold
        """

        D2 = self.kpf_object

        if debug:
            self.logger.info(D2.info())
            type_D2 = type(D2)
            self.logger.info("type_2D = ",type_D2)
            self.logger.info("D2 = ",D2)

        QC_pass = True
        extensions = D2.extensions

        try:
            if 'GREEN_CCD' in extensions:
                scaled_counts = np.array(D2['GREEN_CCD'].data) / np.sqrt(np.array(D2['GREEN_VAR'].data))
                subthreshold = np.sum(scaled_counts < neg_threshold)
                total_pixels = scaled_counts.size
                if debug:
                    priself.logger.infont(f'Number of pixels < {neg_threshold}: {subthreshold}')
                    self.logger.info(f'Total number of pixels: {total_pixels}')
                if ( subthreshold / total_pixels ) > 0.01:
                    QC_pass = False
        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        try:
            if 'RED_CCD' in extensions:
                scaled_counts = (np.array(D2['RED_CCD'].data) / np.sqrt(np.array(D2['RED_VAR'].data))).flatten()
                subthreshold = np.sum(scaled_counts < neg_threshold)
                total_pixels = scaled_counts.size
                if debug:
                    print(f'Number of pixels < {neg_threshold}: {subthreshold}')
                    print(f'Total number of pixels: {total_pixels}')
                if ( subthreshold / total_pixels ) > 0.1:
                    QC_pass = False
        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass

    def D2_lfc_flux(self, threshold=4000, debug=False):
        """
        This Quality Control function checks if the flux values in the green and red chips of the
        given 2D file are above a defined threshold at the 98th percentile.

        Args:
            debug

        Returns:
            QC_pass (bool): True if both green and red channels have 98th percentile values above the
                            threshold, False otherwise.
        """

        try:
            D2 = self.kpf_object
            green_counts = D2['GREEN_CCD'].data
            red_counts = D2['RED_CCD'].data

            QC_pass = True
            if debug:
                self.logger.info("******Green - 98th percentile counts: " + str(np.percentile(green_counts, 98)))
                self.logger.info("******Red - 98th percentile counts: " + str(np.percentile(red_counts, 98)))
            if np.percentile(green_counts, 98) < threshold or np.percentile(red_counts, 98) < threshold:
                QC_pass = False

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass

    def D2_master_bias_age(self, maxage=5, debug=False):
        """
        This Quality Control function checks if the master bias file used to
        process this exposure was created from files taken more than maxage
        (default: 5) days from the exposure itself.

        Args:
            debug

        Returns:
            QC_pass (bool): True if the time of exposure for the files going
                            into the master bias file were taken more than a
                            certain number of days from the exposure itself.
        """

        try:
            D2 = self.kpf_object
            my2D = Analyze2D(D2, logger=self.logger)
            age_master_file = my2D.measure_master_age(kwd='BIASFILE', verbose=debug)

            QC_pass = True
            if abs(age_master_file) > maxage:
                QC_pass = False

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass

    def D2_master_dark_age(self, maxage=5, debug=False):
        """
        This Quality Control function checks if the master dark file used to
        process this exposure was created from files taken more than maxage
        (default: 5) days from the exposure itself.

        Args:
            debug

        Returns:
            QC_pass (bool): True if the time of exposure for the files going
                            into the master dark file were taken more than a
                            certain number of days from the exposure itself.
        """

        try:
            D2 = self.kpf_object
            my2D = Analyze2D(D2, logger=self.logger)
            age_master_file = my2D.measure_master_age(kwd='DARKFILE', verbose=debug)

            QC_pass = True
            if abs(age_master_file) > maxage:
                QC_pass = False

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass

    def D2_master_flat_age(self, maxage=5, debug=False):
        """
        This Quality Control function checks if the master flat file used to
        process this exposure was created from files taken more than maxage (default: 5)
        days from the exposure itself.

        Args:
            debug

        Returns:
            QC_pass (bool): True if the time of exposure for the files going
                            into the master dark file were taken more than a
                            certain number of days from the exposure itself.
        """

        try:
            D2 = self.kpf_object
            my2D = Analyze2D(D2, logger=self.logger)
            age_master_file = my2D.measure_master_age(kwd='FLATFILE', verbose=debug)

            QC_pass = True
            if abs(age_master_file) > maxage:
                QC_pass = False

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass

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


    def monotonic_wavelength_solution(self,debug=False):
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

        try:
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

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass #, bad_orders

    def data_L1_red_green(self,debug=False):
        """
        This Quality Control function checks if the red and green data
        are present in an L1 file, and that all array sizes are as expected.

        Args:
             debug - an optional flag.  If True prints shapes of arrays.

        Returns:
             QC_pass - a boolean signifying that all of the data exists as expected
        """

        try:
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

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def data_L1_CaHK(self, debug=False):
        """
        This Quality Control function checks if the green and red data
        are present in an L1 file, and that all array sizes are as expected.

        Args:
             debug - an optional flag.  If True, prints shapes of arrays.

        Returns:
             QC_pass - a boolean signifying that all of the data exists as expected
        """

        try:
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

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def L1_check_snr_lfc(self, SNR_limit=2800):
        """
        This Quality Control function checks checks the SNR of
        LFC frames, marking satured frames as failing the test.

        Args:
            SNR_limit - max allowable SNR at two wavelengths (548 Ang and 747 Ang)
        Returns:
            QC_pass - a boolean signifying that the QC passed or failed
        """

        try:
            L1 = self.kpf_object
            myL1 = AnalyzeL1(L1, logger=self.logger)
            myL1.measure_L1_snr(snr_percentile=95)
            SNR_452 = myL1.GREEN_SNR[1,-1]  # SNRSC452 - SNR of L1 SCI spectrum (SCI1+SCI2+SCI3) near 452 nm (second bluest order); on Green CCD
            SNR_548 = myL1.GREEN_SNR[25,-1] # SNRSC548 - SNR of L1 SCI spectrum (SCI1+SCI2+SCI3) near 548 nm; on Green CCD
            SNR_652 = myL1.RED_SNR[8,-1]    # SNRSC652 - SNR of L1 SCI spectrum (SCI1+SCI2+SCI3) near 652 nm; on Red CCD
            SNR_747 = myL1.RED_SNR[20,-1]   # SNRSC747 - SNR of L1 SCI spectrum (SCI1+SCI2+SCI3) near 747 nm; on Red CCD
            SNR_852 = myL1.RED_SNR[30,-1]   # SNRSC852 - SNR of L1 SCI spectrum (SCI1+SCI2+SCI3) near 852 nm (second reddest order); on Red CCD

            QC_pass = True
            for SNR in [SNR_452, SNR_548, SNR_652, SNR_747, SNR_852]:
                if SNR >= SNR_limit:
                    QC_pass = False

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def L1_correct_wls_check(self, debug=False):
        """
        This Quality Control function checks if the WLS files used by a given L1
        file are correct. Failure states are as follows:
            (1) The two WLS files do not exist or cannot be opened.
            (2) The two WLS files are the same.
            (3) If data was taken at night, the first WLS file does not correspond
                to that from the prior evening and/or the second WLS file does
                not correspond to that from the following morning.
            (4) If data was taken during the day, the first WLS file does not
                correspond to that from the prior morning and/or the second
                WLS file does not correspond to that from the following evening.

        Args:
             L1 - an L1 object
             debug - an optional flag.  If True, missing data products are noted.

         Returns:
             QC_pass - a boolean signifying that the QC passed for failed
        """

        try:
            L1 = self.kpf_object
            QC_pass = True

            # First, check if WLS files exist
            try:
                WLSFILE = L1.header["PRIMARY"]["WLSFILE"]
                WLSFILE2 = L1.header["PRIMARY"]["WLSFILE2"]
                from kpfpipe.models.level1 import KPF1
                WLSFILE_L1 = KPF1.from_fits(WLSFILE)
                WLSFILE2_L1 = KPF1.from_fits(WLSFILE2)
            except:
                QC_pass = False
                if debug:
                    print("WLSFILE and/or WLSFILE2 does not exist or failed to be read.")
                return QC_pass

            # Next, check if the two WLS files are the same (they should not be)
            if WLSFILE == WLSFILE2:
                QC_pass = False
                if debug:
                    print("WLSFILE and WLSFILE2 are the same.")
                return QC_pass

            # Check if the observations are Keck or SoCal observations
            is_day = False
            if L1.header["PRIMARY"]["OBJECT"] == "SoCal":
                is_day = True

            # If is_day == False, make sure the UTC dates of the WLS agree with the UTC date of the observation
            # If is_day == True, make sure WLSFILE has the same date as the observation and WLSFILE2 has a date one day later
            date_format = "%Y-%m-%d"
            DATE_OBS = datetime.strptime(L1.header["PRIMARY"]["DATE-OBS"], date_format)
            WLSFILE_DATE = datetime.strptime(WLSFILE_L1.header["PRIMARY"]["DATE-OBS"], date_format)
            WLSFILE2_DATE = datetime.strptime(WLSFILE2_L1.header["PRIMARY"]["DATE-OBS"], date_format)
            if is_day == False:
                if DATE_OBS != WLSFILE_DATE:
                    QC_pass = False
                    if debug:
                        print("Date of WLSFILE not the same as date of obs.")
                if DATE_OBS != WLSFILE2_DATE:
                    QC_pass = False
                    if debug:
                        print("Date of WLSFILE2 not the same as date of obs.")
            else:
                if DATE_OBS != WLSFILE_DATE:
                    QC_pass = False
                    if debug:
                        print("Date of WLSFILE not the same as date of obs.")
                if DATE_OBS >= WLSFILE2_DATE:
                    QC_pass = False
                    if debug:
                        print("Date of WLSFILE2 for SoCal obs is not after date of obs.")

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def L1_WLSFILE_age(self, maxage=2, debug=False):
        """
        This Quality Control function checks if the wavelength solution file
        WLSFILE that was used to calibrated the wavelengths in this spectrum
        was created from files taken more than maxage (default: 2) days from
        the exposure itself.

        Args:
            debug: if True, print debugging statements

        Returns:
            QC_pass (bool): True if the time of exposure for the files going
                            into WLSFILE were taken more than a
                            certain number of days from the exposure itself.
        """

        try:
            L1 = self.kpf_object
            myL1 = AnalyzeL1(L1, logger=self.logger)
            age_wls_file = myL1.measure_WLS_age(kwd='WLSFILE', verbose=debug)

            QC_pass = True
            if age_wls_file == None:
                QC_pass = False
            elif abs(age_wls_file) > maxage:
                QC_pass = False

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def L1_WLSFILE2_age(self, maxage=2, debug=False):
        """
        This Quality Control function checks if the wavelength solution file
        WLSFILE2 that was used to calibrated the wavelengths in this spectrum
        was created from files taken more than maxage (default: 2) days from
        the exposure itself.

        Args:
            debug: if True, print debugging statements

        Returns:
            QC_pass (bool): True if the time of exposure for the files going
                            into WLSFILE2 were taken more than a
                            certain number of days from the exposure itself.
        """

        try:
            L1 = self.kpf_object
            myL1 = AnalyzeL1(L1, logger=self.logger)
            age_wls_file = myL1.measure_WLS_age(kwd='WLSFILE2', verbose=debug)

            QC_pass = True
            if age_wls_file == None:
                QC_pass = False
            elif abs(age_wls_file) > maxage:
                QC_pass = False

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def L1_FLAT_SNR(self, snr_min=40, snr_max=1600, debug=False):
        """
        Checks each chip/order/orderlet to see if the peak-of-blaze SNR in a flat 
        is greater than the specified minimum (snr_min) and less than a specified 
        maximum (snr_max).

        Args:
            snr_min: minimum SNR threshold applied to all chips/orders/orderlets
            snr_max: maximum SNR threshold applied to all chips/orders/orderlets
                     SNR max is for SCI1/SCI2/SCI3/SKY.  The value for CAL is 
                     3/4 of the value for the other orders
            debug: if True, print debugging statements

        Returns:
            QC_pass (bool): True if the SNR of each order and orderlet for the 
            green and red CCDs are above a threshold.
        """

        try:
            L1 = self.kpf_object
            myL1 = AnalyzeL1(L1, logger=self.logger)
            myL1.measure_L1_snr()
            data_products = get_data_products_L1(L1)
            use_CAL, use_SCI, use_SKY = False, False, False
            if 'CAL-OBJ' in myL1.L1.header['PRIMARY']:
                if myL1.L1.header['PRIMARY']['CAL-OBJ'] == 'BrdbandFiber':
                    use_CAL = True
            if 'SCI-OBJ' in myL1.L1.header['PRIMARY']:
                if myL1.L1.header['PRIMARY']['SCI-OBJ'] == 'BrdbandFiber':
                    use_SCI = True
            if 'SKY-OBJ' in myL1.L1.header['PRIMARY']:
                if myL1.L1.header['PRIMARY']['SKY-OBJ'] == 'BrdbandFiber':
                    use_SKY = True

            QC_pass = True
            if 'Green' in data_products:
                if use_CAL:
                    QC_pass = QC_pass & (myL1.GREEN_SNR[:,0] < snr_max).all() 
                if use_SCI:
                    QC_pass = QC_pass & (myL1.GREEN_SNR[:,1:3] < snr_max).all() 
                if use_SKY:
                    QC_pass = QC_pass & (myL1.GREEN_SNR[1:,4] < snr_max).all() 
            if 'Red' in data_products:
                if use_CAL:
                    QC_pass = QC_pass & (myL1.RED_SNR[:,0] < snr_max).all() 
                if use_SCI:
                    QC_pass = QC_pass & (myL1.RED_SNR[:,1:3] < snr_max).all() 
                if use_SKY:
                    QC_pass = QC_pass & (myL1.RED_SNR[1:,4] < snr_max).all() 
            if (not 'Green' in data_products) and (not 'Red' in data_products):
                QC_pass = False
            if (not use_CAL) and (not use_SCI) and (not use_SKY):
                QC_pass = False
                
        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False
            
        return QC_pass


    def L1_LFC_lines(self, intensity_thresh=40**2, min_lines=100, divisions_per_order=8, debug=False):
        """
        Checks the quality of LFC spectra by examining the number and distribution 
        of emissions lines per chip/order/orderlet.  It checks that each order 
        has at least min_lines with amplitude intensity_thresh.  It also checks 
        that there is at least one line of that amplitude in 8 (set by 
        divisions_per_order) equal-spaced regions per order.

        Args:
            intensity_thresh (float): minimum line amplitude to be considered good
            min_lines (int):          minimum number of lines in a spectral 
                                      order for it to be considered good
            divisions_per_order (int): number of contiguous subregions each order 
                                       must have at least one peak in
            debug: if True, log debugging statements

        Returns:
            QC_pass (bool): True if each order and orderlet for the 
            green and red CCDs are pass quality checks related to the 
            intensity, number, and distribution of emissions per order.
        """

        try:
            L1 = self.kpf_object
            myL1 = AnalyzeL1(L1, logger=self.logger)
            data_products = get_data_products_L1(L1)

            # Compute good orders
            if 'Green' in data_products:
                SCI_g_fl, CAL_g_fl, SKY_g_fl = myL1.measure_good_comb_orders(chip='green', 
                                               intensity_thresh=intensity_thresh, 
                                               min_lines=min_lines, 
                                               divisions_per_order=divisions_per_order)
            if 'Red' in data_products:
                SCI_r_fl, CAL_r_fl, SKY_r_fl = myL1.measure_good_comb_orders(chip='red', 
                                               intensity_thresh=intensity_thresh, 
                                               min_lines=min_lines, 
                                               divisions_per_order=divisions_per_order)
            
            # Special cases for (order 34 CAL on the Green CCD) and 
            #                   (order  0 SKY on the Red   CCD), which are not fully extracted.
            if CAL_g_fl[1] !=  None:
                if CAL_g_fl[1] == 33:
                    CAL_g_fl[1] = 34
            if SKY_r_fl[0] !=  None:
                if SKY_r_fl[0] == 1:
                    SKY_r_fl[0] = 0

            # usable orders are copied from modules/wavelength_cal/confits/LFC_KPF_{green,red}.cfg
            # a future version of this code would use GetCalibrations for this
            usable_orders_g = [2, 34]
            usable_orders_r = [0, 31]
            
            # Determine which fibers are illuminated by LFC
            use_CAL, use_SCI, use_SKY = False, False, False
            if 'CAL-OBJ' in myL1.L1.header['PRIMARY']:
                if myL1.L1.header['PRIMARY']['CAL-OBJ'] == 'LFCFiber':
                    use_CAL = True
            if 'SCI-OBJ' in myL1.L1.header['PRIMARY']:
                if myL1.L1.header['PRIMARY']['SCI-OBJ'] == 'LFCFiber':
                    use_SCI = True
            if 'SKY-OBJ' in myL1.L1.header['PRIMARY']:
                if myL1.L1.header['PRIMARY']['SKY-OBJ'] == 'LFCFiber':
                    use_SKY = True

            # Check line quality for various combinations of SCI/CAL/SKY and Green/Red 
            QC_pass = True
            if 'Green' in data_products:
                if use_SCI:
                    if None in SCI_g_fl:
                        QC_pass = False
                        if debug:   
                            self.logger.debug('Green SCI orders: False')
                    else:
                        QC_pass = QC_pass & (SCI_g_fl[0] <= usable_orders_g[0]) & (SCI_g_fl[1] >= usable_orders_g[1])
                        if debug:   
                            self.logger.debug('Green SCI orders: ' + str((SCI_g_fl[0] <= usable_orders_g[0]) & (SCI_g_fl[1] >= usable_orders_g[1])))
                if use_CAL:
                    if None in CAL_g_fl:
                        QC_pass = False
                        if debug:   
                            self.logger.debug('Green CAL orders: False')
                    else:
                        QC_pass = QC_pass & (CAL_g_fl[0] <= usable_orders_g[0]) & (CAL_g_fl[1] >= usable_orders_g[1])
                        if debug:   
                            self.logger.debug('Green CAL orders: ' + str((CAL_g_fl[0] <= usable_orders_g[0]) & (CAL_g_fl[1] >= usable_orders_g[1])))
                if use_SKY:
                    if None in SKY_g_fl:
                        QC_pass = False
                        if debug:   
                            self.logger.debug('Green SKY orders: False')
                    else:
                        QC_pass = QC_pass & (SKY_g_fl[0] <= usable_orders_g[0]) & (SKY_g_fl[1] >= usable_orders_g[1])
                        if debug:   
                            self.logger.debug('Green SKY orders: ' + str((SKY_g_fl[0] <= usable_orders_g[0]) & (SKY_g_fl[1] >= usable_orders_g[1])))
            if 'Red' in data_products:
                if use_SCI:
                    if None in SCI_r_fl:
                        QC_pass = False
                        if debug:   
                            self.logger.debug('Red SCI orders: False')
                    else:
                        QC_pass = QC_pass & (SCI_r_fl[0] <= usable_orders_r[0]) & (SCI_r_fl[1] >= usable_orders_r[1])  
                        if debug:   
                            self.logger.debug('Red SCI orders: ' + str((SCI_r_fl[0] <= usable_orders_r[0]) & (SCI_r_fl[1] >= usable_orders_r[1])))
                if use_CAL:
                    if None in CAL_r_fl:
                        QC_pass = False
                        if debug:   
                            self.logger.debug('Red CAL orders: False')
                    else:
                        QC_pass = QC_pass & (CAL_r_fl[0] <= usable_orders_r[0]) & (CAL_r_fl[1] >= usable_orders_r[1])
                        if debug:   
                            self.logger.debug('Red CAL orders: ' + str((CAL_r_fl[0] <= usable_orders_r[0]) & (CAL_r_fl[1] >= usable_orders_r[1])))
                if use_SKY:
                    if None in SKY_r_fl:
                        QC_pass = False
                        if debug:   
                            self.logger.debug('Red SKY orders: False')
                    else:
                        QC_pass = QC_pass & (SKY_r_fl[0] <= usable_orders_r[0]) & (SKY_r_fl[1] >= usable_orders_r[1]) 
                        if debug:   
                            self.logger.debug('Red SKY orders: ' + str((SKY_r_fl[0] <= usable_orders_r[0]) & (SKY_r_fl[1] >= usable_orders_r[1])))
            if (not 'Green' in data_products) and (not 'Red' in data_products):
                QC_pass = False
            if (not use_CAL) and (not use_SCI) and (not use_SKY):
                QC_pass = False
                
        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def L1_Etalon_lines(self, intensity_thresh=40**2, min_lines=100, divisions_per_order=4, debug=False):
        """
        Checks the quality of Etalon spectra by examining the number and distribution 
        of emissions lines per chip/order/orderlet.  It checks that each order 
        has at least min_lines with amplitude intensity_thresh.  It also checks 
        that there is at least one line of that amplitude in 8 (set by 
        divisions_per_order) equal-spaced regions per order.
        This method is very similar to L1_LFC_lines.

        Args:
            intensity_thresh (float): minimum line amplitude to be considered good
            min_lines (int):          minimum number of lines in a spectral 
                                      order for it to be considered good
            divisions_per_order (int): number of contiguous subregions each order 
                                       must have at least one peak in
            debug: if True, log debugging statements

        Returns:
            QC_pass (bool): True if each order and orderlet for the 
            green and red CCDs are pass quality checks related to the 
            intensity, number, and distribution of emissions per order.
        """

        try:
            L1 = self.kpf_object
            myL1 = AnalyzeL1(L1, logger=self.logger)
            data_products = get_data_products_L1(L1)

            # Compute good orders
            if 'Green' in data_products:
                SCI_g_fl, CAL_g_fl, SKY_g_fl = myL1.measure_good_comb_orders(chip='green', 
                                               intensity_thresh=intensity_thresh, 
                                               min_lines=min_lines, 
                                               divisions_per_order=divisions_per_order)
            if 'Red' in data_products:
                SCI_r_fl, CAL_r_fl, SKY_r_fl = myL1.measure_good_comb_orders(chip='red', 
                                               intensity_thresh=intensity_thresh, 
                                               min_lines=min_lines, 
                                               divisions_per_order=divisions_per_order)
            
            # Special cases for (order 34 CAL on the Green CCD) and 
            #                   (order  0 SKY on the Red   CCD), which are not fully extracted.
            if CAL_g_fl[1] !=  None:
                if CAL_g_fl[1] == 33:
                    CAL_g_fl[1] = 34
            if SKY_r_fl[0] !=  None:
                if SKY_r_fl[0] == 1:
                    SKY_r_fl[0] = 0

            # usable orders are copied from modules/wavelength_cal/confits/LFC_KPF_{green,red}.cfg
            # a future version of this code would use GetCalibrations for this
            usable_orders_g = [2, 34]
            usable_orders_r = [0, 31]
            
            # Determine which fibers are illuminated by the Etalon
            use_CAL, use_SCI, use_SKY = False, False, False
            if 'CAL-OBJ' in myL1.L1.header['PRIMARY']:
                if myL1.L1.header['PRIMARY']['CAL-OBJ'] == 'EtalonFiber':
                    use_CAL = True
            if 'SCI-OBJ' in myL1.L1.header['PRIMARY']:
                if myL1.L1.header['PRIMARY']['SCI-OBJ'] == 'EtalonFiber':
                    use_SCI = True
            if 'SKY-OBJ' in myL1.L1.header['PRIMARY']:
                if myL1.L1.header['PRIMARY']['SKY-OBJ'] == 'EtalonFiber':
                    use_SKY = True

            # Check line quality for various combinations of SCI/CAL/SKY and Green/Red 
            QC_pass = True
            if 'Green' in data_products:
                if use_SCI:
                    if None in SCI_g_fl:
                        QC_pass = False
                        if debug:   
                            self.logger.debug('Green SCI orders: False')
                    else:
                        QC_pass = QC_pass & (SCI_g_fl[0] <= usable_orders_g[0]) & (SCI_g_fl[1] >= usable_orders_g[1])
                        if debug:   
                            self.logger.debug('Green SCI orders: ' + str((SCI_g_fl[0] <= usable_orders_g[0]) & (SCI_g_fl[1] >= usable_orders_g[1])))
                if use_CAL:
                    if None in CAL_g_fl:
                        QC_pass = False
                        if debug:   
                            self.logger.debug('Green CAL orders: False')
                    else:
                        QC_pass = QC_pass & (CAL_g_fl[0] <= usable_orders_g[0]) & (CAL_g_fl[1] >= usable_orders_g[1])
                        if debug:   
                            self.logger.debug('Green CAL orders: ' + str((CAL_g_fl[0] <= usable_orders_g[0]) & (CAL_g_fl[1] >= usable_orders_g[1])))
                if use_SKY:
                    if None in SKY_g_fl:
                        QC_pass = False
                        if debug:   
                            self.logger.debug('Green SKY orders: False')
                    else:
                        QC_pass = QC_pass & (SKY_g_fl[0] <= usable_orders_g[0]) & (SKY_g_fl[1] >= usable_orders_g[1])
                        if debug:   
                            self.logger.debug('Green SKY orders: ' + str((SKY_g_fl[0] <= usable_orders_g[0]) & (SKY_g_fl[1] >= usable_orders_g[1])))
            if 'Red' in data_products:
                if use_SCI:
                    if None in SCI_r_fl:
                        QC_pass = False
                        if debug:   
                            self.logger.debug('Red SCI orders: False')
                    else:
                        QC_pass = QC_pass & (SCI_r_fl[0] <= usable_orders_r[0]) & (SCI_r_fl[1] >= usable_orders_r[1])  
                        if debug:   
                            self.logger.debug('Red SCI orders: ' + str((SCI_r_fl[0] <= usable_orders_r[0]) & (SCI_r_fl[1] >= usable_orders_r[1])))
                if use_CAL:
                    if None in CAL_r_fl:
                        QC_pass = False
                        if debug:   
                            self.logger.debug('Red CAL orders: False')
                    else:
                        QC_pass = QC_pass & (CAL_r_fl[0] <= usable_orders_r[0]) & (CAL_r_fl[1] >= usable_orders_r[1])
                        if debug:   
                            self.logger.debug('Red CAL orders: ' + str((CAL_r_fl[0] <= usable_orders_r[0]) & (CAL_r_fl[1] >= usable_orders_r[1])))
                if use_SKY:
                    if None in SKY_r_fl:
                        QC_pass = False
                        if debug:   
                            self.logger.debug('Red SKY orders: False')
                    else:
                        QC_pass = QC_pass & (SKY_r_fl[0] <= usable_orders_r[0]) & (SKY_r_fl[1] >= usable_orders_r[1]) 
                        if debug:   
                            self.logger.debug('Red SKY orders: ' + str((SKY_r_fl[0] <= usable_orders_r[0]) & (SKY_r_fl[1] >= usable_orders_r[1])))
            if (not 'Green' in data_products) and (not 'Red' in data_products):
                QC_pass = False
            if (not use_CAL) and (not use_SCI) and (not use_SKY):
                QC_pass = False
                
        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def L1_wild_WLS(self, EXT=['CAL'], max_stdev_pixels=5, debug=False):
        """
        This Quality Control function checks if the wavelength solution for 
        SCI1, SCI2, and SCI3 on both CCDs are not "wild".  Wild is defined 
        as having a standard deviation (note: standard deviation != RMS) 
        compared to a reference wavelength solution of > 5 pixels for any 
        spectral order.

        Args:
            debug: if True, print debugging statements

        Returns:
            QC_pass (bool): True if the SCI1, SCI2, and SCI3 wavelength solutions
                            are not wild (stdev of WLS compared to reference > 5 
                            pixels for all spectral orders.)
        """

        GREEN_WAVE_extensions = []
        RED_WAVE_extensions   = []

        try:
            if 'CAL' in EXT:
                GREEN_WAVE_extensions.append("GREEN_CAL_WAVE")
                RED_WAVE_extensions.append("RED_CAL_WAVE")
            if 'SCI' in EXT:
                GREEN_WAVE_extensions.append("GREEN_SCI_WAVE1")
                GREEN_WAVE_extensions.append("GREEN_SCI_WAVE2")
                GREEN_WAVE_extensions.append("GREEN_SCI_WAVE3")
                RED_WAVE_extensions.append("RED_SCI_WAVE1")
                RED_WAVE_extensions.append("RED_SCI_WAVE2")
                RED_WAVE_extensions.append("RED_SCI_WAVE3")
            if 'SKY' in EXT:
                GREEN_WAVE_extensions.append("GREEN_SKY_WAVE")
                RED_WAVE_extensions.append("RED_SKY_WAVE")

            L1 = self.kpf_object
            myL1 = AnalyzeL1(L1, logger=self.logger)
            data_products = get_data_products_L1(L1)
            
            # Get reference wavelength solution
            dt = get_datetime_obsid(myL1.ObsID).strftime('%Y-%m-%dT%H:%M:%S.%f')
            if debug:
                print(f'DEFAULT_CALIBRATION_CFG_PATH = ' + DEFAULT_CALIBRATION_CFG_PATH)
            GC = GetCalibrations(dt, DEFAULT_CALIBRATION_CFG_PATH, use_db=False)
            wls_filename = GC.lookup(subset=['rough_wls']) 
            if debug:
                print(f'wls_filename = ' + wls_filename['rough_wls'])
            L1_ref = KPF1.from_fits(wls_filename['rough_wls'])
            myL1_ref = AnalyzeL1(L1_ref)  
            myL1_ref.add_dispersion_arrays()
            
            QC_pass = True
            if 'Green' in data_products:
                for EXT_WAVE in GREEN_WAVE_extensions:
                    if debug:
                        print(f'EXT_WAVE = ' + EXT_WAVE)
                    EXT_DISP = EXT_WAVE.replace('WAVE', 'DISP')
                    norder = myL1.L1[EXT_WAVE].shape[0]
                    for o in range(norder):
                        if not (myL1_ref.L1[EXT_DISP][o,:] == 0).all():
                            pix_diff_std = np.std((myL1.L1[EXT_WAVE][o,:] - myL1_ref.L1[EXT_WAVE][o,:]) / myL1_ref.L1[EXT_DISP][o,:])
                            if debug:
                                print(o, pix_diff_std)
                            if pix_diff_std > max_stdev_pixels:
                                QC_pass = False

            if 'Red' in data_products:
                for EXT_WAVE in RED_WAVE_extensions:
                    if debug:
                        print(f'EXT_WAVE = ' + EXT_WAVE)
                    EXT_DISP = EXT_WAVE.replace('WAVE', 'DISP')
                    norder = myL1.L1[EXT_WAVE].shape[0]
                    for o in range(norder):
                        if not (myL1_ref.L1[EXT_DISP][o,:] == 0).all():
                            pix_diff_std = np.std((myL1.L1[EXT_WAVE][o,:] - myL1_ref.L1[EXT_WAVE][o,:]) / myL1_ref.L1[EXT_DISP][o,:])
                            if debug:
                                print(o, pix_diff_std)
                            if pix_diff_std > max_stdev_pixels:
                                QC_pass = False

        except Exception as e:
            self.logger.error(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def L1_wild_WLS_SCI(self, max_stdev_pixels=5, debug=False):
        """
        Using the method L1_wild_WLS, this Quality Control function checks 
        if the wavelength solution for SCI1, SCI2, and SCI3 on both CCDs 
        are not "wild".  Wild is defined as having a standard deviation 
        (note: standard deviation != RMS) compared to a reference wavelength 
        solution of > 5 pixels for any spectral order.

        Args:
            debug: if True, print debugging statements

        Returns:
            QC_pass (bool): True if the SCI1, SCI2, and SCI3 wavelength solutions
                            are not wild (stdev of WLS compared to reference > 5 
                            pixels for all spectral orders.)
        """
        
        QC_pass = False

        try: 
            QC_pass = self.L1_wild_WLS(EXT=['SCI'], max_stdev_pixels=max_stdev_pixels, debug=debug)
        except Exception as e:
            self.logger.error(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def L1_wild_WLS_SKY(self, max_stdev_pixels=5, debug=False):
        """
        Using the method L1_wild_WLS, this Quality Control function checks 
        if the wavelength solution for SKY on both CCDs 
        are not "wild".  Wild is defined as having a standard deviation 
        (note: standard deviation != RMS) compared to a reference wavelength 
        solution of > 5 pixels for any spectral order.

        Args:
            debug: if True, print debugging statements

        Returns:
            QC_pass (bool): True if the SKY wavelength solutions
                            are not wild (stdev of WLS compared to reference > 5 
                            pixels for all spectral orders.)
        """
        
        QC_pass = False

        try: 
            QC_pass = self.L1_wild_WLS(EXT=['SKY'], max_stdev_pixels=max_stdev_pixels, debug=debug)
        except Exception as e:
            self.logger.error(f"Exception: {e}")
            QC_pass = False

        return QC_pass


    def L1_wild_WLS_CAL(self, max_stdev_pixels=5, debug=False):
        """
        Using the method L1_wild_WLS, this Quality Control function checks 
        if the wavelength solution for CAL on both CCDs 
        are not "wild".  Wild is defined as having a standard deviation 
        (note: standard deviation != RMS) compared to a reference wavelength 
        solution of > 5 pixels for any spectral order.

        Args:
            debug: if True, print debugging statements

        Returns:
            QC_pass (bool): True if the CAL wavelength solutions
                            are not wild (stdev of WLS compared to reference > 5 
                            pixels for all spectral orders.)
        """
        
        QC_pass = False

        try: 
            QC_pass = self.L1_wild_WLS(EXT=['CAL'], max_stdev_pixels=max_stdev_pixels, debug=debug)
        except Exception as e:
            self.logger.error(f"Exception: {e}")
            QC_pass = False

        return QC_pass

# Consider if this QC should use the receipt for keywords to determine DRP version numbers
#    def DRP_version_equal_2D_L1(self, debug=False):
#        """
#        Check if the same DRP produced 2D and L1.
#        """
#        
#        QC_pass = False
#
#        try: 
#            if hasattr(self.kpf_object.header, 'DRPTAG2D'):
#                DRPTAG2D = self.kpf_object.header['DRPTAG2D']
#            else:
#                DRPTAG2D = None
#            if hasattr(self.kpf_object.header, 'DRPTAGL1'):
#                DRPTAGL1 = self.kpf_object.header['DRPTAGL1']
#            elif:
#                
#            else:
#                DRPTAG2D = None
#            QC_pass = ...
#        except Exception as e:
#            self.logger.error(f"Exception: {e}")
#            QC_pass = False
#
#        return QC_pass


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

    def data_L2(self,debug=False):
        """
        This Quality Control function checks if all of the
        expected data (telemetry, CCFs, and RVs) are present.

        Args:
             debug - an optional flag.  If True, prints shapes of arrays.

        Returns:
             QC_pass - a boolean signifying that all of the data exists as expected
        """

        try:
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

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass

    def L2_datetime(self, debug=False):
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

        try:
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

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass
    
    def L2_barycentric_rv_percent_change(self, pos_threshold=1.0, neg_threshold=-1.0, debug=False):
        """
        This QC module checks the MAXPCBCV and MINPCBCV headers within the L2
        primary headers in an L2 object. These headers represent the maximum and
        minimum percent changes from the weighted average CCFBCV (for non-zero-
        weight spectral orders). If an observation's maximum or minimum percent 
        change is greater or less than 1/-1% (respectively), then the method
        returns False. 
        Args:
             pos_threshold - The high percent change threshold (e.g., no orders
                             with a percent change greater than 1%)
             neg_threshold - The low percent change threshold (e.g., no orders
                             with a percent change less than -1%)
             debug - an optional flag.  If True, prints MAXPCBCV/MINPCBCV.
        """

        try:
            L2 = self.kpf_object
            myL2 = AnalyzeL2(L2, logger=self.logger)
            QC_pass = True

            max = myL2.Max_Perc_Delta_Bary_RV
            min = myL2.Min_Perc_Delta_Bary_RV
            if debug:
                print(f"max: {max}, min: {min}")
            if max > pos_threshold :
                QC_pass = False
                return QC_pass
            if min < neg_threshold :
                QC_pass = False
                return QC_pass

        except Exception as e:
            self.logger.info(f"Exception: {e}")
            QC_pass = False

        return QC_pass

