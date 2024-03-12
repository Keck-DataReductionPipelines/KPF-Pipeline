import os
import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.ndimage import convolve1d
from modules.Utils.kpf_parse import get_data_products_L0

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
        fits_keywords (dictionary of strings): Each entry specifies the FITS-header keyword for the metric.
            Must be 8 characters or less, following the FITS standard.
        fits_comments (dictionary of strings): Each entry specifies the FITS-header comment for the metric.
            Must be a short string for brevity (say, under 35 characters), following the FITS standard.
        db_columns (dictionary of strings): Each entry specifies either database_table.column if applicable,
            or None if not.
    """

    def __init__(self):

        self.names = []
        self.descriptions = {}
        self.kpf_data_levels = {} 
        self.data_types = {}
        self.fits_keywords = {}
        self.fits_comments = {}
        self.db_columns = {}

        # Define QC metrics
        name0 = 'jarque_bera_test_red_amp1'
        self.names.append(name0)
        self.descriptions[name0] = 'Jarque-Bera test of pixel values for RED AMP-1 detector.'
        self.kpf_data_levels[name0] = ['L3'] # bogus value L3 to avoid executing
        self.data_types[name0] = 'float'
        self.fits_keywords[name0] = 'JBTRED1'
        self.fits_comments[name0] = 'QC: J-B test for RED AMP-1 detector'
        self.db_columns[name0] = None

        name1 = 'not_junk_check'
        self.names.append(name1)
        self.descriptions[name1] = 'Check if file is not in list of junk files.'
        self.kpf_data_levels[name1] = ['L0', '2D', 'L1', 'L2']
        self.data_types[name1] = 'int'
        self.fits_keywords[name1] = 'NOTJUNK'
        self.fits_comments[name1] = 'QC: Not in the list of junk files check'
        self.db_columns[name1] = None

        name2 = 'monotonic_wavelength_solution_check'
        self.names.append(name2)
        self.descriptions[name2] = 'Check if wavelength solution is monotonic.'
        self.kpf_data_levels[name2] = ['L1']
        self.data_types[name2] = 'int'
        self.fits_keywords[name2] = 'MONOTWLS'
        self.fits_comments[name2] = 'QC: Monotonic wavelength-solution check'
        self.db_columns[name2] = None

        name3 = 'L0_data_products_check'
        self.names.append(name3)
        self.kpf_data_levels[name3] = ['L0']
        self.descriptions[name3] = 'Check if expected L0 data products are present with non-zero array sizes.'
        self.data_types[name3] = 'int'
        self.fits_keywords[name3] = 'DATAPRL0'
        self.fits_comments[name3] = 'QC: L0 data present check'
        self.db_columns[name3] = None

        name4 = 'L0_header_keywords_present_check'
        self.names.append(name4)
        self.kpf_data_levels[name4] = ['L0']
        self.descriptions[name4] = 'Check if expected L0 header keywords are present.'
        self.data_types[name4] = 'int'
        self.fits_keywords[name4] = 'KWRDPRL0'
        self.fits_comments[name4] = 'QC: L0 keywords present check'
        self.db_columns[name4] = None

        name5 = 'exposure_meter_not_saturated_check'
        self.names.append(name5)
        self.kpf_data_levels[name5] = ['L0']
        self.descriptions[name5] = 'Check if 2+ reduced EM pixels are within 90% of saturation in EM-SCI or EM-SKY.'
        self.data_types[name5] = 'int'
        self.fits_keywords[name5] = 'EMSAT'
        self.fits_comments[name5] = 'QC: EM not saturated check'
        self.db_columns[name5] = None

        name6 = 'exposure_meter_flux_not_negative_check'
        self.names.append(name6)
        self.kpf_data_levels[name6] = ['L0']
        self.descriptions[name6] = 'Check for negative flux in the EM-SCI and EM-SKY by looking for 20 consecuitive pixels in the summed spectra with negative flux.'
        self.data_types[name6] = 'int'
        self.fits_keywords[name6] = 'EMNEG'
        self.fits_comments[name6] = 'QC: EM not negative flux check'
        self.db_columns[name6] = None

        name7 = 'L0_datetime_checks'
        self.names.append(name7)
        self.kpf_data_levels[name7] = ['L0']
        self.descriptions[name7] = 'Check for timing inconsistencies in L0 header keywords and Exp Meter table.'
        self.data_types[name7] = 'int'
        self.fits_keywords[name7] = 'TIMCHKL0'
        self.fits_comments[name7] = 'QC: L0 times consistent check'
        self.db_columns[name7] = None

        # Integrity checks
        if len(self.names) != len(self.kpf_data_levels):
            raise ValueError("Length of kpf_data_levels list does not equal number of entries in descriptions dictionary.")

        if len(self.names) != len(self.descriptions):
            raise ValueError("Length of names list does not equal number of entries in descriptions dictionary.")

        if len(self.names) != len(self.data_types):
            raise ValueError("Length of data_types list does not equal number of entries in data_types dictionary.")

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

        print("name | data_type | keyword | comment | db_column | description |")

        qc_names = self.names

        for qc_name in qc_names:

            data_type = self.data_types[qc_name]
            keyword = self.fits_keywords[qc_name]
            comment = self.fits_comments[qc_name]
            db_column = self.db_columns[qc_name]
            description = self.descriptions[qc_name]

            print(qc_name," | ",data_type," | ",keyword," | ",comment," | ",db_column," | ",description)


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

    def __init__(self,kpf_object):
        self.kpf_object = kpf_object
        self.qcdefinitions = QCDefinitions()

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
                print(f'Read the junk file {junk_ObsIDs_csv}.')
        else:
            print(f"The file {junk_ObsIDs_csv} does not exist.")
            return QC_pass
        
        QC_pass = not (df_junk['observation_id'].isin([obsID])).any()
        if debug:
            print(f'{filename} is a Junk file: ' + str(not QC_pass[i]))
    
    
        return QC_pass

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
            print('Data products that are supposed to be in this L0 file: ' + str(data_products))
     
        # Use helper funtion to get data products and check their characteristics.
        QC_pass = True
        data_products_present = get_data_products_L0(L0)
        if debug:
            print('Data products in L0 file: ' + str(data_products_present))
    
        # Check for specific data products
        possible_data_products = ['Green', 'Red', 'CaHK', 'ExpMeter', 'Guider', 'Telemetry', 'Pyrheliometer']
        for dp in possible_data_products:
            if dp in data_products:
                if not dp in data_products_present:
                    QC_pass = False
                    if debug:
                        print(dp + ' not present in L0 file. QC(L0_data_products_check) failed.')
        
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


    def L0_datetime_checks(L0, debug=False):
        """
        This QC module performs the following checks on datetimes in the L0 primary header
        and in the Exposure Meter table (if present).  The timing checks have precision 
        thresholds to only catch significant timing errors and not trigger on small 
        differences related to machine precision or dead time in the Exposure Meter detector.
        This method returns True only if all checks pass.
        
            Time ordering: 
                DATE-BED < DATE-MID < DATE-END
            Duration consistency: 
                DATE-END - DATE-BEG = ELAPSED
            Consistency between Green/Red and overall timing:
                DATE-BEG = GRDATE-B
                DATE-BEG = GRDATE-B
                DATE-END = RDDATE-E
                DATE-END = RDDATE-E
            Consistnecy between Exposure Meter times (Date-Beg, etc.) and overall timing:
                Date-Beg = DATE-BEG
                Date-end = DATE-END
        """
    
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
        This Quality Control function checks to see if a wavelength solution is
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

