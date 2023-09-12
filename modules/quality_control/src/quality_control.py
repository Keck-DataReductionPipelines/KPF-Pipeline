import numpy as np
import numpy.ma as ma

"""
This module contains classes for KPF data quality control (QC).  Various QC metrics are defined in
class QCDefinitions.  Other classes QCL0, QCL1, and QCL0 contain methods to compute QC values,
which are with the QC metrics, for specific data products, and then store them in the primary header
of the corresponding FITS file.  Normally QC values are stored first in the FITS header, but storage
in the KPF pipeline-operations database may be set up later by the database administrator, depending
upon the special requirements for some QC metrics.
"""

iam = 'quality_control'
version = '1.2'

"""
The following are methods common across data levels, which are given at the beginning
of this module, before the QC classes are defined.

Includes helper functions that compute statistics of data of arbitrary shape.
"""

#####################################
# Module helper functions.
#####################################

#
# Print software version.
#
def what_am_i():
    print('Software version:',iam + ' ' + version)

#
# Statistics with outlier rejection (n-sigma data-trimming), ignoring NaNs, across all data array dimensions.
#
def avg_data_with_clipping(data_array,n_sigma = 3.0):

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

#
# Check whether a file is not junk (i.e., = False if junk)
#
def not_junk_check(kpfobs, junk_ObsIDs_csv='/code/KPF-Pipeline/Junk_Observations_for_KPF.csv', debug=False):
    """
    This Quality Control function checks if the input (possibly an array) is in the list of junked files.

    Args:
         kpfobs - possible formats: 1. a single ObsID (string) (e.g. 'KP.20230621.27498.77')
                                    2. a list of ObsIDs (e.g., ['KP.20230621.27611.73', 'KP.20230621.27498.77])
                                    3. a single KPF L0/2D/L1/L2 object
                                    4. a list of KPF L0/2D/L1/L2 objects
         junk_ObsIDs_csv - a CSV with ObsIDs (e.g., 'KP.20230621.27498.77') in the first column
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

    import os
    import numpy as np
    import pandas as pd

    QC_pass = True                # Assume not junk unless explicitly listed in junk_ObsIDs_csv

    # read list of junk files
    if os.path.exists(junk_ObsIDs_csv):
        df_junk = pd.read_csv(junk_ObsIDs_csv)
        if debug:
            print(f'Read the junk file {junk_ObsIDs_csv}.')
    else:
        print(f"The file {junk_ObsIDs_csv} does not exist.")
        return QC_pass

    # initialize variables
    if not type(kpfobs) is list: # convert input to list if necessary
        kpfobs = [kpfobs]
        input_not_list = True # note for later that the input was not a list
    else:
        input_not_list = False

    # convert inputs to ObsIDs (strings), if needed
    for i in range(len(kpfobs)):
        if not (type(kpfobs[i]) is str):
            kpfobs[i] = (kpfobs[i].filename).replace('.fits', '') # this line assumes that kpfobs[i] is a L0/2D/L1/L2 object
            kpfobs[i] = kpfobs[i].replace('_2D', '') # drop _2D suffix, if needed
            kpfobs[i] = kpfobs[i].replace('_L1', '') # drop _L1 suffix, if needed
            kpfobs[i] = kpfobs[i].replace('_L2', '') # drop _L2 suffix, if needed

    # loop through inputs and determine junk status of each
    QC_pass = np.ones(len(kpfobs), dtype=bool) # starting QC values
    for i, obs in enumerate(kpfobs):
        QC_pass[i] = not (df_junk['observation_id'].isin([obs])).any()
        if debug:
            print(f'{obs} is a Junk file: ' + str(not QC_pass[i]))

    # remove list format if input was a single element
    if input_not_list:
        QC_pass = QC_pass[0]

    return QC_pass


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
        data_types (dictionary of strings): Each entry specifies the Python data type of the metric.
            Only string, int, float are allowed.  Use 0/1 for boolean.
        fits_keywords (dictionary of strings): Each entry specifies the FITS-header keyword for the metric.
            Must be 8 characters or less, following the FITS standard.
        fits_comments (dictionary of strings): Each entry specifies the FITS-header comment for the metric.
            Must be a short string for brevity (say, under 35 characters), following the FITS standard.
        db_columns (dictionary of strings): Each entry specifies either database_table.column if applicable,
            or None if not.
        methods (dictionary of lists): Each entry specifies a list of methods that apply to the metric.
    """

    def __init__(self):

        self.names = []
        self.descriptions = {}
        self.data_types = {}
        self.fits_keywords = {}
        self.fits_comments = {}
        self.db_columns = {}
        self.methods = {}

        # Define the QC metrics here.

        name1 = 'jarque_bera_test_red_amp1'
        self.names.append(name1)
        self.descriptions[name1] = 'Jarque-Bera test of pixel values for RED AMP-1 detector.'
        self.data_types[name1] = 'float'
        self.fits_keywords[name1] = 'JBTRED1'
        self.fits_comments[name1] = 'J-B test for RED AMP-1 detector'
        self.db_columns[name1] = None
        self.methods[name1] = ["add_qc_keyword_to_header"]

        name2 = 'monotonic_wavelength_solution_check'
        self.names.append(name2)
        self.descriptions[name2] = 'Check if wavelength solution is monotonic.'
        self.data_types[name2] = 'int'
        self.fits_keywords[name2] = 'MONOTWLS'
        self.fits_comments[name2] = 'Monotonic wavelength-solution check'
        self.db_columns[name2] = None
        self.methods[name2] = ["add_qc_keyword_to_header","monotonic_check","add_qc_keyword_to_header_for_monotonic_wls"]

        name3 = 'not_junk_data_check'
        self.names.append(name3)
        self.descriptions[name3] = 'Check if data in file are not junk.'
        self.data_types[name3] = 'int'
        self.fits_keywords[name3] = 'JUNKDATA'
        self.fits_comments[name3] = 'Not-junk check'
        self.db_columns[name3] = None
        self.methods[name3] = ["add_qc_keyword_to_header"]

        # Integrity checks.

        if len(self.names) != len(self.descriptions):
            raise ValueError("Length of names list does not equal number of entries in descriptions dictionary.")

        if len(self.names) != len(self.data_types):
            raise ValueError("Length of names list does not equal number of entries in data_types dictionary.")

        if len(self.names) != len(self.fits_keywords):
            raise ValueError("Length of names list does not equal number of entries in fits_keywords dictionary.")

        if len(self.names) != len(self.db_columns):
            raise ValueError("Length of names list does not equal number of entries in db_columns dictionary.")

        keys_list = self.data_types.keys()
        for key in keys_list:
            dt = self.data_types[key]
            if dt not in ['string','int','float']:
                err_str = "Error in data type: " + dt
                raise ValueError(err_str)


    def list_qc_metrics(self):

        print("name | data_type | keyword | comment | methods | db_column | description |")

        qc_names = self.names

        for qc_name in qc_names:

            data_type = self.data_types[qc_name]
            keyword = self.fits_keywords[qc_name]
            comment = self.fits_comments[qc_name]
            methods = self.methods[qc_name]
            db_column = self.db_columns[qc_name]
            description = self.descriptions[qc_name]

            print(qc_name,"|",data_type,"|",keyword,"|",comment,"|",methods,"|",db_column,"|",description,"|")



#####################################################################
#
# Superclass QC is normally not to be called directly (although it is not an abstract class, per se).
#

class QC:

    """
    Description:
        This superclass defines QC functions in general and has common methods across
        subclasses QCL0, QC2D, QCL1, and QCL2.

    Class Attributes:
        fits_object (astropy.io object): Returned from function KPF0.from_fits(fits_filename,data_type),
            which is wrapped by function read_fits in this module.
        qcdefinitions (QCDefinitions object): Returned from constructor of QCDefinitions class.

    """

    def __init__(self,fits_object):
        self.fits_object = fits_object
        self.qcdefinitions = QCDefinitions()

    def add_qc_keyword_to_header(self,qc_name,value):

        keyword = self.qcdefinitions.fits_keywords[qc_name]
        comment = self.qcdefinitions.fits_comments[qc_name]

        self.fits_object.header['PRIMARY'][keyword] = (value,comment)



#####################################################################

class QCL0(QC):

    """
    Description:
        This class inherits QC superclass and defines QC functions for L0 files.
        Since the policy is to not modify an L0 FITS file in the archive location
        /data/kpf/L0/yyyymmdd, the class operates on the FITS object that will
        elevate to a higher data level. The QC info is inherited via the FITS header
        and will be prograted downstream in the data-reduction pipeline, and will
        eventually be written to an output FITS file.

    Class Attributes:
        data_type (string): Data type in terms of project (e.g., KPF).
        fits_filename (string): Input FITS filename (include absolute path).
        fits_object (astropy.io object): Returned from function KPF0.from_fits(fits_filename,data_type),
            which is wrapped by function read_fits in this module.
        qcdefinitions (QCDefinitions object): Returned from constructor of QCDefinitions class.

    Example python code to illustrate usage of this module in calling program:

        import modules.Utils.quality_control as qc

        qc.what_am_i()

        in_file = '/code/KPF-Pipeline/KP.20230828.40579.55.fits'
        out_file = '/code/KPF-Pipeline/junk.fits'


        fits_object = from_fits('KPF',in_file)
        qcl0 = qc.QCL0(fits_object)
        name = 'jarque_bera_test_red_amp1'
        value = 3.14159256
        qcl0.add_qc_keyword_to_header(name,value)
        to_fits(qcl0.fits_object,out_file)
    """

    # Call superclass.

    def __init__(self,fits_object):
        super().__init__(fits_object)



#####################################################################

class QC2D(QC):

    """
    Description:
        This class inherits QC superclass and defines QC functions for 2D files.

    Class Attributes:
        fits_object (astropy.io object): Returned from function KPF0.from_fits(fits_filename,data_type),
            which is wrapped by function read_fits in this module.
        qcdefinitions (QCDefinitions object): Returned from constructor of QCDefinitions class.
    """

    # Call superclass.

    def __init__(self,fits_object):
        super().__init__(fits_object)



#####################################################################

class QCL1(QC):

    """
    Description:
        This class inherits QC superclass and defines QC functions for L1 files.

    Class Attributes:
        fits_object (astropy.io object): Returned from function KPF0.from_fits(fits_filename,data_type),
            which is wrapped by function read_fits in this module.
        qcdefinitions (QCDefinitions object): Returned from constructor of QCDefinitions class.

    Example python code to illustrate usage of this module in calling program:

        import modules.Utils.quality_control as qc

        qc.what_am_i()

        l1_file = '/code/KPF-Pipeline/kpf_20230814_master_arclamp_autocal-une-sky-eve_L1.fits'

        fits_object = from_fits('KPF',l1_file)
        qcl1 = qc.QCL1(fits_object)
        qcl1.add_qc_keyword_to_header_for_monotonic_wls()
        to_fits(qcl1.fits_object,l1_file)
    """

    # Call superclass.

    def __init__(self,fits_object):
        super().__init__(fits_object)


    def add_qc_keyword_to_header_for_monotonic_wls(self,qc_name):

        keyword = self.qcdefinitions.fits_keywords[qc_name]
        comment = self.qcdefinitions.fits_comments[qc_name]
        qc_pass = self.monotonic_check(self.fits_filename)

        if qc_pass:
            value = 1
        else:
            value = 0

        self.fits_object.header['PRIMARY'][keyword] = (value,comment)

    def monotonic_check(self,L1,debug=False):
        """
        This Quality Control function checks to see if a wavelength solution is
        monotonic, specifically if wavelength decreases (or stays constant) with
        increasing array index.

        Args:
             L1 - an L1 file that the QC check is to be run on
             debug - an optional flag.  If True, nonmonotonic orders/orderlets will be noted with
                     print statements and plots.

         Returns:
             QC_pass - a boolean signifying that every order/orderlet is monotonic (or not)
             bad_orders - an array of strings listing the nonmonotonic orders and orderlets
        """

        #self.L1 = L1
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
            if ext in L1:  # Check if extension exists (e.g., if RED isn't processed)
                norders = L1[ext].shape[0]
                for o in range(norders):
                    WLS = L1[ext].data[o,:] # wavelength solution of the current order/orderlet
                    isMonotonic = np.all(WLS[:-1] >= WLS[1:]) # this expression determines monotonicity for the orderlet/order
                    if not isMonotonic:
                        QC_pass = False # the QC test fails if one order/orderlet is not monotonic
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
            print("Monotonic = " + str(QC_pass))

        return QC_pass, bad_orders




#####################################################################

class QCL2(QC):

    """
    Description:
        This class inherits QC superclass and defines QC functions for L2 files.

    Class Attributes:
        fits_object (astropy.io object): Returned from function KPF0.from_fits(fits_filename,data_type),
            which is wrapped by function read_fits in this module.
        qcdefinitions (QCDefinitions object): Returned from constructor of QCDefinitions class.

    """

    # Call superclass.

    def __init__(self,fits_object):
        super().__init__(fits_object)


