import re
import pandas as pd
from astropy.io import fits
from datetime import datetime, timedelta
from kpfpipe.models.level0 import KPF0
from kpfpipe.models.level1 import KPF1
from kpfpipe.models.level2 import KPF2

class KPFParse:

    """
    Description:
        This class contains functions to parse the headers of KPF fits files.
        To-do: add methods

    Arguments:
        None

    Attributes:
        None so far
    """

    def __init__(self, ObsID, logger=None):
        self.ObsID = ObsID
        self.datecode = ''
        if logger:
            self.logger = logger
            #self.logger.debug('KPFParse class constructor')
        else:
            self.logger = None
            #print('---->KPFParse class constructor')

class HeaderParse:
    """
    Description:
        This class contains functions to parse the headers of KPF fits files.
        This method works for both KPF DRP objects and if a fits was read using 
        astropy.io.fits.open().

    Arguments:
        KPF - an L0/2D/L1/L2 file
        extension_name = name of extension whose header is returned, e.g. 'PRIMARY'

    Attributes:
        (to-do: add list based on attributes below)
    """

    def __init__(self, KPF, extension_name, logger=None):
        if logger:
            self.logger = logger
            self.logger.debug('Initializing HeaderParse object')
        else:
            self.logger = None
        try: 
            if isinstance(KPF, fits.hdu.hdulist.HDUList):  # it was read from astropy
                self.header = KPF[extension_name].header
            else:   # assume it's an L0/2D/L1/L2 
                self.header = KPF.header[extension_name]
        except:
            self.header = None
            self.logger.error('Failed to read header.')
            print('Failed to read header.')
        self.name = '' # 'Name' of object; see get_name() function below
        self.ObsID = '' # e.g., 'KP.20230708.04519.63'


    def get_name(self, use_star_names=True):
        """
        Returns the name of the source in a spectrum.  For stellar observations, this 
        is the star's name (e.g. '185144' for HD 185144 = sigma Draconis).  
        For calibration spectra, this is the lamp name (ThAr, UNe, LFC, etalon) or 
        bias/dark.  Flats using KPF's regular fibers are distinguished from wide flats.

        Args:
            use_star_names - if True (default), this function will return the name of the star
                           - if False, this function will return 'Star' for stars
                           

        Returns:
            the source/image name
            possible values: 'Bias', 'Dark', 'Flat', 'Wide Flat', 
                             'LFC', 'Etalon', 'ThAr', 'UNe',
                             'Sun', 'Star', <starname>
        """
        try: 
            if 'IMTYPE' in self.header:
                if (('ELAPSED' in self.header) and 
                    ((self.header['IMTYPE'] == 'Bias') or (self.header['ELAPSED'] == 0))):
                        self.name = 'Bias'
                elif self.header['IMTYPE'] == 'Dark':
                    self.name = 'Dark' 
                elif self.header['FFFB'].strip().lower() == 'yes':
                        self.name = 'Wide Flat' # Flatfield Fiber (wide flats)
                elif self.header['IMTYPE'].strip().lower() == 'flatlamp':
                     if 'brdband' in self.header['OCTAGON'].strip().lower():
                        self.name = 'Flat' # Flat through regular fibers
                elif self.header['IMTYPE'].strip().lower() == 'arclamp':
                    if 'lfc' in self.header['OCTAGON'].strip().lower():
                        self.name = 'LFC'
                    if 'etalon' in self.header['OCTAGON'].strip().lower():
                        self.name = 'Etalon'
                    if 'th_' in self.header['OCTAGON'].strip().lower():
                        self.name = 'ThAr'
                    if 'u_' in self.header['OCTAGON'].strip().lower():
                        self.name = 'UNe'
                elif ((self.header['TARGNAME'].strip().lower() == 'sun') or 
                      (self.header['TARGNAME'].strip().lower() == 'socal')):
                    self.name = 'Sun' # SoCal
                if ('OBJECT' in self.header) and ('FIUMODE' in self.header):
                    if (self.header['FIUMODE'] == 'Observing'):
                        if use_star_names:
                            self.name = self.header['OBJECT']
                        else:
                            self.name = 'Star'
            else:
                self.name = ''
        except:
            self.name = ''
        return self.name


    def get_obsid(self):
        """
        Returns the ObsID for a KPF File (L0, 2D, L1, or L2).

        Args:
            None

        Returns:
            ObsID of the form 'KP.20230708.04519.63'
        """
        if 'OFNAME' in self.header:
            self.ObsID = self.header['OFNAME']
            self.ObsID = self.ObsID.replace('.fits', '')
            self.ObsID = self.ObsID.replace('_2D', '')
            self.ObsID = self.ObsID.replace('_L1', '')
            self.ObsID = self.ObsID.replace('_L2', '')
        return self.ObsID
        
    def get_read_speed(self):
        """
        This method determines the read speed of the CCDs.  
        The two options are 'fast' (~12 sec) and 'normal' (~48 sec).
        This method also reports the ACF files used (CCD waveform files) and 
        the read times for each CCD.

        Parameters:
            None 

        Attributes:
            read_speed (string) - 'fast', 'regular', 'unknown'
            green_acf (string) - name of ACF file used to read the Green CCD 
            red_acf (string) - name of ACF file used to read Red CCD 
            green_read_time (double) - seconds to read out the Green CCD 
            red_read_time (double) - seconds to read out the Red CCD 

        Returns:
            a tuple of (read_speed, green_acf, red_acf, green_read_time, red_read_time)
        """
        fast_read_time_max = 20 # sec
        datetime_format = "%Y-%m-%dT%H:%M:%S.%f"
        green_acf = 'unknown'
        red_acf = 'unknown'
        read_speed = 'unknown'
        green_read_time = 0.
        red_read_time = 0.
        
        if hasattr(self, 'header'): 
            # Green CCD Read Time
            try:
                dt1 = datetime.strptime(self.header['GRDATE'], datetime_format) # fits file write time
                dt2 = datetime.strptime(self.header['GRDATE-E'], datetime_format) # shutter-close time
                deltat = dt1-dt2
                green_read_time = deltat.total_seconds()
            except:
                pass
            # Red CCD Read Time
            try:
                dt1 = datetime.strptime(self.header['RDDATE'], datetime_format) # fits file write time
                dt2 = datetime.strptime(self.header['RDDATE-E'], datetime_format) # shutter-close time
                deltat = dt1-dt2
                red_read_time = deltat.total_seconds()
            except:
                pass
            # ACF file for Green CCD
            try:
                green_acf = self.header['GRACFFLN']
            except:
                pass
            # ACF file for Red CCD
            try:
                red_acf = self.header['RDACFFLN']
            except:
                pass
            # Determine read speed ('fast' or 'regular')
            try:
                if ('fast' in green_acf) or ('fast' in red_acf):
                    read_speed = 'fast'
                elif ('regular' in green_acf) or ('regular' in red_acf):
                    read_speed = 'regular'
                else:
                    a = green_read_time
                    b = red_read_time
                    best_read_time = min(x for x in [a, b] if x != 0) if a * b != 0 else (a or b)
                    if (best_read_time > 0) and (best_read_time < fast_read_time_max):
                        read_speed = 'fast'
                    elif best_read_time >= fast_read_time_max:
                        read_speed = 'regular'
            except:
                pass 
            return (read_speed, green_acf, red_acf, green_read_time, red_read_time)


def get_datecode(ObsID):
    """
    Extract the datecode from an ObsID or a KPF filename 

    Args:
        ObsID, e.g. 'KP.20230708.04519.63' or 'KP.20230708.04519.63_2D.fits'

    Returns:
        datecode, e.g. '20230708'
    """
    ObsID = ObsID.replace('.fits', '')
    ObsID = ObsID.replace('_2D', '')
    ObsID = ObsID.replace('_L1', '')
    ObsID = ObsID.replace('_L2', '')
    datecode = ObsID.split('.')[1]

    return datecode



def get_filename(ObsID, level='L0', fullpath=False):
    """
    Extract the datecode from an ObsID or a KPF filename 

    Args:
        ObsID, e.g. 'KP.20230708.04519.63' 
        level - 'L0', '2D', 'L1', or 'L2'
        fullpath - if True, prepends /data/L0, etc.

    Returns:
        datecode, e.g. 'KP.20230708.04519.63_2D.fits' or '/data/2D/20230708/KP.20230708.04519.63_2D.fits'
    """
    path = ''
    if fullpath:
        if level == 'L0':
            path = f'/data/L0/{get_datecode(ObsID)}/'
        elif level == '2D':
            path = f'/data/2D/{get_datecode(ObsID)}/'
        elif level == 'L1':
            path = f'/data/L1/{get_datecode(ObsID)}/'
        elif level == 'L2':
            path = f'/data/L2/{get_datecode(ObsID)}/'
    
    if level == 'L0':
        filename = f'{ObsID}.fits'
    elif level == '2D':
        filename = f'{ObsID}_2D.fits'
    elif level == 'L1':
        filename = f'{ObsID}_L1.fits'
    elif level == 'L2':
        filename = f'{ObsID}_L2.fits'

    return path + filename



def get_datecode_from_filename(filename, datetime_out=False):
    """
    Extract the datecode (YYYYMMDD) from a filename.  
    Return the string datecode or a datetime version if 
    datetime_out is set to True.
    Return None if no datetime is found

    Args:
        filename, e.g. 'kpf_20250115_master_bias_autocal-bias.fits'

    Returns:
        datecode, e.g. '20250115'
    """
    match = re.search(r"(\d{8})", filename)
    if not match:
        return None
    
    datecode = match.group(1)
    
    if datetime_out:
        return datetime.strptime(datecode, "%Y%m%d")#.date()
    else:
        return datecode
    

def get_datetime_obsid(ObsID):
    """
    Return a datetime object for an ObsID.  Note that this datetime is related 
    to the time that files were written to disk that were later assembled into 
    an L0 file was created and is not accurate at the level needed for 
    barycentric corrections.

    Args:
        ObsID, e.g. 'KP.20230708.04519.63' or 'KP.20230708.04519.63_2D.fits'

    Returns:
        datecode, e.g. '20230708'
    """
    datetime_obsid = datetime(year=2000, month=1, day=1)
    ObsID = ObsID.replace('.fits', '')
    ObsID = ObsID.replace('_2D', '')
    ObsID = ObsID.replace('_L1', '')
    ObsID = ObsID.replace('_L2', '')
    datecode = ObsID.split('.')[1]
    seconds = int(ObsID.split('.')[2])
    #print(len(ObsID.split('.')))
    
    if len(ObsID.split('.')) == 4:
        datetime_obsid = datetime(year=int(datecode[0:4]), month=int(datecode[4:6]), day=int(datecode[6:8]))
        datetime_obsid += timedelta(seconds=seconds)
    
    return datetime_obsid


def get_ObsID(file):
    """
    Returns an ObsID (like 'KP.20240113.23249.10') 
    from a filename (like '/data/L1/20240113/KP.20240113.23249.10_L1.fits').
    """
    ObsID = file.split('/')[-1]
    for substring in ['.fits', '_2D', '_L1', '_L2']:
        ObsID = ObsID.replace(substring, '')
    return ObsID


def is_ObsID(ObsID):
    """
    Returns True of the input is a properly formatted ObsID, like 'KP.20240113.23249.10'.
    """
    pattern = r'^KP\.\d{8}\.\d{5}\.\d{2}$'
    is_ObsID_bool = bool(re.match(pattern, ObsID))  
    return is_ObsID_bool


def get_data_products_expected(kpf_object, data_level):
    """
    Returns a list of data products that are expected to be available in a 
    KPF object of a given data level.
    Possible data products:
        L0: Green, Red, CaHK, ExpMeter, Guider, Telemetry, Pyrheliometer
        2D: Green, Red, CaHK, ExpMeter, Guider, Telemetry, Config, Receipt, Pyrheliometer
        L1: Green, Red, CaHK, BC, Telemetry, Config, Receipt
        L2: Green CCF, Red CCF, Green CCF RW, Red CCF RW, RV, Activity, Telemetry, Config, Receipt

    Args:
        kpf_object - a KPF L0 object 
        data_level - 'L0', '2D', 'L1', or 'L2'

    Returns:
        array of data expected data products
    """
    primary_header = HeaderParse(kpf_object, 'PRIMARY')
    header = primary_header.header
    name = primary_header.get_name() # 'Star','Sun','LFC', etc.
    data_products = ['Telemetry']
    if data_level in ['2D', 'L1', 'L2']:
        data_products.append('Config')
        data_products.append('Receipt')
    if 'GREEN' in header:
        if header['GREEN'] == 'YES':
            if data_level in ['L0', '2D', 'L1']:
                data_products.append('Green')
            if data_level in ['L2']:
                data_products.append('Green CCF')
                data_products.append('Green CCF RW')
    if 'RED' in header:
        if header['RED'] == 'YES':
            if data_level in ['L0', '2D', 'L1']:
                data_products.append('Red')
            if data_level in ['L2']:
                data_products.append('Red CCF')
                data_products.append('Red CCF RW')
    if 'CA_HK' in header:
        if header['CA_HK'] == 'YES':
            if data_level in ['L0', '2D', 'L1']:
                data_products.append('CaHK')
            if data_level in ['L2']:
                if name in ['Star', 'Sun']:
                    data_products.append('Activity') # need a better way to determine (what about FWHM, etc.)
    if ('EXPMETER' in header) or ('EXPMETER_SCI' in header) or ('EXPMETER_SKY' in header):
        if header['EXPMETER'] == 'YES':
            if data_level in ['L0', '2D']:
                data_products.append('ExpMeter')
            if data_level in ['L1']:
                if name in ['Star', 'Sun']:
                    data_products.append('BC') # Is this the best way to determine if BC is present?
    if data_level in ['L2']:
        if name in ['Star', 'Sun', 'LFC', 'Etalon', 'ThAr', 'UNe']:
            data_products.append('RV') # Is this the best way to determine if RV is present?  Should it be there for the calibrations?
    if 'GUIDE' in header: 
        if header['GUIDE'] == 'YES':
            if data_level in ['L0', '2D']:
                data_products.append('Guider')
    if hasattr(kpf_object, 'SOCAL PYRHELIOMETER'): # Is this the best way to determine if Pyrheliometer data *should* be present?
        if kpf_object['SOCAL PYRHELIOMETER'].size > 1:
            data_products.append('Pyrheliometer') 

    return data_products


def get_data_products_L0(L0):
    """
    Returns a list of data products available in an L0 file, which are:
    Green, Red, CaHK, ExpMeter, Guider, Telemetry, Config, Pyrheliometer

    Args:
        L0 - a KPF L0 object 

    Returns:
        data_products in a L0 file
    """
    data_products = []
    if hasattr(L0, 'GREEN_AMP1'):
        if L0['GREEN_AMP1'].size > 1:
            data_products.append('Green')
    if hasattr(L0, 'RED_AMP1'):
        if L0['RED_AMP1'].size > 1:
            data_products.append('Red')
    if hasattr(L0, 'CA_HK'):
        if L0['CA_HK'].size > 1:
            data_products.append('HK')
    if hasattr(L0, 'EXPMETER_SCI'):
        if L0['EXPMETER_SCI'].size > 1:
            data_products.append('ExpMeter')
    if hasattr(L0, 'GUIDER_AVG'):
        print('**** Got to 3 *****')
        if (L0['GUIDER_AVG'].size > 1):
            data_products.append('Guider')
    elif hasattr(L0, 'guider_avg'): # Early KPF files used lower case guider_avg
        #if (L0['guider_avg'].size > 1):  # this fails because it checks for GUIDER_AVG
        data_products.append('Guider')
    if hasattr(L0, 'TELEMETRY'):
        if L0['TELEMETRY'].size > 1:
            data_products.append('Telemetry')
    if hasattr(L0, 'SOCAL PYRHELIOMETER'):
        if L0['SOCAL PYRHELIOMETER'].size > 1:
            data_products.append('Pyrheliometer')
    return data_products


def get_data_products_2D(D2):
    """
    Returns a list of data products available in a D2 file, which are:
    Green, Red, CaHK, ExpMeter, Guider, Telemetry, Config, Receipt, Pyrheliometer

    Args:
        2D - a KPF 2D object 

    Returns:
        data_products in a 2D file
    """
    data_products = []
    if hasattr(D2, 'GREEN_CCD'):
        if D2['GREEN_CCD'].size > 1:
            data_products.append('Green')
    if hasattr(D2, 'RED_CCD'):
        if D2['RED_CCD'].size > 1:
            data_products.append('Red')
    if hasattr(D2, 'CA_HK'):
        if D2['CA_HK'].size > 1:
            data_products.append('HK')
    if hasattr(D2, 'EXPMETER_SCI'):
        if D2['EXPMETER_SCI'].size > 1:
            data_products.append('ExpMeter')
    if hasattr(D2, 'GUIDER_AVG'):
        if (D2['GUIDER_AVG'].size > 1):
            data_products.append('Guider')
    elif hasattr(D2, 'guider_avg'): # Early KPF files used lower case guider_avg
        if (D2['guider_avg'].size > 1):
            data_products.append('Guider')
    if hasattr(D2, 'TELEMETRY'):
        if D2['TELEMETRY'].size > 1:
            data_products.append('Telemetry')
    if hasattr(D2, 'RECEIPT'):
        if D2['RECEIPT'].size > 1:
            data_products.append('Receipt')
    if hasattr(D2, 'SOCAL PYRHELIOMETER'):
        if D2['SOCAL PYRHELIOMETER'].size > 1:
            data_products.append('Pyrheliometer')
    return data_products


def get_data_products_L1(L1):
    """
    Returns a list of data products available in a L1 file, which are:
        Green, Red, CaHK, BC, Telemetry, Config, Receipt

    Args:
        L1 - a KPF L1 object 

    Returns:
        data_products in a L1 file
    """
    data_products = []
    if hasattr(L1, 'GREEN_SCI_FLUX1'):
        if L1['GREEN_SCI_FLUX1'].size > 1:
            data_products.append('Green')
    if hasattr(L1, 'RED_SCI_FLUX1'):
        if L1['RED_SCI_FLUX1'].size > 1:
            data_products.append('Red')
    if hasattr(L1, 'CA_HK_SCI'):
        if L1['CA_HK_SCI'].size > 1:
            data_products.append('CaHK')
    if hasattr(L1, 'BARY_CORR'):
        if L1['BARY_CORR'].size > 1:
            data_products.append('BC')
    if hasattr(L1, 'TELEMETRY'):
        if L1['TELEMETRY'].size > 1:
            data_products.append('Telemetry')
    if hasattr(L1, 'CONFIG'):
        if L1['CONFIG'].size > 1:
            data_products.append('Config')
    if hasattr(L1, 'RECEIPT'):
        if L1['RECEIPT'].size > 1:
            data_products.append('Receipt')
    return data_products


def get_data_products_L2(L2):
    """
    Returns a list of data products available in a L2 file, which are:
        Green CCF, Red CCF, Green CCF RW, Red CCF RW, RV, Activity, 
        Telemetry, Config, Receipt

    Args:
        L2 - a KPF L2 object 

    Returns:
        data_products in a L2 file
    """
    data_products = []
    if hasattr(L2, 'GREEN_CCF'):
        if L2['GREEN_CCF'].size > 1:
            data_products.append('Green')
    if hasattr(L2, 'GREEN_CCF_RW'):
        if L2['GREEN_CCF_RW'].size > 1:
            data_products.append('Green RW')
    if hasattr(L2, 'RED_CCF'):
        if L2['RED_CCF'].size > 1:
            data_products.append('Red')
    if hasattr(L2, 'RED_CCF_RW'):
        if L2['RED_CCF_RW'].size > 1:
            data_products.append('Red RW')
    if hasattr(L2, 'RV'):
        if L2['RV'].size > 1:
            data_products.append('RV')
    if hasattr(L2, 'ACTIVITY'):
        if L2['ACTIVITY'].size > 1:
            data_products.append('Activity')
    if hasattr(L2, 'TELEMETRY'):
        if L2['TELEMETRY'].size > 1:
            data_products.append('Telemetry')
    if hasattr(L2, 'CONFIG'):
        if L2['CONFIG'].size > 1:
            data_products.append('Config')
    if hasattr(L2, 'RECEIPT'):
        if L2['RECEIPT'].size > 1:
            data_products.append('Receipt')
    return data_products


def hasattr_with_wildcard(obj, pattern):
    regex = re.compile(pattern)
    return any(regex.match(attr) for attr in dir(obj))


def get_kpf_level(kpf_object):
    """
    Returns a string with the KPF level ('L0', '2D', 'L1', 'L2') corresponding 
    to the input KPF pubject

    Args:
        kpf_object - a KPF object 

    Returns:
        kpf_level ('L0', '2D', 'L1', 'L2')
    """
    
    # L2 if there's an extension that starts with 'CCF' or 'RV'
    if hasattr(kpf_object, 'RV'):
        return 'L2'
    if hasattr(kpf_object, 'CCF'):
        return 'L2'

    # elif L1 if there's an extension that includes 'WAVE'
    if hasattr_with_wildcard(kpf_object, r'.*WAVE.*'):
        return 'L1'

    # elif 2D if GREEN_CCD or RED_CCD has non-zero size
    if hasattr(kpf_object, 'GREEN_CCD'):
        if kpf_object['GREEN_CCD'].size > 1:
            return '2D'
    if hasattr(kpf_object, 'RED_CCD'):
        if kpf_object['RED_CCD'].size > 1:
            return '2D'

    # elif L0 if one of the standard extensions is present with non-zero size
    L0_attrs = ['GREEN_AMP1', 'RED_AMP1', 'CA_HK', 'EXPMETER_SCI', 'GUIDER_AVG', 'GUIDER_CUBE_ORIGINS', 'guider_avg', 'guider_cube_origins']
    for L0_attr in L0_attrs:
        if hasattr(kpf_object, L0_attr):
            if kpf_object[L0_attr].size:
                return 'L0'

    return None

def get_kpf_data(ObsID, data_level, data_dir='/data', return_kpf_object=True):
    """
    Returns the full path of a KPF object or the KPF object itself 
    with a specific data level

    Args:
        ObsID - e.g., 'KP.20230617.61836.73'
        data_level - 'L0', '2D', 'L1', or 'L2'
        data_dir - directory that contains L0/, 2D/, L1/, L2/
        return_kpf_object - if True, return kpf_object; if false, return path to object

    Returns:
        kpf_object 
           or
        full path of file, e.g., /data/2D/20230701/KP.20230701.49940.99_2D.fits
    """
    try:
        datecode = get_datecode(ObsID)
        if data_level == 'L0':
            fullpath = data_dir + '/L0/' + get_datecode(ObsID) + '/' + ObsID + '.fits'
            if return_kpf_object:
                return_object = KPF0.from_fits(fullpath)
            else:
                return_object = fullpath
        elif data_level == '2D':
            fullpath = data_dir + '/2D/' + get_datecode(ObsID) + '/' + ObsID + '_2D.fits'
            if return_kpf_object:
                return_object = KPF0.from_fits(fullpath)
            else:
                return_object = fullpath
        elif data_level == 'L1':
            fullpath = data_dir + '/L1/' + get_datecode(ObsID) + '/' + ObsID + '_L1.fits'
            if return_kpf_object:
                return_object = KPF1.from_fits(fullpath)
            else:
                return_object = fullpath
        elif data_level == 'L2':
            fullpath = data_dir + '/L2/' + get_datecode(ObsID) + '/' + ObsID + '_L2.fits'
            if return_kpf_object:
                return_object = KPF2.from_fits(fullpath)
            else:
                return_object = fullpath
    except:
        return None

    return return_object

