from astropy.io import fits
from datetime import datetime

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


    def get_name(self):
        """
        Returns the name of the source in a spectrum.  For stellar observations, this 
        is the star's name (e.g. '185144' for HD 185144 = sigma Draconis).  
        For calibration spectra, this is the lamp name (ThAr, UNe, LFC, etalon) or 
        bias/dark.  Flats using KPF's regular fibers are distinguished from wide flats.

        Args:
            None

        Returns:
            the source/image type
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
                        self.name = self.header['OBJECT']
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

#datetime_format = "%Y-%m-%dT%H:%M:%S.%f"
#dt1 = datetime.strptime(time1, datetime_format)
#dt2 = datetime.strptime(time2, datetime_format)

# Compute the difference
#difference = dt2 - dt1
    
# Return the total seconds rounded to 0.1-second precision
#round(difference.total_seconds(), 1)
    
#GRDATE  = '2023-09-10T13:29:09.804475' / FITS file write time Kwd green DATE    
#GRDATE-B= '2023-09-10T13:26:56.365516' / Shutter-open time Kwd green DATE-BEG   
#GRDATE-E= '2023-09-10T13:28:56.398782' / Shutter-close time Kwd green DATE-END  
#RDDATE  = '2023-09-10T13:29:09.896907' / FITS file write time Kwd red DATE      
#RDDATE-B= '2023-09-10T13:26:56.365516' / Shutter-open time Kwd red DATE-BEG     
#RDDATE-E= '2023-09-10T13:28:56.398782' / Shutter-close time Kwd red DATE-END    


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


def get_data_products_L0(L0):
    """
    Returns a list of data products available in an L0 file, which are:
    Green, Red, CaHK, ExpMeter, Guider, Telemetry, Config

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
        if (L0['GUIDER_AVG'].size > 1):
            data_products.append('Guider')
#    if hasattr(L0, 'guider_avg'):
#        if (L0['guider_avg'].size > 1):
#            data_products.append('Guider')
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
#    if hasattr(D2, 'guider_avg'):
#        if (D2['guider_avg'].size > 1):
#            data_products.append('Guider')
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
        L1 - a KPF L2 object 

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
