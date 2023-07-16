class KPFParse:

    """
    Description:
        This class contains functions to parse the headers of KPF fits files.

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

    Arguments:
        header - Header for a KPF File (L0, 2D, L1, or L2), e.g. L0['PRIMARY'].header

    Attributes:
        None so far
    """

    def __init__(self, header, logger=None):
        self.header = header
        self.name = '' # 'Name' of object; see get_name() function below
        self.ObsID = '' # e.g., 'KP.20230708.04519.63'
        if logger:
            self.logger = logger
            self.logger.debug('HeaderParse class constructor')
        else:
            self.logger = None
            print('---->HeaderParse class constructor')


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
                if (('EXPTIME' in self.header) and 
                    ((self.header['IMTYPE'] == 'Bias') or (self.header['EXPTIME'] == 0))):
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
        2D - a KPF 2D object 

    Returns:
        data_products in a 2D file
    """
    data_products = []
    if 'GREEN_AMP1' in L0:
        if L0['GREEN_AMP1'].size > 1:
            data_products.append('Green')
    if 'RED_AMP1' in L0:
        if L0['RED_AMP1'].size > 1:
            data_products.append('Red')
    if 'CA_HK' in L0:
        if L0['CA_HK'].size > 1:
            data_products.append('CaHK')
    if 'EXPMETER_SCI' in L0:
        if L0['EXPMETER_SCI'].size > 1:
            data_products.append('ExpMeter')
    if ('GUIDER_AVG' in L0) or ('guider_avg' in L0):
        if (L0['GUIDER_AVG'].size > 1) or (L0['guider_avg'].size > 1):
            data_products.append('Guider')
    if 'TELEMETRY' in L0:
        if L0['TELEMETRY'].size > 1:
            data_products.append('Telemetry')
    if 'SOCAL PYRHELIOMETER' in L0:
        if L0['SOCAL PYRHELIOMETER'].size > 1:
            data_products.append('Config')
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
    if 'GREEN_CCD' in D2:
        if D2['GREEN_CCD'].size > 1:
            data_products.append('Green')
    if 'Red_CCD' in D2:
        if D2['Red_CCD'].size > 1:
            data_products.append('Red')
    if 'CA_HK' in D2:
        if D2['CA_HK'].size > 1:
            data_products.append('CaHK')
    if 'EXPMETER_SCI' in D2:
        if D2['EXPMETER_SCI'].size > 1:
            data_products.append('ExpMeter')
    if 'GUIDER_AVG' in D2:
        if D2['GUIDER_AVG'].size > 1:
            data_products.append('Guider')
    if 'TELEMETRY' in D2:
        if D2['TELEMETRY'].size > 1:
            data_products.append('Telemetry')
    if 'CONFIG' in D2:
        if D2['CONFIG'].size > 1:
            data_products.append('Config')
    if 'RECEIPT' in D2:
        if D2['RECEIPT'].size > 1:
            data_products.append('Receipt')
    if 'SOCAL PYRHELIOMETER' in D2:
        if D2['SOCAL PYRHELIOMETER'].size > 1:
            data_products.append('Pyrheliometer')
    # add support for SOLAR_IRRADIANCE later
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
    if 'GREEN_SCI_FLUX1' in L1:
        if L1['GREEN_SCI_FLUX1'].size > 1:
            data_products.append('Green')
    if 'RED_SCI_FLUX1' in L1:
        if L1['RED_SCI_FLUX1'].size > 1:
            data_products.append('Red')
    if 'CA_HK_SCI' in L1:
        if L1['CA_HK_SCI'].size > 1:
            data_products.append('CaHK')
    if 'BARY_CORR' in L1:
        if L1['BARY_CORR'].size > 1:
            data_products.append('BC')
    if 'TELEMETRY' in L1:
        if L1['TELEMETRY'].size > 1:
            data_products.append('Telemetry')
    if 'CONFIG' in L1:
        if L1['CONFIG'].size > 1:
            data_products.append('Config')
    if 'RECEIPT' in L1:
        if L1['RECEIPT'].size > 1:
            data_products.append('Receipt')
    return data_products

