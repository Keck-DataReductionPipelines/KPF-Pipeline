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

