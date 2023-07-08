class HeaderParse:

    """
    Description:
        This class contains functions to parse the headers of KPF fits files.

    Arguments:
        Header for a KPF File (L0, 2D, L1, or L2), e.g. L0['PRIMARY'].header

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
        
        if ('IMTYPE' in self.header) and ('EXPTIME' in self.header):
            if ((self.header['IMTYPE'] == 'Bias') or 
                (self.header['EXPTIME'] == 0)):
                self.name = 'Bias'
        elif 'IMTYPE' in self.header:
            if self.header['IMTYPE'] == 'Dark':
                self.name = 'Dark' 
        elif 'TARGNAME' in self.header:
            if ((self.header['TARGNAME'].lower() == 'sun') or 
                (self.header['TARGNAME'].lower() == 'socal')):
                self.name = 'Sun' # SoCal
        elif ('OBJECT' in self.header) and ('FIUMODE' in self.header):
            if (self.header['FIUMODE'] == 'Observing'):
                self.name = self.header['OBJECT']
        elif 'FFFB' in self.header:
            if self.header['FFFB'].strip().lower() == 'yes':
                self.name = 'Wide Flat' # Flatfield Fiber (wide flats)
        elif 'IMTYPE' in self.header:
            if self.header['IMTYPE'].lower() == 'flatlamp':
                if 'brdband' in self.header['OCTAGON'].lower():
                    self.name = 'Flat' # Flat through regular fibers
        elif 'IMTYPE' in self.header: # Emission Lamps
            if self.header['IMTYPE'].lower() == 'arclamp':
                if 'lfc' in self.header['OCTAGON'].lower():
                    self.name = 'LFC'
                if 'etalon' in self.header['OCTAGON'].lower():
                    self.name = 'Etalon'
                if 'th_' in self.header['OCTAGON'].lower():
                    self.name = 'ThAr'
                if 'u_' in self.header['OCTAGON'].lower():
                    self.name = 'UNe'
        else:
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

