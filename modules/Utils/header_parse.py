class HeaderParse:

    """
    Description:
        This class contains functions to parse the headers of KPF fits files.

    Arguments:
        KPF File (L0, 2D, L1, or L2)

    Attributes:
        None so far
    """

    def __init__(self, header, logger=None):
        self.header = header
        self.name = '' # 'Name' of object; see get_name() function below
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
            header - header from a KPF L0/2D/L1/L2 file, e.g. L0['PRIMARY'].header

        Returns:
            the source/image type
        """
        
        # Note: the logic below is set so that as soon as the name is determined, it is
        #       returned.  This is so that partially complete headers don't cause this 
        #       to crash.
        if ('IMTYPE' in self.header) and ('EXPTIME' in self.header):
            if ((self.header['IMTYPE'] == 'Bias') or 
                (self.header['EXPTIME'] == 0)):
                self.name = 'Bias'
                return self.name
        if 'IMTYPE' in self.header:
            if self.header['IMTYPE'] == 'Dark':
                self.name = 'Dark' 
                return self.name
        if 'TARGNAME' in self.header:
            if ((self.header['TARGNAME'].lower() == 'sun') or 
                (self.header['TARGNAME'].lower() == 'socal')):
                self.name = 'Sun' # SoCal
                return self.name
        if ('OBJECT' in self.header) and ('FIUMODE' in self.header):
            if (self.header['FIUMODE'] == 'Observing'):
                self.name = self.header['OBJECT']
                return self.name # Stellar
        if 'FFFB' in self.header:
            if self.header['FFFB'].strip().lower() == 'yes':
                self.name = 'Wide Flat' # Flatfield Fiber (wide flats)
                return self.name
        if 'IMTYPE' in self.header:
            if self.header['IMTYPE'].lower() == 'flatlamp':
                if 'brdband' in self.header['OCTAGON'].lower():
                    self.name = 'Flat' # Flat through regular fibers
                    return self.name
        if 'IMTYPE' in self.header: # Emission Lamps
            if self.header['IMTYPE'].lower() == 'arclamp':
                if 'lfc' in self.header['OCTAGON'].lower():
                    self.name = 'LFC'
                    return self.name
                if 'etalon' in self.header['OCTAGON'].lower():
                    self.name = 'Etalon'
                    return self.name
                if 'th_' in self.header['OCTAGON'].lower():
                    self.name = 'ThAr'
                    return self.name
                if 'u_' in self.header['OCTAGON'].lower():
                    self.name = 'UNe'
                    return self.name
        self.name = ''
        return self.name
 