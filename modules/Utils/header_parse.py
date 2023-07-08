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
                return 'Bias'
        if 'IMTYPE' in self.header:
            if self.header['IMTYPE'] == 'Dark':
                return 'Dark' 
        if 'TARGNAME' in self.header:
            if ((self.header['TARGNAME'].lower() == 'sun') or 
                (self.header['TARGNAME'].lower() == 'socal')):
                return 'Sun' # SoCal
        if ('OBJECT' in self.header) and ('FIUMODE' in self.header):
            if (self.header['FIUMODE'] == 'Observing'):
                name = self.header['OBJECT']
                return name # Stellar
        if 'FFFB' in self.header:
            if self.header['FFFB'].strip().lower() == 'yes':
                return 'Wide Flat' # Flatfield Fiber (wide flats)
        if 'IMTYPE' in self.header:
            if self.header['IMTYPE'].lower() == 'flatlamp':
                if 'brdband' in self.header['OCTAGON'].lower():
                    return 'Flat' # Flat through regular fibers
        if 'IMTYPE' in self.header: # Emission Lamps
            if self.header['IMTYPE'].lower() == 'arclamp':
                if 'lfc' in self.header['OCTAGON'].lower():
                    return 'LFC'
                if 'etalon' in self.header['OCTAGON'].lower():
                    return 'Etalon'
                if 'th_' in self.header['OCTAGON'].lower():
                    return 'ThAr'
                if 'u_' in self.header['OCTAGON'].lower():
                    return 'UNe'
        return ''
 