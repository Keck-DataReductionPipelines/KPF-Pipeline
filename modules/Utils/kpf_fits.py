import glob
import numpy as np
from astropy.io import fits

class FitsHeaders:

    """
    Description:
        This class contains functions to retrieve and act upon information
        in the headers of FITS files located within a given data directory.
        Typically, the functions will act upon all FITS files under a given date.

    Arguments:
        search_path (str, which can include file glob): Directory path of FITS files.
        header_keywords (str or list of str): FITS keyword(s) of interest.
        header_values (str or list of str): Value(s) of FITS keyword(s), in list order.

    Attributes:
        header_keywords (str or list of str): FITS keyword(s) of interest.
        header_values (str or list of str): Value(s) of FITS keyword(s), in list order.
        n_header_keywords (int): Number of FITS keyword(s) of interest.
        input_fits_files (list of str): Individual FITS filename(s) that will be searched.
    """

    def __init__(self, search_path, header_keywords, header_values, logger=None):
        self.n_header_keywords = np.size(header_keywords)
        if self.n_header_keywords == 1:
            header_keywords = [header_keywords]
            header_values = [header_values]
        self.header_keywords = header_keywords
        self.header_values = header_values
        self.input_fits_files = glob.glob(search_path)
        if logger:
            self.logger = logger
            self.logger.debug('FitsHeaders class constructor: self.input_fits_files = {}'.format(self.input_fits_files))
        else:
            self.logger = None
            print('---->FitsHeaders class constructor: self.input_fits_files = {}'.format(self.input_fits_files))

    def match_headers_string_lower(self):

        """
        Return list of files that each has lowercase string matches
        to all input FITS kewords/values of interest.
        """

        matched_fits_files = []
        for fits_file in self.input_fits_files:

            match_count = 0
            for i in range(self.n_header_keywords):

                input_value = (self.header_values[i]).lower()

                try:

                    val = fits.getval(fits_file, self.header_keywords[i])
                    fits_value = (val).lower()
                    if (fits_value == input_value):
                        match_count += 1

                except KeyError as err:

                    if self.logger:
                        self.logger.debug('KeyError: {} ({}); skipping...'.format(err,fits_file))
                    else:
                        print('---->KeyError: {} ({}); skipping...'.format(err,fits_file))

                except TypeError as err:

                    if self.logger:
                        self.logger.debug('TypeError: {} ({}); skipping...'.format(err,fits_file))
                    else:
                        print('---->TypeError: {} ({}); skipping...'.format(err,fits_file))

            if match_count == self.n_header_keywords:
                matched_fits_files.append(fits_file)

        if self.logger:
             self.logger.debug('FitsHeaders.match_headers_string_lower(): matched_fits_files = {}'.\
                   format(matched_fits_files))
        else:
            print('---->FitsHeaders.match_headers_string_lower(): matched_fits_files = {}'.\
                format(matched_fits_files))

        return matched_fits_files

    def match_headers_float_le(self):

        """
        Return list of files that each has floating-point
        values that are less than or equal to
        all input FITS kewords/values of interest.
        """

        matched_fits_files = []
        for fits_file in self.input_fits_files:

            match_count = 0
            for i in range(self.n_header_keywords):

                input_value = float(self.header_values[i])

                try:

                    val = fits.getval(fits_file, self.header_keywords[i])

                    if self.logger:
                        self.logger.debug('FitsHeaders.match_headers_float_le(): file,i,keyword,input_value,header_value = {},{},{},{},{}'.\
                            format(fits_file,i,self.header_keywords[i],input_value,val))
                    else:
                        print('---->FitsHeaders.match_headers_float_le(): file,i,keyword,input_value,header_value = {},{},{},{},{}'.\
                            format(fits_file,i,self.header_keywords[i],input_value,val))

                    try:
                        fits_value = float(val)

                        if (fits_value <= input_value): match_count += 1

                    except ValueError as err2:

                        if self.logger:
                            self.logger.debug('ValueError: {}; skipping...'.format(err2))
                        else:
                            print('---->ValueError: {}; skipping...'.format(err2))

                except KeyError as err:

                    if self.logger:
                        self.logger.debug('KeyError: {}; skipping...'.format(err))
                    else:
                        print('---->KeyError: {}; skipping...'.format(err))

                except TypeError as err:

                    if self.logger:
                        self.logger.debug('TypeError: {}; skipping...'.format(err))
                    else:
                        print('---->TypeError: {}; skipping...'.format(err))

            if match_count == self.n_header_keywords:
                matched_fits_files.append(fits_file)

        if self.logger:
             self.logger.debug('FitsHeaders.match_headers_float_le(): matched_fits_files = {}'.\
                   format(matched_fits_files))
        else:
            print('---->FitsHeaders.match_headers_float_le(): matched_fits_files = {}'.\
                format(matched_fits_files))

        return matched_fits_files

    def get_good_flats(self):

        """
        Return list of flat files defined by IMTYPE=‘flatlamp’, but exclude
        those that either don't have  SCI-OBJ == CAL-OBJ and SKY-OBJ == CALOBJ
        or those with SCI-OBJ == "" or SCI-OBJ == "None".
        """

        matched_fits_files = self.match_headers_string_lower()

        filtered_matched_fits_files = []
        for fits_file in matched_fits_files:

            flag = 'remove'

            try:

                val1 = fits.getval(fits_file, 'SCI-OBJ')
                val2 = fits.getval(fits_file, 'CAL-OBJ')
                val3 = fits.getval(fits_file, 'SKY-OBJ')
                val4 = fits.getval(fits_file, 'EXPTIME')        # Require EXPTIME <= 2.0 seconds to avoid saturation.

                if ((val1 == val2) and (val2 == val3) and (val1 != '') and (val1.lower() != 'none') and (val4 <= 2.0)):
                    flag = 'keep'
                    filtered_matched_fits_files.append(fits_file)

                if self.logger:
                    self.logger.debug('flag,val1,val2,val3,val1.lower(),val4 = {},{},{},{},[{}],{}'.\
                        format(flag,val1,val2,val3,val1.lower(),val4))
                else:
                    print('---->flag,val1,val2,val3,val1.lower(),val4 = {},{},{},{},[{}],{}'.\
                        format(flag,val1,val2,val3,val1.lower(),val4))

            except KeyError as err:

                if self.logger:
                    self.logger.debug('KeyError: {}; removing {} from list...'.format(err,fits_file))
                else:
                    print('---->KeyError: {}; removing {} from list...'.format(err,fits_file))

            except TypeError as err:

                if self.logger:
                    self.logger.debug('TypeError: {}; removing {} from list...'.format(err,fits_file))
                else:
                    print('---->TypeError: {}; removing {} from list...'.format(err,fits_file))

        if self.logger:
             self.logger.debug('FitsHeaders.get_good_flats(): filtered_matched_fits_files = {}'.\
                   format(filtered_matched_fits_files))
        else:
            print('---->FitsHeaders.get_good_flats(): filtered_matched_fits_files = {}'.\
                format(filtered_matched_fits_files))

        return filtered_matched_fits_files

    def get_good_darks(self,exptime_minimum):

        """
        Return list of dark files defined by IMTYPE=‘dark’, but include only those
        with EXPTIME greater than or equal to the  specified minimum exposure time.
        """
        matched_fits_files = self.match_headers_string_lower()

        filtered_matched_fits_files = []
        for fits_file in matched_fits_files:

            flag = 'remove'

            try:

                val4 = float(fits.getval(fits_file, 'EXPTIME'))

                if (val4 >= exptime_minimum):
                    flag = 'keep'
                    filtered_matched_fits_files.append(fits_file)

                if self.logger:
                    self.logger.debug('flag,val4 = {},{}'.\
                        format(flag,val4))
                else:
                    print('---->flag,val4 = {},{}'.\
                        format(flag,val4))

            except KeyError as err:

                if self.logger:
                    self.logger.debug('KeyError: {}; removing {} from list...'.format(err,fits_file))
                else:
                    print('---->KeyError: {}; removing {} from list...'.format(err,fits_file))

            except TypeError as err:

                if self.logger:
                    self.logger.debug('TypeError: {}; removing {} from list...'.format(err,fits_file))
                else:
                    print('---->TypeError: {}; removing {} from list...'.format(err,fits_file))

        if self.logger:
             self.logger.debug('FitsHeaders.get_good_darks(): filtered_matched_fits_files = {}'.\
                   format(filtered_matched_fits_files))
        else:
            print('---->FitsHeaders.get_good_darks(): filtered_matched_fits_files = {}'.\
                format(filtered_matched_fits_files))

        return filtered_matched_fits_files

    def get_good_arclamps(self):

        """
        Return list of arclamp files defined by IMTYPE=‘arclamp’, and a list of the various OBJECT keyword settings.
        """

        matched_fits_files = []
        all_arclamp_objects = []
        for fits_file in self.input_fits_files:

            hdul = fits.open(fits_file)
            match_count = 0
            for i in range(self.n_header_keywords):

                input_value = (self.header_values[i]).lower()

                try:

                    val = hdul[0].header[self.header_keywords[i]]
                    fits_value = (val).lower()
                    if (fits_value == input_value):
                        match_count += 1

                except KeyError as err:

                    if self.logger:
                        self.logger.debug('KeyError: {} ({}); skipping...'.format(err,fits_file))
                    else:
                        print('---->KeyError: {} ({}); skipping...'.format(err,fits_file))

                except TypeError as err:

                    if self.logger:
                        self.logger.debug('TypeError: {} ({}); skipping...'.format(err,fits_file))
                    else:
                        print('---->TypeError: {} ({}); skipping...'.format(err,fits_file))

            if match_count == self.n_header_keywords:
                matched_fits_files.append(fits_file)
                obj = hdul[0].header['OBJECT']
                if obj not in all_arclamp_objects:
                    all_arclamp_objects.append(obj)
                
            hdul.close()

        if self.logger:
             self.logger.debug('FitsHeaders.get_good_arclamps(): matched_fits_files = {}'.\
                   format(matched_fits_files))
        else:
            print('---->FitsHeaders.get_good_arclamps(): matched_fits_files = {}'.\
                format(matched_fits_files))

        return matched_fits_files,all_arclamp_objects

    def get_good_biases(self):

        """
        Return list of bias files defined by IMTYPE=‘bias’ and OBJECT='autocal-bias', but include only those
        with EXPTIME less than or equal to the specified maximum exposure time.
        """

        exptime_maximum = 0.0

        matched_fits_files = self.match_headers_string_lower()

        filtered_matched_fits_files = []
        for fits_file in matched_fits_files:

            flag = 'remove'

            try:

                val4 = float(fits.getval(fits_file, 'EXPTIME'))

                if (val4 <= exptime_maximum):
                    flag = 'keep'
                    filtered_matched_fits_files.append(fits_file)

                if self.logger:
                    self.logger.debug('flag,val4 = {},{}'.\
                        format(flag,val4))
                else:
                    print('---->flag,val4 = {},{}'.\
                        format(flag,val4))

            except KeyError as err:

                if self.logger:
                    self.logger.debug('KeyError: {}; removing {} from list...'.format(err,fits_file))
                else:
                    print('---->KeyError: {}; removing {} from list...'.format(err,fits_file))

            except TypeError as err:

                if self.logger:
                    self.logger.debug('TypeError: {}; removing {} from list...'.format(err,fits_file))
                else:
                    print('---->TypeError: {}; removing {} from list...'.format(err,fits_file))

        if self.logger:
             self.logger.debug('FitsHeaders.get_good_biases(): filtered_matched_fits_files = {}'.\
                   format(filtered_matched_fits_files))
        else:
            print('---->FitsHeaders.get_good_biases(): filtered_matched_fits_files = {}'.\
                format(filtered_matched_fits_files))

        return filtered_matched_fits_files
