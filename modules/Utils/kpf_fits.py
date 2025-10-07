import glob
import numpy as np
import subprocess
from astropy.io import fits
from kpfpipe.logger import start_logger

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

    @staticmethod
    def fitsheader_batch(file_paths, keys, extension=0):
        """
        Extract multiple header keywords from multiple FITS files using fitsheader subprocess.
        
        Args:
            file_paths (list): List of FITS file paths
            keys (list): List of header keywords to extract
            extension (int): FITS extension number (default: 0 for primary)
            
        Returns:
            dict: {file_path: {keyword: value, ...}, ...}
        """
        if not file_paths:
            return {}
            
        # Build fitsheader command
        cmd = ["fitsheader", "-e", str(extension)]
        for key in keys:
            cmd += ["-k", key]
        cmd += list(file_paths)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            parsed_result = FitsHeaders._parse_fitsheader_output(result.stdout, file_paths, keys)
            
            # If parsing failed or returned empty, fallback
            if not parsed_result:
                return FitsHeaders._fallback_header_extraction(file_paths, keys, extension)
            
            return parsed_result
            
        except subprocess.CalledProcessError:
            # Fallback to individual fits.getval calls if fitsheader fails
            return FitsHeaders._fallback_header_extraction(file_paths, keys, extension)
        except FileNotFoundError:
            # fitsheader command not found, fallback to astropy
            return FitsHeaders._fallback_header_extraction(file_paths, keys, extension)
    
    @staticmethod
    def _parse_fitsheader_output(output, file_paths, keys):
        """Parse fitsheader output into structured dictionary."""
        result = {}
        current_file = None
        
        for line in output.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if this line contains a filename in the format "# HDU X in filename:"
            filename_match = False
            if line.startswith('# HDU') and ' in ' in line and line.endswith(':'):
                # Extract filename from "# HDU 0 in /path/to/file.fits:"
                parts = line.split(' in ')
                if len(parts) >= 2:
                    filename_part = parts[-1].rstrip(':')  # Remove trailing colon
                    # Find matching file path
                    for path in file_paths:
                        if filename_part == path or path.endswith(filename_part) or filename_part.endswith(path):
                            current_file = path
                            result[current_file] = {}
                            filename_match = True
                            break
            
            if filename_match:
                continue
            
            # Parse keyword = value lines
            if '=' in line and current_file:
                try:
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value_comment = parts[1].strip()
                        
                        # Extract value (before any comment starting with /)
                        if '/' in value_comment:
                            value_str = value_comment.split('/', 1)[0].strip()
                        else:
                            value_str = value_comment.strip()
                        
                        # Remove quotes and convert to appropriate type
                        value_str = value_str.strip("'\"")
                        
                        # Try to convert to number if possible
                        try:
                            if '.' in value_str:
                                value = float(value_str)
                            else:
                                value = int(value_str)
                        except ValueError:
                            value = value_str
                        
                        result[current_file][key] = value
                except Exception:
                    continue
        
        return result
    
    @staticmethod
    def _fallback_header_extraction(file_paths, keys, extension):
        """Fallback to astropy fits.getval if fitsheader fails."""
        result = {}
        for file_path in file_paths:
            result[file_path] = {}
            for key in keys:
                try:
                    result[file_path][key] = fits.getval(file_path, key, ext=extension)
                except Exception:
                    result[file_path][key] = None
        return result


    @staticmethod
    def cleanup_primary_header(fname_input,fname_output,hdr=None):

        print("fname_input =",fname_input)
        print("fname_output =",fname_output)

        hdul_input = fits.open(fname_input)
        nhdul = len(hdul_input)

        hdu_list = []
        hdu_list.append(fits.PrimaryHDU(header=hdr))

        for i in range(1,nhdul):

            type_hdul = type(hdul_input[i])

            print("i,type_hdul =",i,type_hdul)

            hdu_list.append(hdul_input[i])

        hdu = fits.HDUList(hdu_list)
        hdu.writeto(fname_output,overwrite=True,checksum=True)


    def __init__(self, search_path, header_keywords, header_values, logger=None):
        self.n_header_keywords = np.size(header_keywords)
        if not isinstance(header_keywords, list):
            header_keywords = [header_keywords]
        if not isinstance(header_values, list):
            header_values = [header_values]
        self.header_keywords = header_keywords
        self.header_values = header_values
        self.input_fits_files = glob.glob(search_path)
        if logger:
            self.logger = logger
            self.logger.info('FitsHeaders class constructor: self.input_fits_files = {}'.format(self.input_fits_files))
        else:
            print("Starting logger...")
            self.logger = start_logger(self.__class__.__name__, 'configs/framework_logger.cfg')
            print('---->FitsHeaders class constructor: self.input_fits_files = {}'.format(self.input_fits_files))

        n_input_fits_files = len(self.input_fits_files)

        self.logger.info('FitsHeaders constructor: n_input_fits_files = {}'.format(n_input_fits_files))

    def match_headers_string_lower(self):

        """
        Return list of files that each has lowercase string matches
        to all input FITS kewords/values of interest.
        """

        if not self.input_fits_files:
            return []

        # Batch extract all required header keywords using fitsheader subprocess
        headers_data = self.fitsheader_batch(self.input_fits_files, self.header_keywords)

        matched_fits_files = []
        for fits_file in self.input_fits_files:

            match_count = 0
            for i in range(self.n_header_keywords):

                input_value = (self.header_values[i]).lower()

                try:
                    # Get header value from batch extraction
                    file_headers = headers_data.get(fits_file, {})
                    val = file_headers.get(self.header_keywords[i])
                    
                    if val is None:
                        raise KeyError(self.header_keywords[i])
                    
                    fits_value = str(val).lower().strip()
                    if (fits_value == input_value):
                        match_count += 1

                except KeyError as err:

                    if self.logger:
                        self.logger.info('KeyError: {} ({}); skipping...'.format(err,fits_file))
                    else:
                        print('---->KeyError: {} ({}); skipping...'.format(err,fits_file))

                except TypeError as err:

                    if self.logger:
                        self.logger.info('TypeError: {} ({}); skipping...'.format(err,fits_file))
                    else:
                        print('---->TypeError: {} ({}); skipping...'.format(err,fits_file))

            if match_count == self.n_header_keywords:
                matched_fits_files.append(fits_file)

        if self.logger:
             self.logger.info('FitsHeaders.match_headers_string_lower(): matched_fits_files = {}'.\
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

        if not self.input_fits_files:
            return []

        # Batch extract all required header keywords using fitsheader subprocess
        headers_data = self.fitsheader_batch(self.input_fits_files, self.header_keywords)

        matched_fits_files = []
        for fits_file in self.input_fits_files:

            match_count = 0
            for i in range(self.n_header_keywords):

                input_value = float(self.header_values[i])

                try:
                    # Get header value from batch extraction
                    file_headers = headers_data.get(fits_file, {})
                    val = file_headers.get(self.header_keywords[i])
                    
                    if val is None:
                        raise KeyError(self.header_keywords[i])

                    if self.logger:
                        self.logger.info('FitsHeaders.match_headers_float_le(): file,i,keyword,input_value,header_value = {},{},{},{},{}'.\
                            format(fits_file,i,self.header_keywords[i],input_value,val))
                    else:
                        print('---->FitsHeaders.match_headers_float_le(): file,i,keyword,input_value,header_value = {},{},{},{},{}'.\
                            format(fits_file,i,self.header_keywords[i],input_value,val))

                    try:
                        fits_value = float(val)

                        if (fits_value <= input_value): match_count += 1

                    except ValueError as err2:

                        if self.logger:
                            self.logger.info('ValueError: {}; skipping...'.format(err2))
                        else:
                            print('---->ValueError: {}; skipping...'.format(err2))

                except Exception as err:

                    if self.logger is not None:
                        self.logger.info('Exception caught: {} for FITS file {}; skipping...'.format(err,fits_file))
                    else:
                        print('---->Exception caught: {} for FITS file {}; skipping...'.format(err,fits_file))

            if match_count == self.n_header_keywords:
                matched_fits_files.append(fits_file)

        return matched_fits_files

    def get_good_flats(self):

        """
        Return list of flat files defined by IMTYPE=‘flatlamp’, but exclude
        those that either don't have  SCI-OBJ == CAL-OBJ and SKY-OBJ == CAL-OBJ
        or those with SCI-OBJ == "" or SCI-OBJ == "None".
        """

        matched_fits_files = self.match_headers_string_lower()
        
        if not matched_fits_files:
            return []

        # Batch extract all required header keywords using fitsheader subprocess
        required_keys = ['SCI-OBJ', 'CAL-OBJ', 'SKY-OBJ', 'ELAPSED']
        headers_data = self.fitsheader_batch(matched_fits_files, required_keys)

        filtered_matched_fits_files = []
        
        for fits_file in matched_fits_files:

            flag = 'remove'

            try:
                # Get header values from batch extraction
                file_headers = headers_data.get(fits_file, {})
                val1 = file_headers.get('SCI-OBJ')
                val2 = file_headers.get('CAL-OBJ') 
                val3 = file_headers.get('SKY-OBJ')
                val4 = file_headers.get('ELAPSED')        # Require EXPTIME <= 11.0 seconds to avoid saturation.

                # Handle None values (missing headers)
                if val1 is None or val2 is None or val3 is None or val4 is None:
                    raise KeyError("Missing required header keywords")

                # Apply filtering logic (handle string conversion safely)
                val1_str = str(val1) if val1 is not None else ''
                if ((val1 == val2) and (val2 == val3) and (val1 != '') and (val1_str.lower() != 'none') and (val4 <= 11.0)):
                    flag = 'keep'
                    filtered_matched_fits_files.append(fits_file)

                if self.logger:
                    self.logger.info('flag,val1,val2,val3,val1.lower(),val4 = {},{},{},{},[{}],{}'.\
                        format(flag,val1,val2,val3,val1_str.lower(),val4))
                else:
                    print('---->flag,val1,val2,val3,val1.lower(),val4 = {},{},{},{},[{}],{}'.\
                        format(flag,val1,val2,val3,val1_str.lower(),val4))

            except Exception as err:

                if self.logger is not None:
                    self.logger.info('Exception caught: {} for FITS file {}; skipping...'.format(err,fits_file))
                else:
                    print('---->Exception caught: {} for FITS file {}; skipping...'.format(err,fits_file))


        return filtered_matched_fits_files

    def get_good_darks(self,exptime_minimum):

        """
        Return list of dark files defined by IMTYPE=‘dark’, but include only those
        with EXPTIME greater than or equal to the  specified minimum exposure time.
        """
        matched_fits_files = self.match_headers_string_lower()

        if not matched_fits_files:
            return [], []

        # Batch extract all required header keywords using fitsheader subprocess
        required_keys = ['ELAPSED', 'OBJECT']
        headers_data = self.fitsheader_batch(matched_fits_files, required_keys)

        filtered_matched_fits_files = []
        all_dark_objects = []
        for fits_file in matched_fits_files:

            flag = 'remove'

            try:
                # Get header values from batch extraction
                file_headers = headers_data.get(fits_file, {})
                val4 = file_headers.get('ELAPSED')
                obj = file_headers.get('OBJECT')

                # Handle None values (missing headers)
                if val4 is None:
                    raise KeyError("Missing ELAPSED header keyword")

                val4 = float(val4)

                if (val4 >= exptime_minimum):
                    flag = 'keep'
                    filtered_matched_fits_files.append(fits_file)
                    if obj is not None and obj not in all_dark_objects:
                        all_dark_objects.append(obj)

                if self.logger:
                    self.logger.info('flag,val4 = {},{}'.\
                        format(flag,val4))
                else:
                    print('---->flag,val4 = {},{}'.\
                        format(flag,val4))

            except Exception as err:

                if self.logger is not None:
                    self.logger.info('Exception caught: {} for FITS file {}; skipping...'.format(err,fits_file))
                else:
                    print('---->Exception caught: {} for FITS file {}; skipping...'.format(err,fits_file))

        return filtered_matched_fits_files,all_dark_objects

    def get_good_arclamps(self):

        """
        Return list of arclamp files defined by IMTYPE=‘arclamp’, and a list of the various OBJECT keyword settings.
        """

        if not self.input_fits_files:
            return [], []

        # Batch extract all required header keywords using fitsheader subprocess
        # Include both the class header keywords and OBJECT
        all_keywords = self.header_keywords + ['OBJECT']
        headers_data = self.fitsheader_batch(self.input_fits_files, all_keywords)

        matched_fits_files = []
        all_arclamp_objects = []
        for fits_file in self.input_fits_files:

            match_count = 0
            for i in range(self.n_header_keywords):

                input_value = (self.header_values[i]).lower()

                try:
                    # Get header value from batch extraction
                    file_headers = headers_data.get(fits_file, {})
                    val = file_headers.get(self.header_keywords[i])
                    
                    if val is None:
                        raise KeyError(self.header_keywords[i])
                    
                    fits_value = str(val).lower().strip()
                    if (fits_value == input_value):
                        match_count += 1

                except KeyError as err:

                    if self.logger:
                        self.logger.info('KeyError: {} ({}); skipping...'.format(err,fits_file))
                    else:
                        print('---->KeyError: {} ({}); skipping...'.format(err,fits_file))

                except TypeError as err:

                    if self.logger:
                        self.logger.info('TypeError: {} ({}); skipping...'.format(err,fits_file))
                    else:
                        print('---->TypeError: {} ({}); skipping...'.format(err,fits_file))

            if match_count == self.n_header_keywords:

                try:
                    # Get OBJECT value from batch extraction
                    file_headers = headers_data.get(fits_file, {})
                    obj = file_headers.get('OBJECT')
                    
                    if obj is not None:
                        matched_fits_files.append(fits_file)
                        if obj not in all_arclamp_objects:
                            all_arclamp_objects.append(obj)
                    else:
                        raise KeyError('OBJECT')

                except KeyError as err:

                    if self.logger:
                        self.logger.info('KeyError: {} ({}); skipping...'.format(err,fits_file))
                    else:
                        print('---->KeyError: {} ({}); skipping...'.format(err,fits_file))

        return matched_fits_files,all_arclamp_objects

    def get_good_biases(self):

        """
        Return list of bias files defined by IMTYPE=‘bias’ and OBJECT='autocal-bias', but include only those
        with EXPTIME less than or equal to the specified maximum exposure time.
        """

        exptime_maximum = 0.1      # ELAPSED can be slightly greater than zero!

        matched_fits_files = self.match_headers_string_lower()

        if not matched_fits_files:
            return [], []

        # Batch extract all required header keywords using fitsheader subprocess
        required_keys = ['SCISEL', 'SKYSEL', 'FFSHTR', 'SCRAMSHT', 'SIMCALSH', 'ELAPSED', 'OBJECT']
        headers_data = self.fitsheader_batch(matched_fits_files, required_keys)

        filtered_matched_fits_files = []
        all_bias_objects = []
        for fits_file in matched_fits_files:

            flag = 'remove'

            try:
                # Get header values from batch extraction
                file_headers = headers_data.get(fits_file, {})
                
                # These keyword must indicate closed shutters:
                # SCISEL = 'closed ' / Science Select shutter at exp. midpoint
                # SKYSEL = 'closed ' / Sky Select Shutter at exp. midpoint
                # FFSHTR = 'closed ' / Flat field fiber shutter at exp. midpoint
                # SCRAMSHT= 'closed ' / Scrambler shutter at exp. midpoint
                # SIMCALSH= 'closed ' / Simult Cal shutter at exp. midpoint

                SCISEL = file_headers.get('SCISEL')
                SKYSEL = file_headers.get('SKYSEL')
                FFSHTR = file_headers.get('FFSHTR')
                SCRAMSHT = file_headers.get('SCRAMSHT')
                SIMCALSH = file_headers.get('SIMCALSH')
                ELAPSED = file_headers.get('ELAPSED')
                obj = file_headers.get('OBJECT')

                # Handle None values (missing headers)
                if any(val is None for val in [SCISEL, SKYSEL, FFSHTR, SCRAMSHT, SIMCALSH, ELAPSED]):
                    raise KeyError("Missing required header keywords")

                if 'closed' in str(SCISEL) and \
                   'closed' in str(SKYSEL) and \
                   'closed' in str(FFSHTR) and \
                   'closed' in str(SCRAMSHT) and \
                   'closed' in str(SIMCALSH):

                    if (ELAPSED <= exptime_maximum):
                        flag = 'keep'
                        filtered_matched_fits_files.append(fits_file)
                        if obj is not None and obj not in all_bias_objects:
                            all_bias_objects.append(obj)

                if self.logger:
                    self.logger.info('flag,ELAPSED = {},{}'.\
                        format(flag,ELAPSED))
                else:
                    print('---->flag,ELAPSED = {},{}'.\
                        format(flag,ELAPSED))

            except Exception as err:

                if self.logger is not None:
                    self.logger.info('Exception caught: {} for FITS file {}; skipping...'.format(err,fits_file))
                else:
                    print('---->Exception caught: {} for FITS file {}; skipping...'.format(err,fits_file))

        return filtered_matched_fits_files,all_bias_objects
