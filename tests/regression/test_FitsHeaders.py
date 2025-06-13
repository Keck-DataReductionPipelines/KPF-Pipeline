from modules.Utils.kpf_fits import FitsHeaders
from kpfpipe.logger import start_logger

fits_files_path = '/data/L0/kpf_fits'
files_in_dir = fits_files_path + '/KP*.fits'

print("Starting logger from top of python script...")
logger = start_logger('MyLogger','configs/framework_logger.cfg')
print('---->Top of python script: fits_files_path = {}'.format(fits_files_path))

def test_get_good_arclamps():

    imtype_keywords = ['IMTYPE','OBJECT']
    imtype_values_str = ['Arclamp','autocal-thar-sky-eve']

    fh = FitsHeaders(files_in_dir,imtype_keywords,imtype_values_str,logger)
    #all_arclamp_files = fh.get_good_arclamps()        # Actually, this is not used.
    all_arclamp_files = fh.match_headers_string_lower()

    for arclamp_file_path in all_arclamp_files:
        logger.info('arclamp_file_path = {}'.format(arclamp_file_path))

    assert fits_files_path + '/KP.20250307.06307.29.fits' in all_arclamp_files


def test_get_good_flats():

    imtype_keywords = ['IMTYPE','OBJECT']
    imtype_values_str = ['Flatlamp','autocal-flat-all']

    fh = FitsHeaders(files_in_dir,imtype_keywords,imtype_values_str,logger)
    #all_flat_files = fh.get_good_flats()        # Actually, this is not used.
    all_flat_files = fh.match_headers_string_lower()

    for flat_file_path in all_flat_files:
        logger.info('flat_file_path = {}'.format(flat_file_path))

    assert fits_files_path + '/KP.20250307.02828.87.fits' in all_flat_files


def test_get_good_darks():

    imtype_keywords = 'IMTYPE'
    imtype_values_str = 'Dark'

    exptime_minimum = 300.0

    fh = FitsHeaders(files_in_dir,imtype_keywords,imtype_values_str,logger)
    all_dark_files,all_dark_objects = fh.get_good_darks(exptime_minimum)

    for dark_file_path in all_dark_files:
        logger.info('dark_file_path = {}'.format(dark_file_path))

    for dark_object in all_dark_objects:
        logger.info('dark_object = {}'.format(dark_object))

    assert fits_files_path + '/KP.20250307.65367.06.fits' in all_dark_files


def test_get_good_biases():

    imtype_keywords = 'IMTYPE'
    imtype_values_str = 'Bias'

    # Filter bias files with IMTYPE=‘Bias’ and EXPTIME = 0.0 for now.  Later in this class, exclude
    # those FITS-image extensions that don't match the input object specification with OBJECT.

    fh = FitsHeaders(files_in_dir,imtype_keywords,imtype_values_str,logger)
    all_bias_files,all_bias_objects = fh.get_good_biases()

    for bias_file_path in all_bias_files:
        logger.info('bias_file_path = {}'.format(bias_file_path))

    for bias_object in all_bias_objects:
        logger.info('bias_object = {}'.format(bias_object))

    assert fits_files_path + '/KP.20250307.03636.94.fits' in all_bias_files
    assert fits_files_path + '/KP.20250307.03886.63.fits' in all_bias_files
    assert fits_files_path + '/KP.20250307.03686.83.fits' in all_bias_files
    assert fits_files_path + '/KP.20250307.03736.85.fits' in all_bias_files
    assert fits_files_path + '/KP.20250307.03786.80.fits' in all_bias_files
    assert fits_files_path + '/KP.20250307.03836.71.fits' in all_bias_files
    assert fits_files_path + '/KP.20250307.03936.68.fits' in all_bias_files
    assert fits_files_path + '/KP.20250307.03986.64.fits' in all_bias_files


def test_match_headers_float_le():

    """
    Find all fits files with matching EXPTIME less than or equal zero.
    """

    fh = FitsHeaders(files_in_dir, 'ELAPSED', '10.0',logger=logger)
    input_files = fh.match_headers_float_le()

    print('Output from match_headers_float_le...')

    i = 1
    for input_file in (input_files):
        print(i,input_file)
        i += 1

    assert fits_files_path + '/KP.20250307.03636.94.fits' in input_files

def test_match_headers_string_lower():

    """
    Find all fits files with lowercase IMTYPE value matching "none".
    """

    fh2 = FitsHeaders(files_in_dir, 'IMTYPE', 'OBJECT',logger=logger)
    input_files2 = fh2.match_headers_string_lower()

    print('Output from match_headers_string_lower...')

    i = 1
    for input_file in (input_files2):
        print(i,input_file)
        i += 1

    assert fits_files_path + '/KP.20240101.24368.88_L1.fits' in input_files2

if __name__ == '__main__':

    print('files_in_dir = ', files_in_dir)

    test_match_headers_float_le()
    test_match_headers_string_lower()
    test_get_good_biases()
    test_get_good_darks()
    test_get_good_flats()
    test_get_good_arclamps()
