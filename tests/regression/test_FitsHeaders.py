from modules.Utils.kpf_fits import FitsHeaders

def test_match_headers_float_le():

    """
    Find all fits files with matching EXPTIME less than or equal zero.
    """

    fh = FitsHeaders(files_in_dir, 'EXPTIME', '0.0')
    input_files = fh.match_headers_float_le()

    print('Output from match_headers_float_le...')

    i = 1
    for input_file in (input_files):
        print(i,input_file)
        i += 1

def test_match_headers_string_lower():

    """
    Find all fits files with lowercase IMTYPE value matching "none".
    """

    fh2 = FitsHeaders(files_in_dir, 'IMTYPE', 'NONE')
    input_files2 = fh2.match_headers_string_lower()

    print('Output from match_headers_string_lower...')

    i = 1
    for input_file in (input_files2):
        print(i,input_file)
        i += 1


if __name__ == '__main__':

    fits_files_path = '/data'
    files_in_dir = fits_files_path+'/KP*.fits'
    print('files_in_dir = ', files_in_dir)

    test_match_headers_float_le()
    test_match_headers_string_lower()
