import numpy as np
from ctypes import c_double, c_int, CDLL, c_void_p, pointer
from modules.CLib.CCF import neidcode

package_directory = neidcode.__path__[0]
lib_path = '{}/CCF_3d_c.so'.format(package_directory)
c_ccf_lib = CDLL(lib_path)

ccf = c_ccf_lib.ccf
ccf_pixels = c_ccf_lib.ccf_pixels
ccf.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    c_double,
    c_double,
    c_int,
    c_int
    ]
ccf_pixels.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    c_double,
    c_double,
    c_int,
    c_int
    ]
ccf.restype = c_double

def calc_ccf(m_l, m_h, wav, spec, weight, sn, v_r, v_b):
    """
    Python wrapper that calls the C implementation of ccf calculation

    Args:
        m_l (np.array of np.float64): left edges of mask, length N
        m_h (np.array of np.float64): right edges of mask, length N
        wav (np.array of np.float64): the wavelengths of the spectrum 
            [Angstroms], length M
        spec (np.array of np.float64): flux values of the spectrum, length M
        weight (np.array of np.float64): mask weights, length N
        sn (np.array of np.float64): additional SNR scaling factor, length N
            (usually set to array of all 1s)
        v_r (float): the radial velocity at which to calculate the CCF [km/s]
        v_b (float): the barycentric velocity of the spectrum [km/s]

    Returns:
        np.float: the calculated CCF

    """
    n = len(weight)
    m = len(spec)

    ccf_value = ccf(
        m_l, m_h, wav, spec, weight, sn, c_double(v_r), c_double(v_b), 
        c_int(n), c_int(m)
    )
    return ccf_value


def calc_ccf_pixels(m_l, m_h, wav, spec, weight, sn, v_r, v_b):
    """
       Python wrapper that calls the C implementation of ccf calculation on pixels

       Args:
           m_l (np.array of np.float64): left edges of mask, length N
           m_h (np.array of np.float64): right edges of mask, length N
           wav (np.array of np.float64): the wavelengths of the spectrum
               [Angstroms], length M
           spec (np.array of np.float64): flux values of the spectrum, length M
           weight (np.array of np.float64): mask weights, length N
           sn (np.array of np.float64): additional SNR scaling factor, length N
               (usually set to array of all 1s)
           v_r (float): the radial velocity at which to calculate the CCF [km/s]
           v_b (float): the barycentric velocity of the spectrum [km/s]

       Returns:
           np.array: the calculated CCF of pixels

    """

    n = len(weight)
    m = len(spec)

    ccf_pixels.restype = np.ctypeslib.ndpointer(dtype=c_double, shape=(m-2, ))
    ccf_ps = ccf_pixels(
        m_l, m_h, wav, spec, weight, sn, c_double(v_r), c_double(v_b),
        c_int(n), c_int(m)
    )
    return ccf_ps


if __name__ == '__main__':
    """
    Tests barebones functionality of calc_ccf()
    """

    m_l = np.array([0., 1., 2.])
    m_h = np.array([0., 1., 2.])
    weight = np.array([0., 1., 2.])
    sn = np.array([0., 1., 2.])
    wav = np.array([1., 1.2, 1.3, 1.4])
    spec = np.array([1., 1.2, 1.3, 1.4])
    v_r = 4.
    v_b = 5.

    ccf = calc_ccf(m_l, m_h, wav, spec, weight, sn, v_r, v_b)