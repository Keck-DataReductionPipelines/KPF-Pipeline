import numpy as np
import os
import os.path
from barycorrpy import get_BC_vel
from astropy.utils import iers
import pandas as pd

LIGHT_SPEED_M = 299792458.  # light speed in m/s


class RVBaryCentricVelCorrection:
    """Barycentric velocity correction for radial velocity computation.

    This module defines class 'RVBaryCentricVelCorrection' and methods to do barycentric velocity correction for
    one single time point or a period of days.

    Attributes:
        zb_range (numpy.ndarray): An array containing minimum and maximum values from barycentric velocity correction
            over a period of days.
        bc_corr_list (numpy.ndarray): Array of Barycentric velocity correction values over a period of days.

    The following constants define the properties included in the dict instance used for barycentric velocity correction
    calculation (:func:`~alg_barycentric_vel_corr.RVBaryCentricVelCorrection.get_zb_long()` and
    :func:`~alg_barycentric_vel_corr.RVBaryCentricVelCorrection.get_bc_corr_rv()`):
    """

    RA = 'ra'
    """ RA J2000 of the target (hours) """
    DEC = 'dec'
    """ DEC J2000 of the target (degrees) """
    PMRA = 'pmra'
    """proper motion in RA (mas/yr) """
    PMDEC = 'pmdec'
    """ proper motion in DEC (mas/yr) """
    PX = 'parallax'
    """ parallax of the target (mas) """
    LAT = 'obslat'
    """ latitude of the observatory, North (+ve) and South (-ve) (degree) """
    LON = 'obslon'
    """ longitude of the observatory, East (+ve) and West (-ve) (degree) """
    ALT = 'obsalt'
    """ altitude of the observatory (meter) """
    RV = 'star_rv'
    """ star radial velocity (km/s) """

    def __init__(self):
        self.zb_range = None
        self.bc_corr_list = None

    def get_zb_long(self, rv_config, jd, period, instrument=None, data_dir=None):
        """ Barycenter velocity correction minimum and maximum over a period of days.

        Args:
            rv_config (dict): A dict instance storing key-value pairs for barycentric velocity correction calculation.
                The key list includes the name as the constants defined in class, i.e. `RA`, `DEC`, `PMRA`, `PMDEC`,
                `PX`, `RV`, `LAT`, `LON`, and `ALT`.
            jd (float): Starting day in Julian Date format.
            period (int): Period of days.
            instrument (str, optional): Instrument name. Defaults to None.
            data_dir (str, optional): Predefined directory. Defaults to None.
        Returns:
            numpy.ndarray: Minimum and maximum of the barycentric velocity correction during the period starting from
            `jd`.  The first element in the array is the minimum and the second one is the maximum.
        """
        if self.bc_corr_list is None:
            bc_list = self.get_bc_corr_rv(rv_config, instrument, jd, period, data_dir)  # get_bc_corr_rv(2458591.5, 380)
            self.bc_corr_list = bc_list
            self.zb_range = np.array([min(bc_list), max(bc_list)])
        return self.zb_range

    @staticmethod
    def get_bc_corr_rv(rv_config, instrument, start_jd, days_period=None, data_dir=None):
        """ BC correction for a period of days or a single day in Julian Date format.

        Args:
            rv_config (dict):   A dict instance storing key-value pairs for barycentric velocity correction calculation.
                The key list includes names as the constants defined in class, i.e. `RA`, `DEC`, `PMRA`, `PMDEC`,
                `PX`, `RV`, `LAT`, `LON`, and `ALT`.
            instrument (str): instrument name.
            start_jd (float): Starting time point in Julian Date format.
            days_period (int, optional): Period of days for BC correction calculation. Defaults to None.
            data_dir(str, optional): Data directory. Defaults to None.

        Returns:
            numpy.ndarray:  An array of BC correction values over the specified period or a single time point.

        """
        iers.Conf.iers_auto_url.set('ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all')
        bc_corr = np.array([])
        bc_file = None

        if days_period:  # look for pre-stored file first
            ins = '_'+instrument if instrument else ''
            if data_dir is not None:
                bc_file = data_dir + 'radial_velocity_test/data/bc_corr' \
                          + str(start_jd) + '_' + str(days_period) + ins + '.csv'
            if bc_file is not None and os.path.isfile(bc_file):
                df = pd.read_csv(bc_file)
                bc_corr = np.reshape(df.values, (np.size(df.values), ))

        if np.size(bc_corr) == 0:
            jds = np.arange(days_period, dtype=float) + start_jd if days_period else [start_jd]
            bc_corr_list = list()
            for jd in jds:
                bc = get_BC_vel(JDUTC=jd,
                                ra=rv_config[RVBaryCentricVelCorrection.RA],
                                dec=rv_config[RVBaryCentricVelCorrection.DEC],
                                pmra=rv_config[RVBaryCentricVelCorrection.PMRA],
                                pmdec=rv_config[RVBaryCentricVelCorrection.PMDEC],
                                px=rv_config[RVBaryCentricVelCorrection.PX],
                                lat=rv_config[RVBaryCentricVelCorrection.LAT],
                                longi=rv_config[RVBaryCentricVelCorrection.LON],
                                alt=rv_config[RVBaryCentricVelCorrection.ALT],
                                rv=rv_config[RVBaryCentricVelCorrection.RV],
                                leap_update=True)[0][0]/LIGHT_SPEED_M
                bc_corr_list.append(bc)
            bc_corr = np.array(bc_corr_list)

            if bc_file is not None:  # store to csv file
                df = pd.DataFrame(bc_corr)
                df.to_csv(bc_file, index=False, header=False)
        return bc_corr
