import numpy as np
import os
import os.path
from barycorrpy import get_BC_vel
from astropy.utils import iers
import pandas as pd
from astropy import constants as const

LIGHT_SPEED_M = const.c.value  # light speed in m/s


class RVBaryCentricVelCorrection:
    """Barycentric velocity correction for radial velocity computation.

    This module defines class 'RVBaryCentricVelCorrection' and methods to calculate barycentric velocity correction
    and redshift for one single time point or a period of days.

    Attributes:
        zb_range (numpy.ndarray): An array containing minimum and maximum redshift value over a period of days.
        zb_list (numpy.ndarray): Array of redshift values over a period of days.

    The following constants define the properties included in the dict instance used for barycentric velocity correction
    and redshift calculation (:func:`~alg_barycentric_vel_corr.RVBaryCentricVelCorrection.get_zb_long()` and
    :func:`~alg_barycentric_vel_corr.RVBaryCentricVelCorrection.get_zb_from_bc_corr()`):
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
    RV = 'start_rv'
    """ star radial velocity (km/s) """

    def __init__(self):
        self.zb_range = None
        self.zb_list = None

    def get_zb_long(self, rv_config, jd, period, instrument=None, data_dir=None):
        """ Get minimum and maximum redshift over a period of time.

        Args:
            rv_config (dict): A dict instance storing key-value pairs for barycentric velocity correction calculation.
                The key list includes the name as the constants defined in class, i.e. `RA`, `DEC`, `PMRA`, `PMDEC`,
                `PX`, `RV`, `LAT`, `LON`, and `ALT`.
            jd (float): Starting day in Julian Date format.
            period (int): Period of days.
            instrument (str, optional): Instrument name. Defaults to None.
            data_dir (str, optional): Predefined directory. Defaults to None.
        Returns:
            numpy.ndarray: Minimum and maximum redshift values during the period starting from
            `jd`.  The first element in the array is the minimum and the second one is the maximum.
        """
        if self.zb_list is None:
            self.zb_list = self.get_zb_from_bc_corr(rv_config, instrument, jd, period, data_dir)
            self.zb_range = np.array([min(self.zb_list), max(self.zb_list)]) if np.size(self.zb_list) > 0 else None
        return self.zb_range

    @staticmethod
    def get_zb_from_bc_corr(rv_config, instrument, start_jd, days_period=None, data_dir=None):
        """ Find the redshift values for a period of days or a single day by using Barycentric velocity correction.

        Args:
            rv_config (dict):   A dict instance storing key-value pairs for barycentric velocity correction calculation.
                The key list includes names as the constants defined in class, i.e. `RA`, `DEC`, `PMRA`, `PMDEC`,
                `PX`, `RV`, `LAT`, `LON`, and `ALT`.
            instrument (str): instrument name.
            start_jd (float): Starting time point in Julian Date format.
            days_period (int, optional): Period of days for BC correction calculation. Defaults to None.
            data_dir(str, optional): Data directory. Defaults to None.

        Returns:
            numpy.ndarray:  An array of redshift values over the specified period or a single time point.

        """
        iers.Conf.iers_auto_url.set('ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all')
        zb_bc_corr = np.array([])
        zb_bc_file = None

        if days_period:  # look for pre-stored file first
            ins = '_'+instrument if instrument else ''
            if data_dir is not None:
                zb_bc_file = data_dir + 'radial_velocity_test/data/bc_corr' \
                          + str(start_jd) + '_' + str(days_period) + ins + '.csv'
            if zb_bc_file is not None and os.path.isfile(zb_bc_file):
                df = pd.read_csv(zb_bc_file)
                zb_bc_corr = np.reshape(df.values, (np.size(df.values), ))

        if np.size(zb_bc_corr) == 0:
            jds = np.arange(days_period, dtype=float) + start_jd if days_period else [start_jd]
            zb_list = list()
            for jd in jds:
                bc_obj = get_BC_vel(JDUTC=jd,
                                ra=rv_config[RVBaryCentricVelCorrection.RA],
                                dec=rv_config[RVBaryCentricVelCorrection.DEC],
                                pmra=rv_config[RVBaryCentricVelCorrection.PMRA],
                                pmdec=rv_config[RVBaryCentricVelCorrection.PMDEC],
                                px=rv_config[RVBaryCentricVelCorrection.PX],
                                lat=rv_config[RVBaryCentricVelCorrection.LAT],
                                longi=rv_config[RVBaryCentricVelCorrection.LON],
                                alt=rv_config[RVBaryCentricVelCorrection.ALT],
                                rv=rv_config[RVBaryCentricVelCorrection.RV])
                                # leap_update=False)
                bc = bc_obj[0][0]
                if bc:
                    # zb_list.append(bc)
                    zb_list.append(bc/LIGHT_SPEED_M)
                else:
                    zb_list.append(None)

            zb_bc_corr = np.array(zb_list)

            if zb_bc_file is not None:  # store to csv file
                df = pd.DataFrame(zb_bc_corr)
                df.to_csv(zb_bc_file, index=False, header=False)
        return zb_bc_corr
