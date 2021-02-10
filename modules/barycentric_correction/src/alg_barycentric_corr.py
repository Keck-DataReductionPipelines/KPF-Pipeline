import numpy as np
import os
import os.path
from barycorrpy import get_BC_vel
from astropy.utils import iers
import pandas as pd
from astropy import constants as const
from dotenv import load_dotenv
from datetime import date
from astropy.time import Time
from astropy.coordinates import Angle
import configparser
from modules.Utils.config_parser import ConfigHandler
load_dotenv()

LIGHT_SPEED_M = const.c.value  # light speed in m/s


class BarycentricCorrectionAlg:
    """Barycentric velocity correction.

    This module defines class 'BarycentricCorrectionAlg' and methods to calculate barycentric velocity correction
    and redshift for one single time point or a period of days.

    Args:
        obs_config (dict|configparser.ConfigParser): A dict instance or config context containing the key-value
                pairs related to the observation configuration for barycentric correction calculation.
        logger (logging.Logger): Instance of logging.Logger.


    Attributes:
        zb_range (numpy.ndarray): An array containing minimum and maximum redshift value over a period of days.
        zb_list (numpy.ndarray): Array of redshift values over a period of days.
        obs_config (dict): A dict instance storing key-value pairs related to observation configuration.
                The key list includes the name as the constants
                defined in class, i.e. `RA`, `DEC`, `PMRA`, `PMDEC`, `PX`, `RV`, `LAT`, `LON`, `ALT` and `SPEC`.
    Note:
        The following constants define the properties included in the dict instance used for barycentric velocity
        correction or redshift calculation, i.e.
        :func:`~barycentric_correction_alg.BarycentricCorrectionAlg.get_zb_long()`
        and :func:`~barycentric_correction_alg.BarycentricCorrectionAlg.get_zb_from_bc_corr()`
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
    """ star radial velocity estimation (km/s) """
    SPEC = "instrument"
    """ Observation instrument """

    def __init__(self, obs_config, config=None, logger=None):
        self.logger = logger

        self.obs_config = None
        # obs_config setting take the priority
        if obs_config is not None and isinstance(obs_config, dict):
            self.obs_config = obs_config
        elif config is not None and isinstance(config, configparser.ConfigParser):
            p_config = ConfigHandler(config, 'PARAM')
            self.instrument = p_config.get_config_value('instrument', '')
            ins = self.instrument.upper()
            bc_section = ConfigHandler(config, ins, p_config)  # handler containing section of instrument or 'PARAM'
            conf_def = [BarycentricCorrectionAlg.RA,
                        BarycentricCorrectionAlg.DEC,
                        BarycentricCorrectionAlg.PMRA,
                        BarycentricCorrectionAlg.PMDEC,
                        BarycentricCorrectionAlg.PX,
                        BarycentricCorrectionAlg.LAT,
                        BarycentricCorrectionAlg.LON,
                        BarycentricCorrectionAlg.ALT,
                        BarycentricCorrectionAlg.RV,
                        BarycentricCorrectionAlg.SPEC]
            self.obs_config = {}
            for c in conf_def:
                k_val = bc_section.get_config_value(c, '0.0')
                if c == BarycentricCorrectionAlg.RA:
                    self.obs_config[c] = Angle(k_val+"hours").deg
                elif c == self.DEC:
                    self.obs_config[c] = Angle(k_val+"degrees").deg
                elif c == BarycentricCorrectionAlg.SPEC:
                    self.obs_config[c] = self.instrument
                else:
                    self.obs_config[c] = float(k_val)

        self.zb_range = None
        self.zb_list = None

    def get_zb_long(self, jd, period, data_path=None, save_to_path=None):
        """ Get minimum and maximum redshift over a period of time.

        Args:
            jd (float): Starting day in Julian Date format.
            period (int): Period of days.
            data_path (str, optional): The path of the file containing predefined redshift values over a period of
                days.  It could be a path to a file or a directory. Defaults to None for a default path based on
                the setting of `jd`, `period`, instrument name and KPF data test directory.
            save_to_path ((str, optional): The path of the output file for the result. Defaults to None, meaning no
                output file created. It could be a path to a file or a directory.

        Returns:
            numpy.ndarray: Minimum and maximum redshift values during the period starting from
            `jd`.  The first element in the array is the minimum and the second one is the maximum.
        """

        if self.zb_list is None:
            self.zb_list = self.get_zb_list(jd, period, data_path, save_to_path)
            self.zb_range = np.array([min(self.zb_list), max(self.zb_list)]) if np.size(self.zb_list) > 0 else None
        return self.zb_range

    def get_zb_list(self, jd, period, data_path=None, save_to_path=None):
        """ Get redshift values over the period.

        Args:
            jd (float): Starting day in Julian Date format.
            period (int): Period of days.
            data_path (str, optional): The path of the file containing predefined redshift values. Defaults to None for
                a default path.
            save_to_path (str, optional): The path of the output file. Default to None for no output.

        Returns:
            list: redshift values over the period.

        """

        if self.zb_list is None:
            if jd is None:
                jd = Time(date.today().strftime("%Y-%m-%d")).jd - (period if period is not None else 1)
                if self.logger:
                    self.logger.info("start time is set " + str(period) + " days before now.")

            if self.logger:
                self.logger.info("start finding bc..." + " jd is " + str(jd) + " period is "+str(period))
            self.zb_list = self.get_zb_from_bc_corr(self.obs_config, jd, period, data_path, save_to_path)

        return self.zb_list

    @staticmethod
    def get_zb_from_bc_corr(obs_config, start_jd, days_period=None, data_path=None, save_to_path=None):
        """ Find the redshift values for a period of days or a single day by using Barycentric velocity correction or
        from an existing file and store the result to the specified file if there is.

        Args:
            obs_config (dict): A dict instance containing observation configuration.
            start_jd (float): Starting time point in Julian Date format.
            days_period (int, optional): Period of days for Barycentric correction calculation. Defaults to None.
                If the value is None or 1, it means the redshift of a single time point is computed, or a list of
                redshift values over a time period is either computed or accessed from a file as specified by
                `data_path`.
            data_path (str, optional): Path of the data file. Defaults to None for a default path per
                settings of `jd`, `period`, instrument and KPF data test directory.
            save_to_path (str, optional): Path of the output file. Defaults to None for no output.

        Returns:
            numpy.ndarray:  An array of redshift values over the specified period or a single time point.

        """

        # iers.Conf.iers_auto_url.set('ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all')
        # iers_url = "https://datacenter.iers.org/data/9/finals2000A.all"

        iers_url = iers.IERS_A_URL
        iers.Conf.iers_auto_url.set(iers_url)
        zb_bc_corr = np.array([])

        # if a period of time is assigned, check if a file storing the data already exists or not
        if days_period is not None and days_period > 1:
            instrument = obs_config.get(BarycentricCorrectionAlg.SPEC, '')
            zb_bc_file = BarycentricCorrectionAlg.find_existing_zb_file(instrument.lower(),
                                                                        start_jd, days_period, data_path)
            if zb_bc_file is not None and os.path.isfile(zb_bc_file):
                df = pd.read_csv(zb_bc_file, header=None)
                zb_bc_corr = np.reshape(df.values, (np.size(df.values), ))

        # compute redshift from barycentric correction and save the result to the file if there is
        if np.size(zb_bc_corr) == 0 and obs_config is not None:
            zb_list = BarycentricCorrectionAlg.get_bc_corr_period(obs_config, start_jd, days_period)
            zb_bc_corr = np.array(zb_list)

        # store to csv file
        if np.size(zb_bc_corr) != 0 and save_to_path is not None and os.path.exists(os.path.dirname(save_to_path)):
            df = pd.DataFrame(zb_bc_corr)
            df.to_csv(save_to_path, index=False, header=False)

        return zb_bc_corr

    @staticmethod
    def find_existing_zb_file(ins, start_jd, days_period, data_path=None):
        """ Compose the file path storing the redshift values from barycentric correction over a period of days.

        Args:
            ins (str): Observation instrument.
            start_jd (float): Start day in Julian Date format.
            days_period (int): Total days.
            data_path (str): Path of the data file. Defaults to None, meaning a default path based on
                the setting of `start_jd`, `days_period` and instrument under KPF data test directory is applied.

        Returns:
            str: Data path. The default path is like *<directory of KPFPIPE_TEST_DATA>/radial_velocity_test/data/
            bc_corr_<start_jd>_<days_period>_<instrument>.csv*
        """
        is_dir = True
        zb_bc_file = None
        if data_path is not None:
            is_dir = os.path.isdir(data_path)
        else:    # default directory containing file storing the baraycentric correction data
            data_path = os.getenv('KPFPIPE_TEST_DATA') + '/radial_velocity_test/data/'

        # if the output path is a directory,
        # the filename is like "bc_corr_<start_jd>_<days_period>_<ins>.csv"in default.
        if is_dir and data_path is not None:
            ins = '_' + ins if ins else ''
            zb_bc_file = (data_path if data_path.endswith("/") else (data_path + "/")) + \
                         'bc_corr' + str(start_jd) + '_' + str(days_period) + ins + '.csv'
        elif not is_dir and data_path is not None:
            zb_bc_file = data_path

        return zb_bc_file

    @staticmethod
    def get_bc_corr_period(obs_config, start_jd, days_period=None):
        """Compute redshift values from barycentric velocity correction over a period of days.

        Args:
            obs_config (dict): A dict instance containing observation configuration.
            start_jd (float): Starting time point in Julian Date format.
            days_period: Period of days for BC correction calculation. Defaults to None for one day.

        Returns:
            numpy.ndarray: redshift from barycentric velocity correction for a period of days.

        """
        jds = np.arange(days_period, dtype=float) + start_jd if days_period else [start_jd]
        # jds = np.arange(days_period+1, dtype=float) + start_jd
        zb_list = list()
        i = 0    # to remove
        for jd in jds:
            # iso_t = Time(jd, format='jd', scale='utc')
            bc = BarycentricCorrectionAlg.get_bc_corr(obs_config, jd)
            if bc:
                zb_list.append(bc / LIGHT_SPEED_M)
            else:
                zb_list.append(None)
        return zb_list

    @staticmethod
    def get_bc_corr(obs_config, jd):
        """Compute Barycentric correction on single time point.

        Args:
            obs_config (dict): A dict instance containing observation configuration.
            jd (float): Day in Julian Date format.

        Returns:
            float: Barycentric velocity correction number from get_BC_vel.

        """

        bc_obj = get_BC_vel(JDUTC=jd,
                            ra=obs_config[BarycentricCorrectionAlg.RA],
                            dec=obs_config[BarycentricCorrectionAlg.DEC],
                            pmra=obs_config[BarycentricCorrectionAlg.PMRA],
                            pmdec=obs_config[BarycentricCorrectionAlg.PMDEC],
                            px=obs_config[BarycentricCorrectionAlg.PX],
                            lat=obs_config[BarycentricCorrectionAlg.LAT],
                            longi=obs_config[BarycentricCorrectionAlg.LON],
                            alt=obs_config[BarycentricCorrectionAlg.ALT],
                            rv=obs_config[BarycentricCorrectionAlg.RV])
        return bc_obj[0][0]
