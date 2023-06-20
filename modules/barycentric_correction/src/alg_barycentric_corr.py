import numpy as np
import os
import os.path
import logging
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
from modules.Utils.alg_base import ModuleAlgBase
load_dotenv()

LIGHT_SPEED_M = const.c.value  # light speed in m/s


class BarycentricCorrectionAlg(ModuleAlgBase):
    """Barycentric velocity correction.

    This module defines class 'BarycentricCorrectionAlg' and methods to calculate barycentric velocity correction
    and redshift for one single time point or a period of days.

    Args:
        obs_config (dict|configparser.ConfigParser|None): A dict instance or config context containing the key-value
                pairs related to the observation configuration for barycentric correction calculation.
        logger (logging.Logger): Instance of logging.Logger passed from external application.
        logger_name (str, optional): Selection of logger by the specified logger name. Defaults to None.
                Either the defined logger_name or the class name determines the logging.Logger instance for organizing
                the loggers.

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
    EPOCH = 'epoch'
    """ epoch in julian date """
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
    STARNAME = 'starname'
    """ star name """

    def __init__(self, obs_config, config=None, logger=None, logger_name=None):
        ModuleAlgBase.__init__(self, logger_name or self.__class__.__name__, config, logger)

        self.obs_config = None
        # obs_config setting take the priority
        if obs_config is not None and isinstance(obs_config, dict):
            self.obs_config = obs_config
        elif config is not None and isinstance(config, configparser.ConfigParser):
            ins = self.config_param.get_config_value('instrument', '') if self.config_param is not None else ''
            self.instrument = ins.upper()
            if ins:
                # handler containing section of instrument or 'PARAM'
                bc_section = ConfigHandler(config, ins, self.config_param)
                conf_def = [BarycentricCorrectionAlg.RA,
                            BarycentricCorrectionAlg.DEC,
                            BarycentricCorrectionAlg.PMRA,
                            BarycentricCorrectionAlg.PMDEC,
                            BarycentricCorrectionAlg.PX,
                            BarycentricCorrectionAlg.EPOCH,
                            BarycentricCorrectionAlg.LAT,
                            BarycentricCorrectionAlg.LON,
                            BarycentricCorrectionAlg.ALT,
                            BarycentricCorrectionAlg.RV,
                            BarycentricCorrectionAlg.SPEC,
                            BarycentricCorrectionAlg.STARNAME]
                self.obs_config = {}
                for c in conf_def:
                    k_val = bc_section.get_config_value(c, '0.0')
                    if c == BarycentricCorrectionAlg.RA:
                        self.obs_config[c] = Angle(k_val+"hours").deg
                    elif c == self.DEC:
                        self.obs_config[c] = Angle(k_val+"degrees").deg
                    elif c == BarycentricCorrectionAlg.SPEC:
                        self.obs_config[c] = self.instrument
                    elif c == BarycentricCorrectionAlg.STARNAME:
                        self.obs_config[c] = k_val
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
            self.zb_range = np.array([min(self.zb_list), max(self.zb_list)])/LIGHT_SPEED_M \
                if np.size(self.zb_list) > 0 else None
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
                self.d_print("BarycentricCorrectionAlg: start time is set " + str(period) + " days before now.",
                             info=True)

            self.d_print('BarycentricCorrectionAlg: for redshift on jd: ', str(jd), ' and period:  ', str(period),
                         info=True)
            self.zb_list = self.get_zb_from_bc_corr(self.obs_config, jd, period, data_path, save_to_path)

        return self.zb_list

    @staticmethod
    def get_zb_from_bc_corr(obs_config, start_jd, days_period=None, data_path=None, save_to_path=None):
        """ Find the BC values for a period of days or a single day by calling Barycentric velocity correction function
         or from an existing file and store the result to the specified file if there is.

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
            numpy.ndarray:  An array of BC values (m/sec) over the specified period or a single time point.

        """

        # iers.Conf.iers_auto_url.set('ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all')
        # iers_url = "https://datacenter.iers.org/data/9/finals2000A.all"

        iers_url = iers.IERS_A_URL
        iers.Conf.iers_auto_url.set(iers_url)
        zb_bc_corr = np.array([])

        # if a period of time is assigned, check if a file storing the data already exists or not
        if days_period is not None and days_period > 1:
            instrument = obs_config.get(BarycentricCorrectionAlg.SPEC, '')
            target = obs_config.get(BarycentricCorrectionAlg.STARNAME, 'unknown')
            zb_bc_file = BarycentricCorrectionAlg.find_existing_zb_file(instrument.lower(),
                                                                start_jd, days_period, target.lower(), data_path)
            if zb_bc_file is not None and os.path.isfile(zb_bc_file):
                df = pd.read_csv(zb_bc_file, header=None)
                zb_bc_corr = np.reshape(df.values, (np.size(df.values), ))
            elif zb_bc_file is not None and save_to_path is None:
                save_to_path = zb_bc_file

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
    def find_existing_zb_file(ins, start_jd, days_period, target, data_path):
        """ Compose the file path storing the redshift values from barycentric correction over a period of days.

        Args:
            ins (str): Observation instrument.
            start_jd (float): Start day in Julian Date format.
            days_period (int): Total days.
            target (str): star name.
            data_path (str): Path of the data file. Defaults to None, meaning a default path based on
                the setting of `start_jd`, `days_period` and instrument under KPF data test directory is applied.

        Returns:
            str: Data path. The default path is like './+default_bc_file'
            bc_corr_<start_jd>_<days_period>_<instrument>.csv*
        """
        default_bc_file = 'bc_corr' + '_' + str(start_jd) + '_' + str(days_period) + '_' + ins + '_' + target + '.csv'

        if data_path is not None and data_path:
            if os.path.isdir(data_path):
                zb_bc_file = (data_path if data_path.endswith("/") else (data_path + "/")) + default_bc_file
            elif os.path.isfile(data_path):
                zb_bc_file = data_path
            else:  # a non-existing dir or non-existing file
                dirname = os.path.dirname(data_path)
                basename = os.path.basename(data_path)
                if not os.path.isdir(dirname):
                    os.makedirs(os.path.dirname(data_path), exist_ok=True)
                if basename:
                    zb_bc_file = data_path
                else:
                    zb_bc_file = dirname + '/' + default_bc_file
        else:
            zb_bc_file = './' + default_bc_file

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
        for jd in jds:
            # iso_t = Time(jd, format='jd', scale='utc')
            bc = BarycentricCorrectionAlg.get_bc_corr(obs_config, jd)
            if bc:
                # zb_list.append(bc / LIGHT_SPEED_M)
                zb_list.append(bc)
            else:
                zb_list.append(0.0)
        return zb_list

    @staticmethod
    def get_bc_corr(obs_config, jd):
        """Compute Barycentric correction on single time point.

        Args:
            obs_config (dict): A dict instance containing observation configuration.
            jd (float): Day in Julian Date format.

        Returns:
            float: Barycentric velocity [m/s] correction from barycorrpy.get_BC_vel.

        """
        star = obs_config[BarycentricCorrectionAlg.STARNAME].lower()
        if star == 'sun':
            # epoch, SolSystemTarget, predictive
            bc_obj = get_BC_vel(JDUTC=jd,
                                ra=None,
                                dec=None,
                                epoch=None,
                                pmra=None,
                                pmdec=None,
                                px=None,
                                lat=obs_config[BarycentricCorrectionAlg.LAT],
                                longi=obs_config[BarycentricCorrectionAlg.LON],
                                alt=obs_config[BarycentricCorrectionAlg.ALT],
                                SolSystemTarget='Sun',
                                predictive=True, zmeas=0,
                                rv=None,
                                #rv=obs_config[BarycentricCorrectionAlg.RV]
                                )
            return -bc_obj[0][0]
        else:
            bc_obj = get_BC_vel(JDUTC=jd,
                            ra=obs_config[BarycentricCorrectionAlg.RA],
                            dec=obs_config[BarycentricCorrectionAlg.DEC],
                            epoch=obs_config[BarycentricCorrectionAlg.EPOCH],
                            pmra=obs_config[BarycentricCorrectionAlg.PMRA],
                            pmdec=obs_config[BarycentricCorrectionAlg.PMDEC],
                            px=obs_config[BarycentricCorrectionAlg.PX],
                            lat=obs_config[BarycentricCorrectionAlg.LAT],
                            longi=obs_config[BarycentricCorrectionAlg.LON],
                            alt=obs_config[BarycentricCorrectionAlg.ALT],
                            rv=obs_config[BarycentricCorrectionAlg.RV])
            return bc_obj[0][0]
