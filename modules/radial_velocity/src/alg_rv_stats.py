import numpy as np
from astropy.time import Time
import math

MJD_TO_JD = 2400000.5


class RadialVelocityStats:
    """ This module defines class ' RadialVelocityStats' and methods to do statistic analysis on radial velocity
    results. (this is currently for radial velocity development and testing only).

    Attributes:
         rv_result_set (list): A container storing radial velocity result from fits of level 1 data.
         total_set (int): Total elements in `rv_result_set`.
    """

    def __init__(self, obs_rv_results: list = None):
        self.rv_result_set = list() if obs_rv_results is None else obs_rv_results.copy()
        self.total_set = 0 if obs_rv_results is None else len(obs_rv_results)

    def get_collection(self):
        return self.rv_result_set, self.total_set

    def add_data(self, ccf_rv: float, obj_jd: float):
        self.rv_result_set.append({'jd': obj_jd, 'mean_rv': ccf_rv})
        self.total_set = len(self.rv_result_set)
        return self.rv_result_set, self.total_set

    def analyze_multiple_ccfs(self, ref_date=None):
        """ Statistic analysis on radial velocity numbers of multiple observation resulted by `RadialVelocityAlg`.

        Args:
            ref_date (str, optional): Reference time in the form Julian date format.  Defaults to None.

        Returns:
            dict: Analysis data.

        """
        obs_rvs, total_obs = self.get_collection()
        jd_list = np.array([obs_rv['jd'] for obs_rv in obs_rvs])
        if ref_date is None:
            ref_jd = self.get_start_day(jd_list)
        else:
            ref_jd = Time(ref_date, format='isot', scale='utc').jd
        rv_stats = dict()
        rv_stats['start_jd'] = ref_jd
        rv_stats['hour'] = (jd_list-ref_jd) * 24.0
        rv_stats['day'] = (jd_list-ref_jd)
        rv_stats['values'] = np.array([obs_rv['mean_rv'] for obs_rv in obs_rvs])
        rv_stats['mean'] = np.mean(rv_stats['values'])
        rv_stats['sigma'] = np.std(rv_stats['values'] - rv_stats['mean'])

        return rv_stats

    @staticmethod
    def get_start_day(jd_list: np.ndarray):
        min_jd = np.amin(jd_list)
        day_part = math.floor(min_jd - MJD_TO_JD)
        return MJD_TO_JD+day_part
