import logging
import sys

from modules.calibration_lookup.src.alg import GetCalibrations
from modules.calibration_lookup.src.calibration_lookup import CalibrationLookup


def test_calibration_lookup():
    datetime = '2024-02-28T09:16:52.826'

    default_config_path = 'modules/calibration_lookup/configs/default.cfg'
    
    cals = GetCalibrations(datetime, default_config_path)
    caldict = cals.lookup()
    print("Calibrations for datetime {}:\n{}".format(datetime, caldict))

if __name__ == '__main__':
    test_calibration_lookup()