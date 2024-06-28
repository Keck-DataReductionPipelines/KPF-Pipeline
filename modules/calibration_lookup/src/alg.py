
from datetime import datetime
import pandas as pd
import os
import re
import psycopg2

import database.modules.utils.kpf_db as db
from keckdrpframework.models.arguments import Arguments
from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.logger import start_logger

def query_database(obsdate, cal_type_pairs, cal_file_levels, log):
    """
    Returns [query_db_nearest_master_files_exit_code,nearest_master_files_list].

    """

    log.debug('----> obsdate = {}'.format(obsdate))
    log.debug('----> cal_file_levels = {}'.format(cal_file_levels))
    log.debug('----> cal_type_pairs = {}'.format(cal_type_pairs))

    query_db_nearest_master_files_exit_code = 0


    contentbitmask_list = [3]        # Mask values for just GREEN and RED.

    nearest_master_files_list = []

    dbh = db.KPFDB()             # Open database connection.

    for contentbitmask in contentbitmask_list:
        for cal_file_level,cal_type_pair in zip(cal_file_levels,cal_type_pairs):
            log.debug('cal_file_level = {}'.format(cal_file_level))
            log.debug('cal_type_pair = {}'.format(cal_type_pair))

            dbh.get_nearest_master_file(obsdate,cal_file_level,contentbitmask,cal_type_pair)
            log.debug('database-query exit_code = {}'.format(dbh.exit_code))
            log.debug('Calibration database-query filename = {}'.format(dbh.filename))
            if dbh.exit_code == 0:
                cal_file_record = [dbh.cId, dbh.db_level, dbh.db_cal_type, dbh.db_object, contentbitmask, dbh.infobits, dbh.filename]
                nearest_master_files_list.append(cal_file_record)
            else:
                log.debug('*** Error: Calibration file cannot be queried from database; returning...')
                query_db_nearest_master_files_exit_code = 1


    dbh.close()      # Close database connection.

    exit_list = [query_db_nearest_master_files_exit_code,nearest_master_files_list]

    return exit_list


def query_wls(datetime, cal_type_pairs, max_cal_file_age, log):
    """
    Returns [exitcode_for_before_query,[before_master_file_record],exitcode_for_after_query,[after_master_file_record]].

    """

    # hard code some parameters for WLS lookup
    #
    #
    # cal_type_pairs = [['WLS','autocal-lfc-all'], ['WLS', 'autocal-thar-all']]
    #
    # Only the first element of input argument cal_type_pairs list is pertinent below.
    #

    results_before = None
    results_after = None

    cal_file_level = 1  # can assume WLS is in L1 format
    contentbitmask = 3
    object_before = cal_type_pairs[0][1] + "-eve"
    object_after = object_before.replace('eve', 'morn')

    cal_type_pair_before = ["WLS",object_before]
    cal_type_pair_after = ["WLS",object_after]

    log.debug('----> datetime = {}'.format(datetime))
    log.debug('----> cal_file_level = {}'.format(cal_file_level))
    log.debug('----> contentbitmask = {}'.format(contentbitmask))
    log.debug('----> cal_type_pair_before = {}'.format(cal_type_pair_before))
    log.debug('----> cal_type_pair_after = {}'.format(cal_type_pair_after))
    log.debug('----> max_cal_file_age = {}'.format(max_cal_file_age))


    dbh = db.KPFDB()             # Open database connection.

    dbh.get_nearest_master_file_before(datetime,cal_file_level,contentbitmask,cal_type_pair_before,max_cal_file_age)
    log.debug('database-query exit_code = {}'.format(dbh.exit_code))
    log.debug('Calibration database-query filename before = {}'.format(dbh.filename))
    exit_code_before = dbh.exit_code
    if exit_code_before == 0:
        results_before = [dbh.cId, dbh.db_level, dbh.db_cal_type, dbh.db_object, contentbitmask, dbh.infobits, dbh.filename]
    else:
        log.debug('*** Error: Calibration file before cannot be queried from database; returning...')
        exit_code_before = 1

    dbh.get_nearest_master_file_after(datetime,cal_file_level,contentbitmask,cal_type_pair_after,max_cal_file_age)
    log.debug('database-query exit_code = {}'.format(dbh.exit_code))
    log.debug('Calibration database-query filename after = {}'.format(dbh.filename))
    exit_code_after = dbh.exit_code
    if exit_code_after == 0:
        results_after = [dbh.cId, dbh.db_level, dbh.db_cal_type, dbh.db_object, contentbitmask, dbh.infobits, dbh.filename]
    else:
        log.debug('*** Error: Calibration file after cannot be queried from database; returning...')
        exit_code_after = 1

    dbh.close()      # Close database connection.

    exit_list = [exit_code_before,results_before,exit_code_after,results_after]

    return exit_list


def extract_from_db_results(results, cal_type):
    if cal_type.lower() == 'wls':
        results_list = [None, None]
        if results[0] == 0:
            results_list[0] = results[1][6]
        if results[2] == 0:
            results_list[1] = results[3][6]
        return results_list
    elif results[0] == 1:
        return ''
    elif cal_type.lower() == 'wls':
        return [results[1][6], results[3][6]]
    else:
        cal_list = results[1]
        for cal in cal_list:
            if cal_type.lower() == cal[2].lower():
                return cal[6]

        cals = []
        for i in enumerate(results):
            cal_list = results

class GetCalibrations:
    """This utility looks up the associated calibrations for a given datetime and
       returns a dictionary with all calibration types.

    """
    def __init__(self, datetime, default_config_path, logger=None):

        # Initialize DB class
        # self.db_lookup = QueryDBNearestMasterFilesFramework(self.action, self.context)

        #Input arguments
        self.datetime = datetime   # ISO datetime string
        self.config = ConfigClass(default_config_path)
        if logger == None:
            self.log = start_logger('GetCalibrations', default_config_path)
        else:
            self.log = logger

        self.caldate_files = eval(self.config['PARAM']['date_files'])
        self.lookup_map = eval(self.config['PARAM']['lookup_map'])
        self.db_cal_types = eval(self.config['PARAM']['db_cal_types'])
        self.db_cal_file_levels = eval(self.config['PARAM']['db_cal_file_levels'])
        self.wls_cal_types = eval(self.config['PARAM']['wls_cal_types'])
        self.max_age = eval(self.config['PARAM']['max_cal_age'])
        self.defaults = eval(self.config['PARAM']['defaults'])

    def lookup(self):
        dt = datetime.strptime(self.datetime, "%Y-%m-%dT%H:%M:%S.%f")
        date_str = datetime.strftime(dt, "%Y%m%d")

        output_cals = {}
        db_results = None
        for cal,lookup in self.lookup_map.items():
            if lookup == 'file':
                filename = self.caldate_files[cal]
                df = pd.read_csv(filename, header=0, skipinitialspace=True)
                for i, row in df.iterrows():
                    start = datetime.strptime(row['UT_start_date'], "%Y-%m-%d %H:%M:%S")
                    end = datetime.strptime(row['UT_end_date'], "%Y-%m-%d %H:%M:%S")
                    if start <= dt < end:
                        try:
                            output_cals[cal] = eval(row['CALPATH'])
                        except SyntaxError:
                            output_cals[cal] = row['CALPATH']
            elif lookup == 'database' and db_results == None:
                db_results = query_database(date_str, self.db_cal_types, self.db_cal_file_levels, self.log)
                if db_results[0] == 0:
                    output_cals[cal] = extract_from_db_results(db_results, cal)
                else:
                    output_cals[cal] = self.defaults[cal]
            elif lookup == 'database' and db_results != None:
                if db_results[0] == 0:
                    output_cals[cal] = extract_from_db_results(db_results, cal)
                else:
                    output_cals[cal] = self.defaults[cal]
            elif lookup == 'wls':
                wls_results = query_wls(self.datetime, self.wls_cal_types, self.max_age, self.log)
                if len(wls_results) > 1 and (wls_results[0] == 0 or wls_results[2] == 0):
                    wls_files = extract_from_db_results(wls_results, cal)
                    if wls_files[0] == None:
                        wls_files[0] = wls_files[1]
                    if wls_files[1] == None:
                        wls_files[1] = wls_files[0]
                    output_cals[cal] = wls_files
                else:
                    output_cals[cal] = self.defaults[cal]

        return output_cals

