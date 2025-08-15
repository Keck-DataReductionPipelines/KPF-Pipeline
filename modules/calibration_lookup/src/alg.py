
import sys
if sys.version_info >= (3, 9):
    from importlib import resources
else:
    import importlib_resources as resources

import pandas as pd
from datetime import datetime
import time

from database.modules.utils.kpf_db import KPFDB
from keckdrpframework.models.arguments import Arguments
from kpfpipe.config.pipeline_config import ConfigClass
from kpfpipe.logger import start_logger
from astropy.io.fits import getheader

class GetCalibrations:
    """This utility looks up the associated calibrations for a given datetime and
       returns a dictionary with all calibration types.

    """
    def __init__(self, datetime, default_config_path, use_db=True, logger=None):

        """
        use_db (boolean) - to disable db access, set to False (e.g., when looking up file-based keywords only)
        """
        
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
        self.use_db = use_db

    def lookup(self, subset=None):
        start_time = time.time()
        dt = datetime.strptime(self.datetime, "%Y-%m-%dT%H:%M:%S.%f")
        date_str = datetime.strftime(dt, "%Y%m%d")

        output_cals = {}
        db_results = None
        if subset == None:
            subset = self.lookup_map.keys()
        
        # Track timing for each lookup type
        file_lookup_time = 0
        database_lookup_time = 0
        wls_lookup_time = 0
        
        # Collect all database calibration requests for batch processing
        db_requests = []
        cal_type_mapping = {}  # Map from database names back to original names
        
        for cal,lookup in self.lookup_map.items():
            if cal not in subset:
                continue
                
            if lookup == 'database':
                # Find the corresponding database calibration type
                for lvl, cal_type in zip(self.db_cal_file_levels, self.db_cal_types):
                    if cal_type[0].lower() == cal:
                        # Handle special case for traceflat
                        cal_type_lookup = cal_type.copy()
                        if cal_type[0] == 'traceflat':
                            cal_type_lookup[0] = 'Flat'
                            cal_type_mapping['flat'] = 'traceflat'  # Map 'flat' result back to 'traceflat'
                        else:
                            cal_type_mapping[cal_type[0].lower()] = cal_type[0].lower()
                        
                        db_requests.append((lvl, cal_type_lookup))
                        break
        
        # Determine if we need a DB connection at all
        needs_db = any(self.lookup_map.get(cal) in ('database', 'wls', 'etalon') for cal in subset)
        db = None

        try:
            # Execute single batch query for all database calibrations
            if db_requests and self.use_db:
                if db is None:
                    db = KPFDB(logger=self.log)
                db_start = time.time()
                try:
                    # Use batch query for better performance
                    batch_results = db.get_nearest_master_batch(self.datetime, db_requests)
                    
                    # Process batch results
                    for lvl, cal_type in db_requests:
                        cal_type_name = cal_type[0].lower()
                        original_name = cal_type_mapping[cal_type_name]
                        
                        if cal_type_name in batch_results:
                            db_results = batch_results[cal_type_name]
                            if db_results[0] == 0:
                                if isinstance(cal_type[1], list):
                                    # Handle multi-results (like ordertrace)
                                    if original_name not in output_cals:
                                        output_cals[original_name] = []
                                    output_cals[original_name].append(db_results[1])
                                else:
                                    output_cals[original_name] = db_results[1]
                            else:
                                self.log.warning(f"Database lookup failed for {cal_type[0]} (exit code {db_results[0]}), using default")
                                output_cals[original_name] = self.defaults[original_name]
                        else:
                            output_cals[original_name] = self.defaults[original_name]
                    
                    database_lookup_time += time.time() - db_start
                    self.log.info(f"Batch database lookup completed in {time.time() - db_start:.3f}s for {len(db_requests)} calibration types")
                    
                except Exception as e:
                    self.log.warning(f"Exception during batch database lookup: {e}, falling back to individual lookups")
                    # Fallback to individual lookups if batch fails
                    for lvl, cal_type in db_requests:
                        original_name = cal_type_mapping[cal_type[0].lower()]
                        try:
                            db_results = db.get_nearest_master(self.datetime, lvl, cal_type)
                            if db_results[0] == 0:
                                if isinstance(cal_type[1], list):
                                    if original_name not in output_cals:
                                        output_cals[original_name] = []
                                    output_cals[original_name].append(db_results[1])
                                else:
                                    output_cals[original_name] = db_results[1]
                            else:
                                output_cals[original_name] = self.defaults[original_name]
                        except Exception as e2:
                            self.log.warning(f"Exception during individual lookup for {cal_type[0]}: {e2}, using default")
                            output_cals[original_name] = self.defaults[original_name]
        
            for cal,lookup in self.lookup_map.items():
                if cal not in subset:
                    continue
                    
                cal_start_time = time.time()
                
                if lookup == 'file':
                    file_start = time.time()
                    filename = self.caldate_files[cal]
                    fndir, fn = filename.split("/", 1)
                    # Use resources.open_text() to read the .csv because it has a relative path within repo
                    with resources.open_text(fndir, fn) as f:
                        df = pd.read_csv(f, header=0, skipinitialspace=True)
                    for i, row in df.iterrows():
                        start = datetime.strptime(row['UT_start_date'], "%Y-%m-%d %H:%M:%S")
                        end = datetime.strptime(row['UT_end_date'], "%Y-%m-%d %H:%M:%S")
                        if start <= dt < end:
                            try:
                                output_cals[cal] = eval(row['CALPATH'])
                            except SyntaxError:
                                output_cals[cal] = row['CALPATH']
                    file_lookup_time += time.time() - file_start
                    self.log.info(f"File lookup for {cal}: {time.time() - file_start:.3f}s")
                    
                elif lookup == 'wls' or lookup == 'etalon':
                    if self.use_db and db is None:
                        db = KPFDB(logger=self.log)
                    wls_start = time.time()
                    for cal_type in self.wls_cal_types:
                        wls_results = db.get_bracketing_wls(self.datetime, cal_type[1], max_cal_delta_time=self.max_age)
                        if len(wls_results) > 1 and (wls_results[0] == 0 or wls_results[2] == 0):
                            wls_files = [wls_results[1], wls_results[3]]
                            if wls_files[0] == None:
                                wls_files[0] = wls_files[1]
                            if wls_files[1] == None:
                                wls_files[1] = wls_files[0]
                            output_cals[cal] = wls_files
                            break
                        else:
                            output_cals[cal] = self.defaults[cal]
                            wls_files = output_cals[cal]

                    if lookup == 'etalon':
                        try:
                            new_dt = getheader(wls_files[0])['DATE-BEG']
                            self.datetime = new_dt
                        except:  # no DB available
                            pass
                        self.lookup_map['etalonmask'] = 'database'
                        output_cals[cal] = self.lookup(subset=['etalonmask'])['etalonmask']
                    
                    wls_lookup_time += time.time() - wls_start
                    self.log.info(f"WLS/Etalon lookup for {cal}: {time.time() - wls_start:.3f}s")
                
                self.log.info(f"Total time for {cal} ({lookup}): {time.time() - cal_start_time:.3f}s")
        finally:
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass

        total_time = time.time() - start_time
        self.log.info(f"=== LOOKUP TIMING SUMMARY ===")
        self.log.info(f"Total lookup time: {total_time:.3f}s")
        self.log.info(f"File lookups: {file_lookup_time:.3f}s")
        self.log.info(f"Database lookups: {database_lookup_time:.3f}s")
        self.log.info(f"WLS/Etalon lookups: {wls_lookup_time:.3f}s")
        self.log.info(f"Other overhead: {total_time - file_lookup_time - database_lookup_time - wls_lookup_time:.3f}s")

        return output_cals
    
    def clear_cache(self):
        """Clear the Redis cache for this calibration lookup"""
        try:
            from database.modules.utils.kpf_db import clear_cache
            clear_cache()
            self.log.info("Redis cache cleared successfully")
        except Exception as e:
            self.log.warning(f"Could not clear Redis cache: {e}")

