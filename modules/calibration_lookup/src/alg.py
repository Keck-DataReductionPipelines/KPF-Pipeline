
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
    def __init__(self, datetime, default_config_path, use_db=True, logger=None, verbose=False):
        """
        use_db (boolean) - to disable db access, set to False (e.g., when looking up file-based keywords only)
        """
        self.verbose = verbose
        if self.verbose:
            import time
            init_start = time.time()
        
        # Initialize DB class
        # self.db_lookup = QueryDBNearestMasterFilesFramework(self.action, self.context)

        #Input arguments
        self.datetime = datetime   # ISO datetime string
        
        if self.verbose:
            config_start = time.time()
        self.config = ConfigClass(default_config_path)
        if self.verbose:
            config_time = time.time() - config_start
        
        if self.verbose:
            logger_start = time.time()
        if logger == None:
            self.log = start_logger('GetCalibrations', default_config_path)
        else:
            self.log = logger
        if self.verbose:
            logger_time = time.time() - logger_start
        
        if self.verbose:
            eval_start = time.time()
        self.caldate_files = eval(self.config['PARAM']['date_files'])
        self.lookup_map = eval(self.config['PARAM']['lookup_map'])
        self.db_cal_types = eval(self.config['PARAM']['db_cal_types'])
        self.db_cal_file_levels = eval(self.config['PARAM']['db_cal_file_levels'])
        self.wls_cal_types = eval(self.config['PARAM']['wls_cal_types'])
        self.max_age = eval(self.config['PARAM']['max_cal_age'])
        self.defaults = eval(self.config['PARAM']['defaults'])
        if self.verbose:
            eval_time = time.time() - eval_start
        
        self.use_db = use_db
        
        if self.verbose:
            total_init_time = time.time() - init_start
            self.log.debug(f"GetCalibrations init timing:")
            self.log.debug(f"  Config loading: {config_time*1000:.2f}ms")
            self.log.debug(f"  Logger setup: {logger_time*1000:.2f}ms")
            self.log.debug(f"  Config eval: {eval_time*1000:.2f}ms")
            self.log.debug(f"  Total init: {total_init_time*1000:.2f}ms")

    def lookup(self, subset=None):
        if self.verbose:
            start_time = time.time()
        
        # Check cache first for the complete result
        # Apply timestamp rounding to avoid microsecond differences in cache keys
        dt = datetime.strptime(self.datetime, "%Y-%m-%dT%H:%M:%S.%f")
        # Round to nearest minute (same logic as in kpf_db.py)
        rounded_dt = dt.replace(second=0, microsecond=0)
        rounded_datetime = rounded_dt.strftime("%Y-%m-%dT%H:%M:%S")
        cache_key = f"calibration_lookup_complete:{rounded_datetime}"
        
        # Handle the case where subset is None (use all keys)
        if subset is None:
            subset = list(self.lookup_map.keys())
        
        try:
            from database.modules.utils.kpf_db import _get_cached_result
            cached_result = _get_cached_result(cache_key)
            if cached_result is not None:
                # Check if all requested keys are in the cached result
                missing_keys = [key for key in subset if key not in cached_result]
                if self.verbose:
                    self.log.debug(f"Cache check - cached_result keys: {sorted(cached_result.keys())}")
                    self.log.debug(f"Cache check - requested subset: {subset}")
                    self.log.debug(f"Cache check - missing_keys: {missing_keys}")
                
                if not missing_keys:
                    if self.verbose:
                         self.log.debug(f"Complete cache HIT - all {len(subset)} keys found")
                    return {key: cached_result[key] for key in subset}
                else:
                    if self.verbose:
                        self.log.debug(f"Partial cache HIT - {len(subset) - len(missing_keys)}/{len(subset)} keys found, missing: {missing_keys}")
                    # Start with cached results for keys we have
                    output_cals = {key: cached_result[key] for key in subset if key in cached_result}
                    # We'll need to query for missing keys
            else:
                if self.verbose:
                    self.log.debug(f"Complete cache MISS for key: {cache_key}")
                output_cals = {}
                missing_keys = list(subset)
        except Exception as e:
            if self.verbose:
                self.log.debug(f"Cache check failed: {e}")
            output_cals = {}
            missing_keys = list(subset)
        
        if self.verbose:
            # Time the datetime parsing
            dt_start = time.time()
            dt = datetime.strptime(self.datetime, "%Y-%m-%dT%H:%M:%S.%f")
            date_str = datetime.strftime(dt, "%Y%m%d")
            dt_time = time.time() - dt_start
        
        # Don't overwrite output_cals if we had cached results
        if 'output_cals' not in locals() or not output_cals:
            output_cals = {}
        
        db_results = None
        if subset is None:
            subset = self.lookup_map.keys()
        
        # Track timing for each lookup type
        file_lookup_time = 0
        database_lookup_time = 0
        wls_lookup_time = 0
        
        if self.verbose:
            self.log.debug(f"Lookup method timing:")
            self.log.debug(f"  Datetime parsing: {dt_time*1000:.2f}ms")
        
        # Collect all database calibration requests for batch processing
        db_requests = []
        cal_type_mapping = {}  # Map from database names back to original names
        
        for cal,lookup in self.lookup_map.items():
            if cal not in missing_keys:  # Only process missing keys
                continue
                
            if lookup == 'database':
                # Find the corresponding database calibration type
                found_mapping = False
                
                # Special handling for flat -> Flat mapping
                if cal == 'flat':
                    for lvl, cal_type in zip(self.db_cal_file_levels, self.db_cal_types):
                        if cal_type[0].lower() == 'flat':
                            cal_type_lookup = cal_type.copy()
                            cal_type_mapping['flat'] = 'flat'
                            if self.verbose:
                                self.log.debug(f"Found special flat -> Flat database mapping (level {lvl})")
                            db_requests.append((lvl, cal_type_lookup))
                            found_mapping = True
                            break
                
                # Regular mapping for other calibration types
                if not found_mapping:
                    for lvl, cal_type in zip(self.db_cal_file_levels, self.db_cal_types):
                        if cal_type[0].lower() == cal:
                            # Map the database result back to the original calibration name
                            cal_type_mapping[cal_type[0].lower()] = cal
                            
                            if self.verbose:
                                self.log.debug(f"Found database mapping for {cal} -> {cal_type[0]} (level {lvl})")
                            db_requests.append((lvl, cal_type))
                            found_mapping = True
                            break
                
                if not found_mapping:
                    if self.verbose:
                        self.log.debug(f"No database mapping found for {cal}")
        
        # Determine if we need a DB connection at all
        needs_db = any(self.lookup_map.get(cal) in ('database', 'wls', 'etalon') for cal in subset)
        db = None

        try:
            # Execute single batch query for all database calibrations
            if db_requests and self.use_db:
                if db is None:
                    db = KPFDB(logger=self.log)
                if self.verbose:
                    db_start = time.time()
                try:
                    # Use batch query for better performance
                    batch_results = db.get_nearest_master_batch(self.datetime, db_requests)
                    
                    # Process batch results
                    for lvl, cal_type in db_requests:
                        cal_type_name = cal_type[0].lower()
                        original_name = cal_type_mapping[cal_type_name]
                        
                        if self.verbose:
                            self.log.debug(f"Processing batch result for {cal_type_name} -> {original_name}")
                        
                        if cal_type_name in batch_results:
                            db_results = batch_results[cal_type_name]
                            if db_results[0] == 0:
                                if isinstance(cal_type[1], list):
                                    # Handle multi-results (like ordertrace)
                                    # db_results[1] is already a list, don't wrap it in another list
                                    output_cals[original_name] = db_results[1]
                                else:
                                    output_cals[original_name] = db_results[1]
                                if self.verbose:
                                    self.log.debug(f"Successfully set {original_name} = {output_cals[original_name]}")
                            else:
                                self.log.warning(f"Database lookup failed for {cal_type[0]} (exit code {db_results[0]}), using default")
                                output_cals[original_name] = self.defaults[original_name]
                                if self.verbose:
                                    self.log.debug(f"Using default for {original_name} = {output_cals[original_name]}")
                        else:
                            output_cals[original_name] = self.defaults[original_name]
                            if self.verbose:
                                self.log.debug(f"No batch result for {cal_type_name}, using default for {original_name} = {output_cals[original_name]}")
                    
                    if self.verbose:
                        database_lookup_time += time.time() - db_start
                        self.log.debug(f"Batch database lookup completed in {time.time() - db_start:.3f}s for {len(db_requests)} calibration types")
                    
                except Exception as e:
                    self.log.warning(f"Exception during batch database lookup: {e}, falling back to individual lookups")
                    # Fallback to individual lookups if batch fails
                    for lvl, cal_type in db_requests:
                        original_name = cal_type_mapping[cal_type[0].lower()]
                        try:
                            db_results = db.get_nearest_master(self.datetime, lvl, cal_type)
                            if db_results[0] == 0:
                                if isinstance(cal_type[1], list):
                                    # db_results[1] is already a list, don't wrap it in another list
                                    output_cals[original_name] = db_results[1]
                                else:
                                    output_cals[original_name] = db_results[1]
                            else:
                                output_cals[original_name] = self.defaults[original_name]
                        except Exception as e2:
                            self.log.warning(f"Exception during individual lookup for {cal_type[0]}: {e2}, using default")
                            output_cals[original_name] = self.defaults[original_name]
            
            # Process remaining calibration types (file, WLS, etalon)
            if self.verbose:
                self.log.debug(f"Processing remaining calibration types, missing_keys: {missing_keys}")
            for cal,lookup in self.lookup_map.items():
                if cal not in missing_keys:  # Only process missing keys
                    if self.verbose:
                        self.log.debug(f"Skipping {cal} (not in missing_keys)")
                    continue
                    
                if self.verbose:
                    cal_start_time = time.time()
                
                if lookup == 'file':
                    if self.verbose:
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
                    if self.verbose:
                        file_lookup_time += time.time() - file_start
                        self.log.debug(f"File lookup for {cal}: {time.time() - file_start:.3f}s")
                    
                elif lookup == 'wls' or lookup == 'etalon':
                    if self.use_db and db is None:
                        db = KPFDB(logger=self.log)
                    if self.verbose:
                        wls_start = time.time()
                    wls_files = None  # Initialize wls_files
                    for cal_type in self.wls_cal_types:
                        wls_results = db.get_bracketing_wls(self.datetime, cal_type[1], max_cal_delta_time=self.max_age)
                        if len(wls_results) > 1 and (wls_results[0] == 0 or wls_results[2] == 0):
                            wls_files = [wls_results[1], wls_results[3]]
                            if wls_files[0] == None:
                                wls_files[0] = wls_files[1]
                            if wls_files[1] == None:
                                wls_files[1] = wls_files[0]
                            
                            # Ensure deterministic file selection by sorting file paths
                            wls_files = sorted(wls_files)
                            output_cals[cal] = wls_files
                            
                            if self.verbose:
                                self.log.debug(f"Selected WLS files for {cal}: {wls_files}")
                            break
                    
                    # If no WLS files found, set default
                    if wls_files is None:
                        output_cals[cal] = self.defaults[cal]

                    if lookup == 'etalon' and wls_files is not None:
                        try:
                            new_dt = getheader(wls_files[0])['DATE-BEG']
                            etalon_datetime = new_dt
                            if self.verbose:
                                self.log.debug(f"Using etalon_datetime from WLS file: {etalon_datetime}")
                        except:  # no DB available
                            etalon_datetime = self.datetime
                            if self.verbose:
                                self.log.debug(f"Using fallback etalon_datetime: {etalon_datetime}")
                        
                        # Look up etalonmask using the INPUT timestamp (self.datetime), not the WLS file timestamp
                        # The WLS file timestamp is unrelated to when we need the etalonmask
                        etalonmask_found = False
                        for lvl, cal_type in zip(self.db_cal_file_levels, self.db_cal_types):
                            if cal_type[0].lower() == 'etalonmask':
                                try:
                                    # Use self.datetime (input timestamp) for etalonmask lookup
                                    etalonmask_result = db.get_nearest_master(self.datetime, lvl, cal_type)
                                    if etalonmask_result[0] == 0:
                                        output_cals['etalonmask'] = etalonmask_result[1]
                                        etalonmask_found = True
                                    else:
                                        output_cals['etalonmask'] = self.defaults['etalonmask']
                                        etalonmask_found = True
                                    break
                                except Exception as e:
                                    self.log.warning(f"Exception during etalonmask lookup: {e}, using default")
                                    output_cals['etalonmask'] = self.defaults['etalonmask']
                                    etalonmask_found = True
                                    break
                        
                        # If no etalonmask found in database types, use default
                        if not etalonmask_found:
                            output_cals['etalonmask'] = self.defaults['etalonmask']
                        
                        # Remove etalonmask from missing_keys so it doesn't get processed again in the main loop
                        if 'etalonmask' in missing_keys:
                            missing_keys.remove('etalonmask')
                            if self.verbose:
                                self.log.debug(f"Removed etalonmask from missing_keys, remaining: {missing_keys}")
                        else:
                            if self.verbose:
                                self.log.debug(f"etalonmask not in missing_keys: {missing_keys}")
                    
                    if self.verbose:
                        wls_lookup_time += time.time() - wls_start
                        self.log.debug(f"WLS/Etalon lookup for {cal}: {time.time() - wls_start:.3f}s")
                
                if self.verbose:
                    self.log.debug(f"Total time for {cal} ({lookup}): {time.time() - cal_start_time:.3f}s")
        finally:
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass

        if self.verbose:
            total_time = time.time() - start_time
            self.log.debug(f"=== LOOKUP TIMING SUMMARY ===")
            self.log.debug(f"Total lookup time: {total_time:.3f}s")
            self.log.debug(f"File lookups: {file_lookup_time:.3f}s")
            self.log.debug(f"Database lookups: {database_lookup_time:.3f}s")
            self.log.debug(f"WLS/Etalon lookups: {wls_lookup_time:.3f}s")
            self.log.debug(f"Other overhead: {total_time - file_lookup_time - database_lookup_time - wls_lookup_time:.3f}s")

        if self.verbose:
            # Debug: Check what's in output_cals before caching
            self.log.debug(f"output_cals before caching: {sorted(output_cals.keys())}")
            self.log.debug(f"output_cals values: {output_cals}")
        
        # Cache the complete result (including any new keys we just looked up)
        try:
            from database.modules.utils.kpf_db import _set_cached_result
            # Merge with existing cached results if we had a partial hit
            if 'cached_result' in locals() and cached_result is not None:
                complete_result = {**cached_result, **output_cals}
                if self.verbose:
                    self.log.debug(f"Merging cached results: {len(cached_result)} existing + {len(output_cals)} new = {len(complete_result)} total")
            else:
                complete_result = output_cals
                if self.verbose:
                    self.log.debug(f"Caching complete new result with {len(output_cals)} keys")
            
            _set_cached_result(cache_key, complete_result)
            if self.verbose:
                self.log.debug(f"Cached complete result for key: {cache_key}")
        except Exception as e:
            if self.verbose:
                self.log.debug(f"Failed to cache complete result: {e}")

        # Ensure we return exactly the keys requested in subset, in the same order
        final_result = {}
        for key in subset:
            if key in output_cals:
                final_result[key] = output_cals[key]
            else:
                if self.verbose:
                    # This shouldn't happen, but provide a fallback
                    self.log.debug(f"WARNING: Key {key} not found in output_cals, using default")
                    final_result[key] = self.defaults.get(key, None)
        
        if self.verbose:
            # Debug: Print the results for comparison
            self.log.debug(f"Requested subset: {subset}")
            self.log.debug(f"Final result keys: {sorted(final_result.keys())}")
            self.log.debug(f"Final result values: {final_result}")
        
        return final_result
    
    def clear_cache(self):
        """Clear the Redis cache for this calibration lookup"""
        try:
            from database.modules.utils.kpf_db import clear_cache
            clear_cache()
            if self.verbose:
                self.log.debug("Redis cache cleared successfully")
        except Exception as e:
            self.log.warning(f"Could not clear Redis cache: {e}")

