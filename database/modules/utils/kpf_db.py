import os
import psycopg2
import re
import hashlib
import pandas as pd
import numpy as np
import time
from astropy.time import Time
from functools import lru_cache
import threading
from datetime import datetime, timedelta
import configparser as cp
import json
import pickle

from kpfpipe.models.level1 import KPF1
from kpfpipe.logger import start_logger

DEFAULT_CFG_PATH = '/code/KPF-Pipeline/database/modules/utils/kpf_db.cfg'

# Redis cache configuration
_redis_client = None
_cache_enabled = True
_cache_ttl = 300  # 5 minutes TTL

# Detect test environment
_is_test_env = os.getenv('CI') == 'true' or os.getenv('TESTING') == 'true'

def _get_redis_client(verbose=False):
    """Get Redis client, creating it if needed"""
    global _redis_client, _cache_enabled
    if _redis_client is None:
        try:
            import redis
            # Try default port first, then alternative port
            ports_to_try = [6379, 6380]
            _redis_client = None
            
            for port in ports_to_try:
                try:
                    _redis_client = redis.Redis(
                        host='127.0.0.1',
                        port=port,
                        db=0,
                        decode_responses=False,  # Keep as bytes for pickle
                        socket_connect_timeout=1,
                        socket_timeout=1
                    )
                    # Test connection
                    _redis_client.ping()
                    if verbose:
                        print(f"DEBUG: Redis connection established on port {port}")
                    break
                except Exception as e:
                    print(f"DEBUG: Redis connection failed on port {port}: {e}")
                    _redis_client = None
                    continue
            
            if _redis_client is None:
                print("DEBUG: Redis connection failed on all ports, falling back to no cache")
                _cache_enabled = False
                
        except ImportError:
            print("DEBUG: Redis Python client not available, falling back to no cache")
            _redis_client = None
            _cache_enabled = False
    return _redis_client

def _load_cache_config():
    """Load cache configuration from config file"""
    try:
        config_obj = cp.ConfigParser()
        config_obj.read(DEFAULT_CFG_PATH)
        
        if 'CACHE' in config_obj:
            cache_cfg = config_obj['CACHE']
            enabled = cache_cfg.getboolean('enabled', True)
            ttl_seconds = cache_cfg.getint('ttl_seconds', 300)
            timestamp_rounding_minutes = cache_cfg.getint('timestamp_rounding_minutes', 1)
        else:
            # Default values if no CACHE section exists
            enabled = False
            ttl_seconds = 300
            timestamp_rounding_minutes = 1
            
        return enabled, ttl_seconds, timestamp_rounding_minutes
    except Exception:
        # Fallback to defaults if config reading fails
        return True, 300, 1

# Load cache configuration
_cache_enabled, _cache_ttl, _cache_rounding = _load_cache_config()

# Debug: Print cache configuration
verbose=False
if verbose:
    print(f"DEBUG: Cache configuration loaded:")
    print(f"DEBUG:   enabled: {_cache_enabled}")
    print(f"DEBUG:   ttl_seconds: {_cache_ttl}")
    print(f"DEBUG:   timestamp_rounding_minutes: {_cache_rounding}")

def _get_cache_key(obs_date, cal_requests, verbose=False):
    """Create a cache key that rounds timestamps to avoid microsecond differences"""
    # Round timestamp based on configurable rounding
    dt = datetime.strptime(obs_date, "%Y-%m-%dT%H:%M:%S.%f")
    if _cache_rounding > 1:
        # Round to nearest N minutes
        minutes = (dt.minute // _cache_rounding) * _cache_rounding
        rounded_dt = dt.replace(minute=minutes, second=0, microsecond=0)
    else:
        # Round to nearest minute
        rounded_dt = dt.replace(second=0, microsecond=0)
    
    rounded_obs_date = rounded_dt.strftime("%Y-%m-%dT%H:%M:%S")
    
    # Create a hash of the calibration requests
    requests_hash = hash(tuple(sorted(str(req) for req in cal_requests)))
    
    cache_key = f"calibration_lookup:{rounded_obs_date}_{requests_hash}"
    
    # Debug logging
    if verbose:
        print(f"DEBUG: Original timestamp: {obs_date}")
        print(f"DEBUG: Rounded timestamp: {rounded_obs_date}")
        print(f"DEBUG: Requests hash: {requests_hash}")
        print(f"DEBUG: Final cache key: {cache_key}")
    
    return cache_key

def _get_cached_result(cache_key, verbose=False):
    """Get cached result from Redis if it exists and is not expired"""
    if not _cache_enabled:
        if _is_test_env:
            return None  # Less verbose in test environment
        if verbose:
            print(f"DEBUG: Cache disabled, returning None")
        return None
        
    redis_client = _get_redis_client()
    if redis_client is None:
        return None
        
    try:
        start_time = time.time()
        cached_data = redis_client.get(cache_key)
        redis_time = time.time() - start_time
        
        if cached_data is not None:
            # Unpickle the cached result
            unpickle_start = time.time()
            result = pickle.loads(cached_data)
            unpickle_time = time.time() - unpickle_start
            
            if not _is_test_env:
                if verbose:
                    print(f"DEBUG: Redis cache HIT for key: {cache_key}")
                    print(f"DEBUG: Redis GET time: {redis_time*1000:.2f}ms, Unpickle time: {unpickle_time*1000:.2f}ms")
            return result
        else:
            if not _is_test_env:
                if verbose:
                    print(f"DEBUG: Redis cache MISS for key: {cache_key}")
                    print(f"DEBUG: Redis GET time: {redis_time*1000:.2f}ms")
    except Exception as e:
        if not _is_test_env:
            print(f"DEBUG: Redis cache error: {e}")
    
    return None

def _set_cached_result(cache_key, result, verbose=False):
    """Store result in Redis cache with TTL"""
    if not _cache_enabled:
        if not _is_test_env:
            if verbose:
                print(f"DEBUG: Cache disabled, not storing result")
        return
        
    redis_client = _get_redis_client()
    if redis_client is None:
        return
        
    try:
        # Pickle the result for storage
        pickle_start = time.time()
        pickled_result = pickle.dumps(result)
        pickle_time = time.time() - pickle_start
        
        redis_start = time.time()
        redis_client.setex(cache_key, _cache_ttl, pickled_result)
        redis_time = time.time() - redis_start
        
        if not _is_test_env:
            if verbose:
                print(f"DEBUG: Redis cached result for key: {cache_key} with TTL {_cache_ttl}s")
                print(f"DEBUG: Pickle time: {pickle_time*1000:.2f}ms, Redis SET time: {redis_time*1000:.2f}ms")
    except Exception as e:
        print(f"DEBUG: Redis cache storage error: {e}")

def clear_cache(verbose=False):
    """Clear all Redis cache entries for this application"""
    if not _cache_enabled:
        return
        
    redis_client = _get_redis_client()
    if redis_client is None:
        return
        
    try:
        # Get all keys with our prefixes and delete them
        keys1 = redis_client.keys("calibration_lookup:*")
        keys2 = redis_client.keys("calibration_lookup_complete:*")
        all_keys = keys1 + keys2
        
        if all_keys:
            redis_client.delete(*all_keys)
            if not _is_test_env:
                if verbose:
                    print(f"DEBUG: Cleared {len(all_keys)} cache entries")
            else:
                if verbose:
                    print(f"DEBUG: Cleared {len(all_keys)} cache entries ({len(keys1)} individual + {len(keys2)} complete)")
        else:
            if not _is_test_env:
                if verbose:
                    print("DEBUG: No cache entries to clear")
            else:
                if verbose:
                    print("DEBUG: No cache entries to clear")
    except Exception as e:
        if not _is_test_env:
            print(f"DEBUG: Error clearing cache: {e}")
        else:
            print(f"DEBUG: Error clearing cache: {e}")

def clear_cache_for_timestamp(obs_date, verbose=False):
    """Clear cache entries for a specific timestamp"""
    if not _cache_enabled:
        return
        
    redis_client = _get_redis_client()
    if redis_client is None:
        return
        
    try:
        # Round timestamp the same way we do for cache keys
        dt = datetime.strptime(obs_date, "%Y-%m-%dT%H:%M:%S.%f")
        if _cache_rounding > 1:
            minutes = (dt.minute // _cache_rounding) * _cache_rounding
            rounded_dt = dt.replace(minute=minutes, second=0, microsecond=0)
        else:
            rounded_dt = dt.replace(second=0, microsecond=0)
        
        rounded_obs_date = rounded_dt.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Find and delete keys for this timestamp
        pattern = f"calibration_lookup:{rounded_obs_date}_*"
        keys = redis_client.keys(pattern)
        if keys:
            redis_client.delete(*keys)
            if not _is_test_env:
                if verbose:
                    print(f"DEBUG: Cleared {len(keys)} cache entries for timestamp {rounded_obs_date}")
    except Exception as e:
        if not _is_test_env:
            print(f"DEBUG: Error clearing cache for timestamp: {e}")

# Common methods.

def md5(fname):
    """
    Returns checksum = 68 if it fails to compute the MD5 checksum.
    """

    hash_md5 = hashlib.md5()

    try:
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        print("*** Error: Failed to compute checksum =",fname,"; quitting...")
        return 68


class KPFDB:

    """
    Class to facilitate execution of queries in the KPF operations database.
    For each query a different method is defined.

    Returns exitcode:
         0 = Normal
         2 = Exception raised closing database connection
        64 = Cannot connect to database
        65 = Input file does not exist
        66 = File checksum does not match database checksum
        67 = Could not execute query
        68 = Failed to compute checksum
    """

    def __init__(self, logger=None, verbose=False):
        if logger == None:
            self.log = start_logger('KPFDB', DEFAULT_CFG_PATH)
        else:
            self.log = logger

        self.verbose = verbose
        self.exit_code = 0
        self.cId = None
        self.db_level = None
        self.db_cal_type = None
        self.db_object = None
        self.infobits = None
        self.filename = None
        self.conn = None
        self.cur = None
        
        # Connection pooling - reuse connection if available
        self._connection_pool = {}
        self._max_connections = 5

        # Get database connection parameters from environment.

        dbport = os.getenv('DBPORT')
        dbname = os.getenv('DBNAME')
        dbuser = os.getenv('DBUSER')
        dbpass = os.getenv('DBPASS')
        dbserver = os.getenv('DBSERVER')

        # Connect to database
        db_fail = True
        n_attempts = 3
        wait_times = [1,3,10]
        for i in range(n_attempts):
            try:
                self.conn = psycopg2.connect(host=dbserver,database=dbname,port=dbport,user=dbuser,password=dbpass)
                db_fail = False
                break
            except:
                self.log.warning("Could not connect to database, retrying...")
                db_fail = True
                time.sleep(wait_times[1])

        if db_fail:
            self.log.warning(f"Could not connect to database after {n_attempts} attempts...")
            self.exit_code = 64
            return

        # Open database cursor.

        self.cur = self.conn.cursor()

        # Select database version.

        q1 = 'SELECT version();'
        if self.verbose:
            self.log.debug('q1 = {}'.format(q1))
        self.cur.execute(q1)
        db_version = self.cur.fetchone()
        if self.verbose:
            self.log.debug('PostgreSQL database version = {}'.format(db_version))

        # Check database current_user.

        q2 = 'SELECT current_user;'
        if self.verbose:
            self.log.debug('q2 = {}'.format(q2))
        self.cur.execute(q2)
        for record in self.cur:
            if self.verbose:
                self.log.debug('record = {}'.format(record))

    def _get_connection(self):
        """Get a database connection, reusing existing ones when possible"""
        if self.conn is None or self.conn.closed:
            # Reconnect if connection is lost
            dbport = os.getenv('DBPORT')
            dbname = os.getenv('DBNAME')
            dbuser = os.getenv('DBUSER')
            dbpass = os.getenv('DBPASS')
            dbserver = os.getenv('DBSERVER')
            
            try:
                self.conn = psycopg2.connect(host=dbserver,database=dbname,port=dbport,user=dbuser,password=dbpass)
                self.cur = self.conn.cursor()
                if self.verbose:
                    self.log.debug("Reconnected to database")
            except Exception as e:
                self.log.error(f"Failed to reconnect to database: {e}")
                return False
        return True

    @lru_cache(maxsize=128)
    def _cached_query(self, query_hash, query):
        """Cache query results to avoid repeated identical queries"""
        if not self._get_connection():
            return pd.DataFrame([])
        
        try:
            results = pd.read_sql_query(query, self.conn)
            return results
        except Exception as e:
            self.log.warning(f"Error running cached database query: {e}")
            return pd.DataFrame([])

    def query_to_pandas(self, query):
        if not self._get_connection():
            return pd.DataFrame([])
            
        try:
            results = pd.read_sql_query(query, self.conn)
        except Exception as e:
            self.log.warning(f"Error running database query: {e}")
            results = pd.DataFrame([])

        return results

    def get_nearest_master(self, obs_date, cal_file_level, cal_type_pair, contentbitmask=3, max_cal_delta_time='1000 days'):
        """Get the master file closest in time to obs_date of the specified calibration type
        
        Args:
            obs_date (string): ISO formatted datetime string
            cal_file_level (int): data level (0, 1, 2)
            cal_type_pair (list): two-element list. First element is the calibration type (e.g. bias) and the second element is the object name (e.g. autocal-bias)
            contentbitmask (int): (optional) contentbitmask flag to match (default=3)
            max_cal_delta_time (string): (optional) maximum delta time between obs_date and the calibration file to consider (default = 14 days)

        Returns:
            string: path to calibration file
        
        """
        query_template = f"""
SELECT *,
(minmjd + maxmjd)/2 as meanmjd
FROM calfiles
WHERE CAST('{obs_date}' as date) BETWEEN (startdate - INTERVAL '{max_cal_delta_time}') AND (startdate + INTERVAL '{max_cal_delta_time}')
AND level = '{cal_file_level}'
AND caltype = '{cal_type_pair[0].lower()}'
AND object like '%{cal_type_pair[1]}%'
ORDER BY startdate;"""
        
        # AND contentbits = {contentbitmask}

        # print(query_template)
        df = self.query_to_pandas(query_template)
        if len(df) == 0:
            return [1, None]

        print(df)

        obst = Time(obs_date)
        obs_jd = obst.mjd
        
        # only look backwards for etalon masks
        # if cal_type_pair[0].lower() == 'etalonmask':
        #     df = df[df['meanmjd'] < obs_jd]

        df['delta'] = (df['meanmjd'] - obs_jd).abs()
        if df['delta'].isnull().all():
            odt = pd.to_datetime(obs_date)
            df['delta'] = odt - pd.to_datetime(df['startdate'])

        best_match = df.loc[df['delta'].idxmin()]
        fname = os.path.join('/', best_match['filename'])
        self.verify_checksum(fname, best_match['checksum'])

        return [self.exit_code, fname]
    
    def get_bracketing_wls(self, obs_date, object_name, contentbitmask=3, max_cal_delta_time='90 days'):
        """Get the WLS files that bracket a given obs_date
        
        Args:
            obs_date (string): ISO formatted datetime string
            object_name (list): Partial object name to search for (e.g. autocal-lfc-all). Object name in database must contain this name but doesn't need to be a complete match.
            contentbitmask (int): (optional) contentbitmask flag to match (default=3)
            max_cal_delta_time (string): (optional) maximum delta time between obs_date and the calibration file to consider (default = 3 days)
        
        """

        query_template = f"""
SELECT *,
(minmjd + maxmjd)/2 as meanmjd
FROM calfiles
WHERE CAST('{obs_date}' as date) BETWEEN (startdate - INTERVAL '{max_cal_delta_time}') AND (startdate + INTERVAL '{max_cal_delta_time}')
and level = 1
AND caltype = 'wls'
AND (object like '%{object_name}-eve%' OR object like '%{object_name}-morn%')
ORDER BY startdate;"""
        
        df = self.query_to_pandas(query_template)
        if len(df) == 0:
            return [1, None, 1, None]

        obst = Time(obs_date)
        obs_jd = obst.mjd

        mjds = []
        for i, row in df.iterrows():
            try:
                minmjd_check = pd.isna(row['minmjd'])
                maxmjd_check = pd.isna(row['maxmjd'])
                if minmjd_check or maxmjd_check:
                    raise ValueError("Min MJD and/or max MJD is not numeric.")
                else:
                    mjds.append((row['maxmjd'] + row['minmjd']) / 2)
            except ValueError:
                fname = '/' + row['filename']
                l1 = KPF1.from_fits(fname)
                dt = l1.header['PRIMARY']['DATE-MID']
                mjd = Time(dt).mjd
                mjds.append(mjd)

        df['meanmjd'] = mjds

        df['delta'] = (df['meanmjd'] - obs_jd).abs()
        before_df = df[df['meanmjd'] < obs_jd]
        after_df = df[df['meanmjd'] >= obs_jd]

        try:
            if len(before_df) == 0:
                raise ValueError("No files found before observation time")
            best_before = before_df.loc[before_df['delta'].idxmin()]
            fname_before = os.path.join('/', best_before['filename'])
            if not os.path.exists(fname_before):
                raise IOError(f"{fname_before} does not exist.")
            self.verify_checksum(fname_before, best_before['checksum'])
            before_code = 0
        except (TypeError, ValueError, IOError):
            fname_before = None
            before_code = 1
        try:
            if len(after_df) == 0:
                raise ValueError("No files found after observation time")
            best_after = after_df.loc[after_df['delta'].idxmin()]
            fname_after = os.path.join('/', best_after['filename'])
            if not os.path.exists(fname_after):
                raise IOError(f"{fname_after} does not exist.")
            self.verify_checksum(fname_after, best_after['checksum'])
            after_code = 0
        except (TypeError, ValueError, IOError):
            fname_after = None
            after_code = 1

        return [before_code, fname_before, after_code, fname_after]


    def get_nearest_master_file(self,obs_date,cal_file_level,contentbitmask,cal_type_pair,max_cal_file_age='1000 days'):

        '''
        Get nearest master file for the specified set of input parameters.

        obs_date is a YYYYMMDD (string or number)
        '''

        # Reinitialize.

        self.cId = None
        self.db_level = None
        self.db_cal_type = None
        self.db_object = None
        self.infobits = None
        self.filename = None
        self.exit_code = 0


        # Define query template.

        query_template =\
            "select * from getCalFile(" +\
            "cast(OBSDATE as date)," +\
            "cast(LEVEL as smallint)," +\
            "cast('CALTYPE' as character varying(32))," +\
            "cast('OBJECT' as character varying(32))," +\
            "cast(CONTENTBITMASK as integer), " +\
            "cast('MAXFILEAGE' as interval)) as " +\
            "(cId integer," +\
            " level smallint," +\
            " caltype varchar(32)," +\
            " object varchar(32)," +\
            " filename varchar(255)," +\
            " checksum varchar(32)," +\
            " infobits integer," +\
            " startDate date);"

        obs_date_str = str(obs_date)
        obsdate = "'" + obs_date_str[0:4] + "-" + obs_date_str[4:6] + "-" + obs_date_str[6:8] + "'"


        # Query database for all cal_types.

        if self.verbose:
            self.log.debug('----> cal_file_level = {}'.format(cal_file_level))
            self.log.debug('----> contentbitmask = {}'.format(contentbitmask))
            self.log.debug('----> cal_type_pair = {}'.format(cal_type_pair))

        levelstr = str(cal_file_level)
        cal_type = cal_type_pair[0]
        object = cal_type_pair[1]

        rep = {"OBSDATE": obsdate,
               "LEVEL": levelstr,
               "CALTYPE": cal_type,
               "OBJECT": object,
               "MAXFILEAGE": max_cal_file_age}

        rep["CONTENTBITMASK"] = str(contentbitmask)

        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        query = pattern.sub(lambda m: rep[re.escape(m.group(0))], query_template)

        if self.verbose:
            self.log.debug('query = {}'.format(query))


        # Execute query.

        try:
            self.cur.execute(query)

        except (Exception, psycopg2.DatabaseError) as error:
            self.log.error('*** Error executing query ({}); skipping...'.format(query))
            self.exit_code = 67
            return


        record = self.cur.fetchone()

        if record is not None:
            cId = record[0]
            db_level = record[1]
            db_cal_type = record[2]
            db_object = record[3]
            filename = '/' + record[4]         # docker run has -v /data/kpf/masters:/masters
            checksum = record[5]
            infobits = record[6]

            if self.verbose:
                self.log.debug('cId = {}'.format(cId))
                self.log.debug('filename = {}'.format(filename))
                self.log.debug('checksum = {}'.format(checksum))

            self.verify_checksum(filename, checksum)

            self.cId = cId
            self.db_level = db_level
            self.db_cal_type = db_cal_type
            self.db_object = db_object
            self.infobits = infobits

    def verify_checksum(self, filename, checksum):
        # See if file exists.
        isExist = os.path.exists(filename)
        if self.verbose:
            self.log.debug('File existence = {}'.format(isExist))

        if isExist is True:
            if self.verbose:
                self.log.debug("File exists...")
        else:
            self.log.error("*** Error: File does not exist; quitting...")
            self.exit_code = 65
            return


        # Compute checksum and compare with database value.

        cksum = md5(filename)
        if self.verbose:
            self.log.debug('cksum = {}'.format(cksum))

        if  cksum == 68:
            self.exit_code = 68
            return

        if cksum == checksum:
            if self.verbose:
                self.log.debug("File checksum is correct ({})...".format(filename))
            self.filename = filename
            self.exit_code = 0
        else:
            self.log.error("*** Error: File checksum is incorrect ({}); quitting...".format(filename))
            self.exit_code = 66
            return


    def close(self):

        '''
        Close database cursor and then connection.
        '''

        try:
            self.cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            self.log.error(error)
            self.exit_code = 2
        finally:
            if self.conn is not None:
                self.conn.close()
                if self.verbose:
                    self.log.debug('Database connection closed.')

    def get_nearest_master_batch(self, obs_date, cal_requests, max_cal_delta_time='1000 days'):
        """Get multiple master files in a single database query for better performance
        
        Args:
            obs_date (string): ISO formatted datetime string
            cal_requests (list): List of tuples (cal_file_level, cal_type_pair)
            max_cal_delta_time (string): maximum delta time between obs_date and the calibration file to consider
        
        Returns:
            dict: Dictionary mapping cal_type to [exit_code, filename]
        """
        if not cal_requests:
            return {}
            
        start_time = time.time()
        
        # Don't cache individual batch results - they're already cached in the complete result cache
        # This prevents cache key inconsistencies when cal_requests change between runs
        
        # Build a single optimized query for all calibration types
        query_parts = []
        for level, cal_type_pair in cal_requests:
            cal_type = cal_type_pair[0].lower()
            object_name = cal_type_pair[1]
            
            # Handle list objects (like ordertrace)
            if isinstance(object_name, list):
                for obj in object_name:
                    query_parts.append(f"""
                        (level = {level} AND caltype = '{cal_type}' AND object LIKE '%{obj}%')
                    """)
            else:
                query_parts.append(f"""
                    (level = {level} AND caltype = '{cal_type}' AND object LIKE '%{object_name}%')
                """)
        
        # Use optimized query with proper indexing hints
        query = f"""
        WITH all_cals AS (
            SELECT *,
                   (minmjd + maxmjd)/2 as meanmjd
            FROM calfiles
            WHERE CAST('{obs_date}' as date) BETWEEN (startdate - INTERVAL '90 days') AND (startdate + INTERVAL '90 days')
            AND ({' OR '.join(query_parts)})
        ),
        ranked_cals AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY level, caltype, object 
                       ORDER BY ABS((minmjd + maxmjd)/2 - {Time(obs_date).mjd})
                   ) as rn
            FROM all_cals
        )
        SELECT level, caltype, object, filename, checksum, minmjd, maxmjd, meanmjd
        FROM ranked_cals 
        WHERE rn = 1
        ORDER BY level, caltype;
        """
        
        # Add query optimization hints
        if self.verbose:
            self.log.debug(f"Executing batch query for {len(cal_requests)} calibration types")
        df = self.query_to_pandas(query)
        
        # Process results and match back to original requests
        results = {}
        obst = Time(obs_date)
        obs_jd = obst.mjd
        
        for level, cal_type_pair in cal_requests:
            cal_type = cal_type_pair[0].lower()
            object_name = cal_type_pair[1]
            
            # Handle list objects (like ordertrace)
            if isinstance(object_name, list):
                # For list objects, we need to collect all matching results
                matching_results = []
                for obj in object_name:
                    mask = (df['level'] == level) & (df['caltype'] == cal_type) & (df['object'].str.contains(obj, na=False))
                    matching_rows = df[mask]
                    if len(matching_rows) > 0:
                        best_match = matching_rows.iloc[0]
                        fname = os.path.join('/', best_match['filename'])
                        self.verify_checksum(fname, best_match['checksum'])
                        if self.exit_code == 0:
                            matching_results.append(fname)
                
                if matching_results:
                    results[cal_type] = [0, matching_results]
                else:
                    results[cal_type] = [1, None]
            else:
                # Find matching result for single object
                mask = (df['level'] == level) & (df['caltype'] == cal_type) & (df['object'].str.contains(object_name, na=False))
                matching_rows = df[mask]
                
                if len(matching_rows) == 0:
                    results[cal_type] = [1, None]
                    continue
                    
                # Get the best match (already ranked by the query)
                best_match = matching_rows.iloc[0]
                
                # Verify file and checksum
                fname = os.path.join('/', best_match['filename'])
                self.verify_checksum(fname, best_match['checksum'])
                
                results[cal_type] = [self.exit_code, fname]
        
        query_time = time.time() - start_time
        if verbose:
            self.log.info(f"Batch query completed in {query_time:.3f}s for {len(cal_requests)} calibration types")
        
        return results
