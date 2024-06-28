import os
from os.path import exists
import numpy as np
import numpy.ma as ma
import configparser as cp
import psycopg2
import re
import hashlib
import ast

from modules.quicklook.src.analyze_l0 import AnalyzeL0

# Pipeline dependencies
from kpfpipe.logger import *
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from kpfpipe.pipelines.fits_primitives import to_fits
from keckdrpframework.models.arguments import Arguments

import database.modules.utils.kpf_db as db

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/read_noise/configs/default.cfg'

debug = 0


class ReadNoiseFramework(KPF0_Primitive):

    """
    Description:
        Analyzes an L0 FITS file.  Computes the instantaneous read noise and stores it in the database ReadNoise table.
        Updates the header of the 2D FITS file only if backfill_repopulate_db_recs = 0.

    Arguments:
        data_type (str): Type of data (e.g., KPF).
        l0_filename (str): Full path and filename of L0 FITS file within container.
        n_sigma (float): Number of sigmas for overscan-value outlier rejection (e.g., 3.0).
        rId (float): Primary database key of L0 FITS file in L0Files database record.


    """

    def __init__(self, action, context):

        KPF0_Primitive.__init__(self, action, context)

        self.data_type = self.action.args[0]
        self.l0_filename = self.action.args[1]
        self.n_sigma = self.action.args[2]
        self.rId = self.action.args[3]
        
        self.gain_dict = {
            'GREEN_AMP1': 5.175,
            'GREEN_AMP2': 5.208,
            'GREEN_AMP3': 5.52,
            'GREEN_AMP4': 5.39,
            'RED_AMP1': 5.02,
            'RED_AMP2': 5.27,
            'RED_AMP3': 5.32,
            'RED_AMP4': 5.23,
            'CA_HK': 5.0}

        try:
            self.module_config_path = context.config_path['read_noise']
            print("--->",self.__class__.__name__,": self.module_config_path =",self.module_config_path)
        except:
            self.module_config_path = DEFAULT_CFG_PATH

        print("{} class: self.module_config_path = {}".format(self.__class__.__name__,self.module_config_path))

        print("Starting logger...")
        self.logger = start_logger(self.__class__.__name__, self.module_config_path)

        if self.logger is not None:
            print("--->self.logger is not None...")
        else:
            print("--->self.logger is None...")

        self.logger.info('Started {}'.format(self.__class__.__name__))
        self.logger.debug('module_config_path = {}'.format(self.module_config_path))

        module_config_obj = cp.ConfigParser()
        res = module_config_obj.read(self.module_config_path)
        if res == []:
            raise IOError('failed to read {}'.format(self.module_config_path))

        module_param_cfg = module_config_obj['PARAM']

        backfill_repopulate_db_recs_cfg_str = module_param_cfg.get('backfill_repopulate_db_recs')
        self.backfill_repopulate_db_recs_cfg = ast.literal_eval(backfill_repopulate_db_recs_cfg_str)
        self.backfill_repopulate_db_query_template = module_param_cfg.get('backfill_repopulate_db_query_template')

        self.logger.info('self.data_type = {}'.format(self.data_type))
        self.logger.info('self.l0_filename = {}'.format(self.l0_filename))
        self.logger.info('self.n_sigma = {}'.format(self.n_sigma))
        self.logger.info('self.rId = {}'.format(self.rId))

        self.logger.info('self.backfill_repopulate_db_recs_cfg = {}'.format(self.backfill_repopulate_db_recs_cfg))

        self.logger.info('Type of self.backfill_repopulate_db_recs_cfg = {}'.format(type(self.backfill_repopulate_db_recs_cfg)))

        self.logger.info('self.backfill_repopulate_db_query_template = {}'.format(self.backfill_repopulate_db_query_template))


    def compute_clip_corr(self,n_sigma):

        """
        Compute a correction factor to properly reinflate the variance after it is
        naturally diminished via data-clipping.  Employ a simple Monte Carlo method
        and standard normal deviates to simulate the data-clipping and obtain the
        correction factor.
        """

        var_trials = []
        for x in range(0,10):
            a = np.random.normal(0.0, 1.0, 1000000)
            med = np.median(a, axis=0)
            p16 = np.percentile(a, 16, axis=0)
            p84 = np.percentile(a, 84, axis=0)
            sigma = 0.5 * (p84 - p16)
            mdmsg = med - n_sigma * sigma
            b = np.less(a,mdmsg)
            mdpsg = med + n_sigma * sigma
            c = np.greater(a,mdpsg)
            mask = np.any([b,c],axis=0)
            mx = ma.masked_array(a, mask)
            var = ma.getdata(mx.var(axis=0))
            var_trials.append(var)

        np_var_trials = np.array(var_trials)
        avg_var_trials = np.mean(np_var_trials)
        std_var_trials = np.std(np_var_trials)
        corr_fact = 1.0 / avg_var_trials

        if debug == 1:
            print('---->compute_clip_corr(): n_sigma,avg_var_trials,std_var_trials,corr_fact = {},{},{},{}'.\
                format(n_sigma,avg_var_trials,std_var_trials,corr_fact))

        return corr_fact


    def statistics_with_clipping(self,data_array,n_sigma,cf):

        """
        Statistics with outlier rejection (n-sigma data-trimming), ignoring NaNs, across all data array dimensions.
        """

        a = np.array(data_array)

        med = np.nanmedian(a)
        p16 = np.nanpercentile(a,16)
        p84 = np.nanpercentile(a,84)
        sigma = 0.5 * (p84 - p16)
        mdmsg = med - n_sigma * sigma
        b = np.less(a,mdmsg)
        mdpsg = med + n_sigma * sigma
        c = np.greater(a,mdpsg)
        d = np.where(np.isnan(a),True,False)
        mask = b | c | d
        mx = ma.masked_array(a, mask)
        avg = ma.getdata(mx.mean())
        var = ma.getdata(mx.var()) * cf
        cnt = ma.getdata(mx.count())

        if debug == 1:
            print("------------------------------->avg=",avg)
            print('------------------------------->Type of avg= {}'.format(type(avg)))
            myavg = avg.item()     # Convert data type from <class 'numpy.ndarray'> to <class 'float'>
            print("------------------------------->myavg=",myavg)
            print('------------------------------->Type of myavg= {}'.format(type(myavg)))
            print("------------------------------->var=",var)
            print('------------------------------->Type of var= {}'.format(type(var)))
            myvar = var.item()     # Convert data type from <class 'numpy.float64'> to <class 'float'>
            print("------------------------------->myvar=",myvar)
            print('------------------------------->Type of myvar= {}'.format(type(myvar)))
            print("------------------------------->cnt=",cnt)
            print('------------------------------->Type of cnt= {}'.format(type(cnt)))
            mycnt = cnt.item()     # Convert data type from <class 'numpy.ndarray'> to <class 'int'>
            print("------------------------------->mycnt=",mycnt)
            print('------------------------------->Type of mycnt= {}'.format(type(mycnt)))

        return avg,var,cnt


    def compute_read_noise(self,ext,fname_input,hdul_input,n_sigma,cf):

        """
        Compute read noise for given read-out channel.
        """

        try:
            data_input = np.array(hdul_input[ext])
        except:
            if debug == 1:
                print("FITS extension {} not found in header ({}); skipping...".\
                    format(ext,fname_input))
            return


        n_dims = len(np.shape(data_input))
        self.logger.debug('------------->ext,n_dims = {},{}'.format(ext,n_dims))
        if n_dims != 2:       # Check if valid data extension
            return

        nx = (np.shape(data_input))[0]
        ny = (np.shape(data_input))[1]

        if debug == 1:
            self.logger.debug('--------->ext,nx,ny = {},{},{}'.format(ext,nx,ny))

        naxis1 = hdul_input.header[ext]["NAXIS1"]
        naxis2 = hdul_input.header[ext]["NAXIS2"]

        if debug == 1:
            print("naxis1,naxis2 = {},{}".\
                    format(naxis1,naxis2))

        nystart = 100
        nyend = naxis2 - 200

        nxoscanstart = 10
        nxoscanend = 50

        if "AMP1" in ext:
            nxstart = naxis1 - nxoscanend
            nxend = naxis1 - nxoscanstart
        elif "AMP3" in ext:
            nxstart = naxis1 - nxoscanend
            nxend = naxis1 - nxoscanstart
        else:
            nxstart = nxoscanstart
            nxend = nxoscanend

        if debug == 1:
            print("nystart,nyend,nxstart,nxend = {},{},{},{}".\
                    format(nystart,nyend,nxstart,nxend))

        data = np.array(data_input[nystart:nyend,nxstart:nxend])

        #
        # Statistics with outlier rejection (n-sigma data-trimming), ignoring NaNs.
        #

        mean,var,cnt = self.statistics_with_clipping(data,n_sigma,cf)

        if debug == 1:
            print("cnt = {}".\
                format(cnt))

        try:
            gain = hdul_input.header[ext]["CCDGAIN"]
        except:
            self.logger.info("*** Error: GAIN keyword not found in header ({}); defaulting to dictionary ({})...".\
                format(fname_input,self.gain_dict[ext]))
            gain = self.gain_dict[ext]


        # Less robust single-image method.

        read_noise = gain * np.sqrt(var) / (2**16)

        self.logger.info("extname,Gain (electrons/DN),Read noise (electrons) = {},{},{}".\
            format(ext,gain,read_noise))


        return read_noise


    def computeReadNoiseForSingleL0File(self,input_rid,input_filename,n_sigma,cf,cur):

        read_noise_exit_code = 0


        # See if file exists.

        isExist = os.path.exists(input_filename)
        #self.logger.info('File existence = {}'.format(isExist))

        if isExist is False:
            self.logger.info('Input file does not exist...')
            read_noise_exit_code = 65
            return read_noise_exit_code


        ###########################################################################

        # There is no point having the following list as a configuration parameter because it is closely tied to the database schema.
        # Also, we want the FITS-header keywords to be fixed for all 2D FITS files.

        lev0_ffi_exts = ['GREEN_AMP1','GREEN_AMP2','GREEN_AMP3','GREEN_AMP4','RED_AMP1','RED_AMP2','RED_AMP3','RED_AMP4','CA_HK']
        read_noise_kwds = ['RNGREEN1','RNGREEN2','RNGREEN3','RNGREEN4','RNRED1','RNRED2','RNRED3','RNRED4','RNCAHK']
        read_noise_kwd_dict = dict(zip(lev0_ffi_exts, read_noise_kwds))


        # Read image data object from L0 FITS file.

        hdul_input = KPF0.from_fits(input_filename,self.data_type)

        read_noise_dict = {}
        for ffi in lev0_ffi_exts:
            rn = self.compute_read_noise(ffi,input_filename,hdul_input,n_sigma,cf)

            if rn != None:
                 rn = round(rn, 5)     # Round to 5 places to the right of the decimal point (stored as real in Postgres database,
                                       # and the 7 decimal place displayed by the psql client is a random digit in memory, and not
                                       # a rounded result).  Ensures the rounded database result matches FITS keyword value precisely.

            self.logger.debug('----->ffi,rn = {},{}'.format(ffi,rn))

            read_noise_dict[ffi] = rn

            if rn is None:
                rn = "null"

            if ffi == "GREEN_AMP1":
                rngreen1 = rn

            elif ffi == "GREEN_AMP2":
                rngreen2 = rn

            elif ffi == "GREEN_AMP3":
                rngreen3 = rn

            elif ffi == "GREEN_AMP4":
                rngreen4 = rn

            if ffi == "RED_AMP1":
                rnred1 = rn

            elif ffi == "RED_AMP2":
                rnred2 = rn

            if ffi == "RED_AMP3":
                rnred3 = rn

            elif ffi == "RED_AMP4":
                rnred4 = rn

            elif ffi == "CA_HK":
                rncahk = rn


        try:
            myL0 = AnalyzeL0(hdul_input)

            try:
                greenreadtime = round(myL0.green_read_time, 3)
            except AttributeError:
                greenreadtime = "null"

            try:
                redreadtime = round(myL0.red_read_time, 3)
            except AttributeError:
                redreadtime = "null"

            try:
                readspeed = myL0.read_speed
            except AttributeError:
                readspeed = "null"

        except:
            greenreadtime = "null"
            redreadtime = "null"
            readspeed = "null"

            
        self.logger.info('greenreadtime = {}'.format(greenreadtime))
        self.logger.info('redreadtime = {}'.format(redreadtime))
        self.logger.info('readspeed = {}'.format(readspeed))

        ###########################################################################



        try:
            rngreen1_str = str(rngreen1)
        except KeyError:
            rngreen1_str = "null"

        try:
            rngreen2_str = str(rngreen2)
        except KeyError:
            rngreen2_str = "null"

        try:
            rngreen3_str = str(rngreen3)
        except KeyError:
            rngreen3_str = "null"

        try:
            rngreen4_str = str(rngreen4)
        except KeyError:
            rngreen4_str = "null"

        try:
            rnred1_str = str(rnred1)
        except KeyError:
            rnred1_str = "null"

        try:
            rnred2_str = str(rnred2)
        except KeyError:
            rnred2_str = "null"

        try:
            rnred3_str = str(rnred3)
        except KeyError:
            rnred3_str = "null"

        try:
            rnred4_str = str(rnred4)
        except KeyError:
            rnred4_str = "null"

        try:
            rncahk_str = str(rncahk)
        except KeyError:
            rncahk_str = "null"

        try:
            greenreadtime_str = str(greenreadtime)
        except KeyError:
            greenreadtime_str = "null"

        try:
            redreadtime_str = str(redreadtime)
        except KeyError:
            redreadtime_str = "null"

        if readspeed == "null" or readspeed == None or readspeed == '':
            readspeed_str = "null"
        else:
            readspeed_str = "'" + readspeed + "'"

        # Define query template for database stored function that executes insert/update SQL statement.

        query_template =\
            "select * from registerReadNoise(" +\
            "cast(RID as integer)," +\
            "cast(RNGREEN1 as real)," +\
            "cast(RNGREEN2 as real)," +\
            "cast(RNGREEN3 as real)," +\
            "cast(RNGREEN4 as real)," +\
            "cast(RNRED1 as real)," +\
            "cast(RNRED2 as real)," +\
            "cast(RNRED3 as real)," +\
            "cast(RNRED4 as real)," +\
            "cast(RNCAHK as real)," +\
            "cast(GREENREADSPEED as real)," +\
            "cast(REDREADSPEED as real)," +\
            "cast(READSPEED as character varying(16)));"

        # Substitute values into template for registering database record.

        rIdstr = str(input_rid)

        rep = {"RID": rIdstr}

        rep["RNGREEN1"] = rngreen1_str
        rep["RNGREEN2"] = rngreen2_str
        rep["RNGREEN3"] = rngreen3_str
        rep["RNGREEN4"] = rngreen4_str

        rep["RNRED1"] = rnred1_str
        rep["RNRED2"] = rnred2_str
        rep["RNRED3"] = rnred3_str
        rep["RNRED4"] = rnred4_str

        rep["RNCAHK"] = rncahk_str

        rep["GREENREADSPEED"] = greenreadtime_str
        rep["REDREADSPEED"] = redreadtime_str
        rep["READSPEED"] = readspeed_str

        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        query = pattern.sub(lambda m: rep[re.escape(m.group(0))], query_template)

        self.logger.info('query = {}'.format(query))


        # Execute query.

        try:
            cur.execute(query)

            try:
                for record in cur:
                    self.logger.info(record)
            except:
                    self.logger.info("Nothing returned from database stored function; continuing...")

        except (Exception, psycopg2.DatabaseError) as error:
            self.logger.info('*** Error inserting record ({}); skipping...'.format(error))
            read_noise_exit_code = 66


        # If and only if running the code for a single frame:
        # Update FITS header of corresponding 2D FITS file, if it exists.

        if self.backfill_repopulate_db_recs_cfg == 0:

            fits_filename = input_filename
            fits_filename = fits_filename.replace('L0', '2D')
            fits_filename = fits_filename.replace('.fits', '_2D.fits')

            fits_filename_exists = exists(fits_filename)
            if not fits_filename_exists:
                self.logger.info('*** File does not exist ({}); skipping...'.format(fits_filename))
            else:

                fits_obj = KPF0.from_fits(fits_filename,self.data_type)

                for ffi in lev0_ffi_exts:

                    rn = read_noise_dict[ffi]

                    if rn == None:
                        continue

                    kwd = read_noise_kwd_dict[ffi]

                    try:
                        fits_obj.header['PRIMARY'][kwd] = (rn,'Instantaneous ' + ffi + ' read noise [electrons]')
                        self.logger.info('ffi,kwd,rn = {},{},{}'.format(ffi,kwd,rn))
                    except KeyError:
                        pass

            fits_obj.header['PRIMARY']["GREENTRT"] = (greenreadtime,'GREEN chip total read time [seconds]')
            fits_obj.header['PRIMARY']["REDTRT"] = (redreadtime,'RED chip total read time [seconds]')
            fits_obj.header['PRIMARY']["READSPED"] = (readspeed,'Categorization of read speed')

            fits_obj.to_fits(fits_filename)

        return read_noise_exit_code


    def _perform(self):

        """
        Perform the following steps:
        1. Connect to pipeline-operations database
        2. Perform calculations for record(s) in the ReadNoiseDatabase table.
           a. if self.backfill_repopulate_db_recs_cfg == 0, do just for the L0 FITS file 
              specified in the recipe config file, and write results to the FITS header
              of the corresponding 2D file.  The L0 FITS file(s) must exist and MD5 checksum
              stored in datbase is not checked in this case..
           b. if self.backfill_repopulate_db_recs_cfg == 1, do for the L0 FITS files
              returned from the database query specified in the module-specific config file,
              but DO NOT write results to the FITS header of the corresponding 2D file.
              In this case, the L0 FITS file(s) must exist and have MD5 checksum matching
              that stored in L0Files database record.
        3. Disconnect from database.


        Returns exitcode:
            0 = Normal
            2 = Exception raised closing database connection
           64 = Cannot connect to database
           65 = Input file does not exist
           66 = Could not insert database record
        """
            
        read_noise_exit_code = 0


        # Get database connection parameters from environment.

        dbport = os.getenv('DBPORT')
        dbname = os.getenv('DBNAME')
        dbuser = os.getenv('DBUSER')
        dbpass = os.getenv('DBPASS')
        dbserver = os.getenv('DBSERVER')


        # Connect to database

        try:
            conn = psycopg2.connect(host=dbserver,database=dbname,port=dbport,user=dbuser,password=dbpass)
        except:
            self.logger.info('Could not connect to database...')
            read_noise_exit_code = 64
            return Arguments(read_noise_exit_code)


        # Open database cursor.

        cur = conn.cursor()


        # Select database version.

        q1 = 'SELECT version();'
        self.logger.info('q1 = {}'.format(q1))
        cur.execute(q1)
        db_version = cur.fetchone()
        self.logger.info('PostgreSQL database version = {}'.format(db_version))


        # Check database current_user.

        q2 = 'SELECT current_user;'
        self.logger.info('q2 = {}'.format(q2))
        cur.execute(q2)
        for record in cur:
            self.logger.info('record = {}'.format(record))
            pass


        ###########################################################################
        # Perform calculations for database record(s).
        ###########################################################################

        n_sigma = self.n_sigma    # Number of sigmas for outlier rejection.

        cf = self.compute_clip_corr(n_sigma)    # Correction factor to re-inflate variance.

        cf = round(cf, 3)     # Round to 3 places to the right of the decimal point, in order to to reduce stochasticity.

        self.logger.info("n_sigma,cf = {},{}".format(n_sigma,cf))


        if self.backfill_repopulate_db_recs_cfg == 0:


            # Compute read noise for a single L0 FITS file.

            read_noise_exit_code = self.computeReadNoiseForSingleL0File(self.rId,self.l0_filename,n_sigma,cf,cur)


            # Commit transaction.

            conn.commit()

        else:


            # Use self.l0_filename to get the inside-container path

            filename_match = re.match(r"(.+?)(\d\d\d\d\d\d\d\d/)(KP.+)", self.l0_filename)

            try:
                filename_path_prefix_from_config_file = filename_match.group(1)
                filename_path_date = filename_match.group(2)
                filename_only = filename_match.group(3)

                print("-------------------> filename_path_prefix_from_config_file =",filename_path_prefix_from_config_file)
                print("-------------------> filename_path_date =",filename_path_date)
                print("-------------------> filename_only =",filename_only)

            except:
                print("-------------------> No filename match found")


            # Query for all frames, up to countmax.

            countmax = 1000000

            count = 0
            
            #query_template = "select rid,filename,checksum from L0Files where dateobs > '2022-12-01' order by mjdobs limit 101;"
            query_template = self.backfill_repopulate_db_query_template

            query = query_template

            print("query to get list of files = ",query)


            self.logger.info('query = {}'.format(query))


            # Execute query.

            try:
                cur.execute(query)

            except (Exception, psycopg2.DatabaseError) as error:
                self.logger.info('*** Error querying records ({}); skipping...'.format(error))
                read_noise_exit_code = 66


            rId_list = []
            filename_list = []
            checksum_list = []
                
            for record in cur:
                self.logger.info('record = {}'.format(record))

                if record is not None:
                    rId = record[0]
                    filename = record[1]
                    checksum = record[2]

                    rId_list.append(rId)
                    filename_list.append(filename)
                    checksum_list.append(checksum)
                    
            self.logger.info('Number of files returned by database query = {}'.format(len(rId_list)))

            for i in range(0,len(rId_list)):

                rId = rId_list[i]
                filename = filename_list[i]
                checksum = checksum_list[i]


                # Replace the outside-container path with the inside-container path.

                filename_match = re.match(r"(.+?)(\d\d\d\d\d\d\d\d/)(KP.+)", filename)

                try:
                    filename_path_prefix = filename_match.group(1)
                    filename_path_date = filename_match.group(2)
                    filename_only = filename_match.group(3)

                    print("-------------------> filename_path_prefix =",filename_path_prefix)
                    print("-------------------> filename_path_date =",filename_path_date)
                    print("-------------------> filename_only =",filename_only)

                    filename = filename_path_prefix_from_config_file + filename_path_date + filename_only
                    print("-------------------> filename =",filename)

                except:
                    print("-------------------> No filename match found")
                    continue

                
                # See if file exists.

                isExist = os.path.exists(filename)

                if debug == 1:
                    print('File,existence = {},{}'.format(filename,isExist))

                if isExist == False:
                    self.logger.info('*** Error: File does not exist ({}); skipping...'.format(filename))
                    continue


                # Compute checksum and compare with database value.

                cksum = db.md5(filename)

                if debug == 1:
                    print('cksum = {}'.format(cksum))

                if cksum == checksum:
                    if debug == 1:
                        print("File checksum is correct...")
                else:
                    self.logger.info('*** Error: File checksum is incorrect ({}); skipping...'.format(filename))
                    continue


 
                self.logger.info('rId,filename = {},{}'.format(rId,filename))


                # Compute read noise for a single L0 FITS file.

                read_noise_exit_code = self.computeReadNoiseForSingleL0File(rId,filename,n_sigma,cf,cur)
                self.logger.info('read_noise_exit_code returned from method self.computeReadNoiseForSingleL0File = {}'.format(read_noise_exit_code))

                if read_noise_exit_code != 0:
                    break


                # Commit transaction.

                conn.commit()


                # Increment counter.
            
                count = count + 1

                print("count =",count)
                self.logger.info('count,countmax = {},{}'.format(count,countmax))

                if count >= countmax:
                    break;


        ###########################################################################
        ###########################################################################


        # Close database cursor and then connection.

        try:
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            self.logger.info('*** Error closing database connection ({}); skipping...'.format(error))
            read_noise_exit_code = 2
        finally:
            if conn is not None:
                conn.close()

        self.logger.info('Finished {}'.format(self.__class__.__name__))

        return Arguments(read_noise_exit_code)
