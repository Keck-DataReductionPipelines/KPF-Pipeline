import os
import psycopg2
import re
import hashlib

max_cal_file_age = '1000 days'

def md5(fname):
    hash_md5 = hashlib.md5()

    try:
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        print("*** Error: Cannot open file =",fname,"; quitting...")
        exit(65)

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
    """

    def __init__(self):

        self.exit_code = 0
        self.filename = None
        self.conn = None

        # Get database connection parameters from environment.

        dbport = os.getenv('DBPORT')
        dbname = os.getenv('DBNAME')
        dbuser = os.getenv('DBUSER')
        dbpass = os.getenv('DBPASS')
        dbserver = os.getenv('DBSERVER')

        # Connect to database

        try:
            self.conn = psycopg2.connect(host=dbserver,database=dbname,port=dbport,user=dbuser,password=dbpass)
        except:
            print("Could not connect to database...")
            self.exit_code = 64
            return

        # Open database cursor.

        self.cur = self.conn.cursor()

        # Select database version.

        q1 = 'SELECT version();'
        print('q1 = {}'.format(q1))
        self.cur.execute(q1)
        db_version = self.cur.fetchone()
        print('PostgreSQL database version = {}'.format(db_version))

        # Check database current_user.

        q2 = 'SELECT current_user;'
        print('q2 = {}'.format(q2))
        self.cur.execute(q2)
        for record in self.cur:
            print('record = {}'.format(record))

    def get_nearest_master_file(self,obs_date,cal_file_level,contentbitmask,cal_type_pair):

        '''
        Get nearest master file for the specified set of input parameters.
        '''

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

        print('----> cal_file_level = {}'.format(cal_file_level))
        print('----> contentbitmask = {}'.format(contentbitmask))
        print('----> cal_type_pair = {}'.format(cal_type_pair))

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

        print('query = {}'.format(query))

        self.cur.execute(query)
        record = self.cur.fetchone()

        if record is not None:
            cId = record[0]
            db_level = record[1]
            db_cal_type = record[2]
            db_object = record[3]
            filename = '/' + record[4]         # docker run has -v /data/kpf/masters:/masters
            checksum = record[5]
            infobits = record[6]

            print('cId = {}'.format(cId))
            print('filename = {}'.format(filename))
            print('checksum = {}'.format(checksum))


            # See if file exists.

            isExist = os.path.exists(filename)
            print('File existence = {}'.format(isExist))

            if isExist is True:
                print("File exists...")
            else:
                print("*** Error: File does not exist; quitting...")
                self.exit_code = 65
                return


            # Compute checksum and compare with database value.

            if isExist is True:
                cksum = md5(filename)
                print('cksum = {}'.format(cksum))

                if cksum == checksum:
                    print("File checksum is correct ({})...".format(filename))
                    self.filename = filename
                else:
                    print("*** Error: File checksum is incorrect ({}); quitting...".format(filename))
                    self.exit_code = 66
                    return


    def close(self):

        '''
        Close database cursor and then connection.
        '''

        try:
            self.cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            self.exit_code = 2
        finally:
            if self.conn is not None:
                self.conn.close()
                print('Database connection closed.')
