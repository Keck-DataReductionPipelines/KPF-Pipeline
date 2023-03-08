#################################################################################
# Register master files in KPF pipeline-operations database for given date.
#################################################################################

import os
import sys
import glob
import psycopg2
import re
from astropy.io import fits
import hashlib
from datetime import datetime, timezone
import time

exitcode = 0

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()



# Get date from command-line argument.

datearg = (sys.argv)[1]

print("datearg =",datearg)

search_path = '/masters' + '/' + datearg + '/' + '*.fits'
print("search_path =",search_path)
input_files = glob.glob(search_path)

master_files = []
for file in input_files:
    if '_norect' in file: continue                  # Omit *_norect.fits.
    print("file =",file)
    master_files.append(file)


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
    print("Could not connect to database...")


# Open database cursor.

cur = conn.cursor()


# Select database version.

cur.execute('SELECT version()')
db_version = cur.fetchone()
print('PostgreSQL database version: ', db_version)


# Check database current_user.

q1 = 'SELECT current_user;'
print(q1)
cur.execute(q1)
for record in cur:
    print(record)


# Loop over L0 FITS master files.
# Unfortunately in the master_bias_framework.py, etc., the FITS keyword for the output is called IMTYPE instead of CALTYPE.
# Build insert/update SQL statement.
# Insert/update record in CalFiles table.

query_template =\
    "select * from registerCalFile(" +\
    "cast(STARTDATE as date)," +\
    "cast(ENDDATE as date)," +\
    "cast(LEVEL as smallint)," +\
    "cast('IMTYPE' as character varying(32))," +\
    "cast('TARGOBJ' as character varying(32))," +\
    "cast(NFRAMES as smallint)," +\
    "cast(MINMJD as double precision)," +\
    "cast(MAXMJD as double precision)," +\
    "cast(INFOBITS as integer)," +\
    "cast('FILENAME' as character varying(255))," +\
    "cast('CHECKSUM' as character varying(32))," +\
    "cast(FILESTATUS as smallint)," +\
    "cast('registerCalFilesForDate.py' as character varying(32))," +\
    "cast('CREATED' as timestamp without time zone)," +\
    "cast('Automated pipeline process.' as character varying(255)));"

startdate = "'" + datearg[0:4] + "-" + datearg[4:6] + "-" + datearg[6:8] + "'"
enddate = "null"
filestatus = 1
filestatusstr = str(filestatus)

for fits_file in master_files:

    print("-----> fits_file =",fits_file)


    # In the case of *L1.fits and *L2.fits files, certain keywords are not in the FITS header,
    # so we either parse them from the filename or infer them.
    
    filename_match = re.match(r".+_master_(.+)_L[1:2]\.fits", fits_file)

    try:
        filename_caltype = filename_match.group(1)
        #print("-----1-----> fn_caltype =",filename_caltype)

        filename_match = re.match(r"(.+?)_(.+)", filename_caltype)

        try:
            filename_caltype = filename_match.group(1)
            filename_object = filename_match.group(2)

            #print("-------2--------> fn_caltype =",filename_caltype)
            #print("-------2--------> fn_object =",filename_object)

        except:
            #print("-------2--------> No filename match found")

            if filename_caltype == 'bias':
                filename_object = "autocal-bias"
            elif filename_caltype == 'dark':
                filename_object = "autocal-dark"
            elif filename_caltype == 'flat':
                filename_object = "autocal-flat-all"
            
        
    except:
        #print("-----1-----> No filename match found")

        pass

    try:
        if re.match(r".+_master_bias", fits_file):
            filename_caltype = 'bias'
        elif re.match(r".+_master_dark", fits_file):
            filename_caltype = 'dark'
        elif re.match(r".+_master_flat", fits_file):
            filename_caltype = 'flat'
    except:
        pass

    try:
        if filename_caltype == 'bias':
            filename_object = "autocal-bias"
        elif filename_caltype == 'dark':
            filename_object = "autocal-dark"
        elif filename_caltype == 'flat':
            filename_object = "autocal-flat-all"
    except:
        pass
    
    try:
        print("---------------------------------> fn_caltype =",filename_caltype)
    except:
        print("---------------------------------> fn_caltype not defined...")

    try:
        print("---------------------------------> fn_object =",filename_object)
    except:
        print("---------------------------------> fn_object not defined...")

    level = 0
    if '_L1.fits' in fits_file: level = 1
    elif '_L2.fits' in fits_file: level = 2
    levelstr = str(level)

    rep = {"STARTDATE": startdate,
           "ENDDATE": enddate,
           "LEVEL": levelstr,
           "FILESTATUS": filestatusstr}

    rep["FILENAME"] = fits_file[1:]      # Remove leading slash to make it a relative path.
    
    cksum = md5(fits_file)
    rep["CHECKSUM"] = cksum
    
    hdul = fits.open(fits_file)

    ext_list = [0,3,3,3,3,[3,4,5],3]
    kwd_list = ["IMTYPE","TARGOBJ","NFRAMES","MINMJD","MAXMJD","INFOBITS","CREATED"]
    for kwd,ext in zip(kwd_list,ext_list):
        
        print("kwd =",kwd)
        print("ext =",ext)
        
        try:
            if kwd == "INFOBITS":
                try:
                    val1 = hdul[ext[0]].header[kwd]
                except:
                    print("Exception raised: cannot find FITS keyword INFOBITS in extension 4; continuing...")
                    val1 = -1
                try:
                    val2 = hdul[ext[1]].header[kwd]
                except:
                    print("Exception raised: cannot find FITS keyword INFOBITS in extension 5; continuing...")
                    val2 = -1
                try:
                    val3 = hdul[ext[2]].header[kwd]
                except:
                    print("Exception raised: cannot find FITS keyword INFOBITS in extension 6; continuing...")
                    val3 = -1
                val = -1
                if val1 >= 0 and val2 >= 0: val = val1 | val2
                if val3 >= 0 and val >=0: val = val | val3
                if val < 0: val = "null"
            else:
                val = hdul[ext].header[kwd]
            
            if kwd == "CREATED":
                val = val.replace("T"," ")
                val = val.replace("Z","")

        except:
            if kwd == "TARGOBJ":
                if filename_object is not None:
                    val = filename_object
                else:
                    print("Exception raised: cannot find FITS keyword =", "OBJECT")
                    exit(64)
            elif kwd == "NFRAMES" or kwd == "MINMJD" or kwd == "MAXMJD":
                val = "null"
            elif kwd == "CREATED":
                datetimenow = datetime.now(timezone.utc)
                now_val = datetimenow.strftime("%Y-%m-%d %H:%M:%S")
                print("now() =", now_val)
                ti_c = os.path.getctime(fits_file)
                c_ti = time.ctime(ti_c)
                t_obj = time.strptime(c_ti)
                val = time.strftime("%Y-%m-%d %H:%M:%S", t_obj)
                print("Since no FITS keyword CREATED exists use file timestamp =", val)
            else:
                print("Exception raised: cannot find FITS keyword =", kwd)
                exit(64)
        print("fits_file,kwd,val =",fits_file,kwd,val)

        if type(val) != "str":
            valstr = str(val)
        else:
            valstr = val
        rep[kwd] = valstr

    hdul.close()

    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    query = pattern.sub(lambda m: rep[re.escape(m.group(0))], query_template)

    print("query =",query)

    cur.execute(query)
    for record in cur:
        print(record)


# Commit transaction.

conn.commit()


# Close database cursor and then connection.

try:
    cur.close()
except (Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if conn is not None:
        conn.close()
        print('Database connection closed.')

        
# Termination.

print("Terminating with exitcode =",exitcode)

exit(exitcode)
