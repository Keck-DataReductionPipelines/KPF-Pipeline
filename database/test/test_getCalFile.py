#################################################################################
# Get nearest-in-time master file in KPF pipeline-operations database for given date.
#################################################################################

import os
import sys
import glob
import psycopg2
import re
from astropy.io import fits
import hashlib

exitcode = 0

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


# Get date from command-line argument.

datearg = (sys.argv)[1]

print("datearg =",datearg)

master_file_path = '/masters' + '/' + datearg + '/' + '*.fits'
print("master_file_path =",master_file_path)


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


# Query database.

query_template =\
    "select * from getCalFile(" +\
    "cast(OBSDATE as date)," +\
    "cast(LEVEL as smallint)," +\
    "cast('CALTYPE' as character varying(32))," +\
    "cast('OBJECT' as character varying(32))) as " +\
    "(cId integer," +\
    " level smallint," +\
    " caltype varchar(32)," +\
    " object varchar(32)," +\
    " filename varchar(255)," +\
    " checksum varchar(32)," +\
    " infobits integer," +\
    " startDate date);"

obsdate = "'" + datearg[0:4] + "-" + datearg[4:6] + "-" + datearg[6:8] + "'"
level = 0
levelstr = str(level)
caltype = 'Flat'
object = 'Autocal-flat-all'

rep = {"OBSDATE": obsdate,
       "LEVEL": levelstr,
       "CALTYPE": caltype,
       "OBJECT": object}

rep = dict((re.escape(k), v) for k, v in rep.items()) 
pattern = re.compile("|".join(rep.keys()))
query = pattern.sub(lambda m: rep[re.escape(m.group(0))], query_template)

print("query =",query)

cur.execute(query)
record = cur.fetchone()

cId = record[0]
filename = '/' + record[4]        # docker run has -v /data/kpf/masters:/masters
checksum = record[5]

print("cId =",cId)
print("filename =",filename)
print("checksum =",checksum)


# See if file exists.

isExist = os.path.exists(filename)
print("File existence =",isExist)


# Compute checksum and compare with database value.

cksum = md5(filename)
print("Computed checksum =",cksum)

if cksum == checksum:
    print("File checksum is correct...")
else:
    print("*** Error: File checksum is incorrect; quitting...")
    exitcode = 64


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
