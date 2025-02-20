#################################################################################
# Inside a Docker container, this python script will do the following:
# 1. Clean out all files and directories in /masters/<yyyyymmdd>
# 2. Remove records in the CalFiles table of the KPF operations database
#    that are associated with startdate = <yyyymmdd>.
#################################################################################

import os
import sys
import psycopg2

import database.modules.utils.kpf_db as db

exitcode = 0


# Process identification.

iam = "database/scripts/cleanupMastersOnDiskAndDatabaseForDate.py"
iam_version = "1.0"
print("iam =",iam)
print("iam_version =",iam_version)


# Get date from command-line argument.

datearg = (sys.argv)[1]

print("datearg =",datearg)


########################################################################################################
# Functions.
########################################################################################################

def remove_masters_for_date(datearg):


    # Specify the pattern for the files and directories you want to delete with rm -rf <file_pattern>

    file_pattern = '/masters' + '/' + datearg + '/*'
    print("Files to remove =",file_pattern)

    rm_cmd = 'rm -rf ' + file_pattern
    print('rm_cmd =',rm_cmd)
    exitcode_from_rm = os.system(rm_cmd)
    print('exitcode_from_rm =',exitcode_from_rm)

    return exitcode_from_rm


########################################################################################################
# Main program.
########################################################################################################


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
    exitcode = 64


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


# Clean out database records for given startdate.

query = "DELETE from CalFiles where startdate = '" + datearg + "';"
print(query)

cur.execute(query)


try:
    for record in cur:
        print(record)
except:
        print("Nothing returned from database query executed; continuing...")


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


# If the database record deletion was successful, then remove the files from disk.

if exitcode == 0:
    exitcode = remove_masters_for_date(datearg)


# Termination.

print("Terminating with exitcode =",exitcode)


exit(exitcode)
