#################################################################################
# Inside a Docker container, this python script will do the following:
# 1. Clean out all files and directories in /masters/<yyyyymmdd>.  The implemented
#    function also removes top-level directory /masters/<yyyyymmdd>.
# 2. Remove records in the CalFiles table of the KPF operations database
#    that are associated with startdate = <yyyymmdd>.
#################################################################################

import os
import sys
import psycopg2
import shutil
import re


import database.modules.utils.kpf_db as db

exitcode = 0


# Process identification.

iam = "database/scripts/cleanupMastersOnDiskAndDatabaseForDate.py"
iam_version = "1.0"
print("iam =",iam)
print("iam_version =",iam_version)


# Get date from command-line argument.

datearg = (sys.argv)[1]


########################################################################################################
# Functions.
########################################################################################################

def is_only_numbers(input_string):
    """Checks if a string contains only digits."""
    return bool(re.match(r'^\d+$', input_string))

def remove_all(directory_path):
    try:
        shutil.rmtree(directory_path)
        print(f"Successfully removed: {directory_path}")
    except FileNotFoundError:
        print(f"Error: Directory not found: {directory_path}")
    except OSError as e:
        print(f"Error removing {directory_path}: {e}")


########################################################################################################
# Main program.
########################################################################################################

if is_only_numbers(datearg) and len(datearg) == 8:
    print("datearg =",datearg)
else:
    print("*** Error: datearg is not yyyymmdd format; quitting...")
    print("datearg = [",datearg,"]")
    exitcode = 64
    exit(exitcode)


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
    exit(exitcode)


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
    directory_to_remove = "/masters/" + datearg
    remove_all(directory_to_remove)


# Termination.

print("Terminating with exitcode =",exitcode)


exit(exitcode)
