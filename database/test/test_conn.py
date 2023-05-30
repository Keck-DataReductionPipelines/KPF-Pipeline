#################################################################################
# Test code to verify database connectivity.
#################################################################################

import psycopg2
import os


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


# Select multiple records from a pipeline-operations table.

cur.execute("select * from calfiles limit 2;")
for record in cur:
    print(record)
    startdate = record[4]
    print("startdate =",startdate)
    enddate = record[5]
    print("startdate =",enddate)


# Select multiple records from a database system table.

cur.execute("select * from pg_user;")
for record in cur:
    print(record)


# Close database cursor and then connection.

try:
    cur.close()
except (Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if conn is not None:
        conn.close()
        print('Database connection closed.')
