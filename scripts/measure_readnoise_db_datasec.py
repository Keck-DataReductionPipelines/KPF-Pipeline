#################################################################################################
# measure_readnoise_db_datasec.py      Oct 17, 2023         Russ Laher (laher@ipac.caltech.edu)
#
# Directly measure read noise in KPF bias frames for all read-out channels in
# GREEN and RED chips, using the difference-image technique, on DATASEC image regions
# (meaning not in overscan regions).
#
# Var(X-Y) = Var(X) + Var(Y).  Since Var(X) = Var(Y), Sigma(X) = Sigma(Y) = Sigma(X-Y)/sqrt(2).
#
# Compute statistics with outlier rejection (n-sigma data-trimming), ignoring NaNs.
#################################################################################################

import os
import math
import numpy as np
import numpy.ma as ma
from astropy.io import fits
import psycopg2
import hashlib

debug = 0
countmax = 1000000

#
# Methods
#

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

def compute_clip_corr(n_sigma):

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

#
# Statistics with outlier rejection (n-sigma data-trimming), ignoring NaNs, across all data array dimensions.
#
def statistics_with_clipping(data_array,n_sigma,cf):

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

    return avg,var,cnt

def compute_read_noise(ext,fname1_input,fname2_input,n_sigma,cf):

    """
    Compute read noise for given read-out channel.
    """

    hdul1_input = fits.open(fname1_input)
    hdul2_input = fits.open(fname2_input)

    try:
        data1_input = hdul1_input[ext].data
    except:
        if debug == 1:
            print("FITS extension {} not found in header ({}); skipping...".\
                format(ext,fname1_input))
        return

    try:
        data2_input = hdul2_input[ext].data
    except:
        if debug == 1:
            print("FITS extension {} not found in header ({}); skipping...".\
                format(ext,fname2_input))
        return

    naxis1 = hdul1_input[ext].header["NAXIS1"]
    naxis2 = hdul1_input[ext].header["NAXIS2"]

    if debug == 1:
        print("Image 1: naxis1,naxis2 = {},{}".\
                format(naxis1,naxis2))

    nx1 = (np.shape(data1_input))[0]
    ny1 = (np.shape(data1_input))[1]

    nx2 = (np.shape(data2_input))[0]
    ny2 = (np.shape(data2_input))[1]

    if nx1 != nx2 or ny1 != ny2:
        if debug == 1:
            print("Data shapes not same ({},{},{}); skipping...".\
                format(ext,fname1_input,fname2_input))
        return

    if debug == 1:
        print("naxis1,naxis2 = {},{}".\
                format(naxis1,naxis2))

                
    nxstart = 100
    nxend = 1900
    nystart = 100
        
    if naxis2 >= 4000:
        nyend = 3900
    else:
        nyend = 1900

    data1 = np.array(data1_input[nystart:nyend,nxstart:nxend])
    data2 = np.array(data2_input[nystart:nyend,nxstart:nxend])


    data = data1 - data2
    #data = data2                 # Uncomment this line and comment above line to test less robust single-image method.

    #
    # Statistics without outlier rejection.
    #
    #    mean = np.mean(data)
    #    var = np.var(data)
    #    min = np.min(data)
    #    max = np.max(data)

    #
    # Statistics with outlier rejection (n-sigma data-trimming), ignoring NaNs.
    #

    mean,var,cnt = statistics_with_clipping(data,n_sigma,cf)

    if debug == 1:
        print("cnt = {}".\
            format(cnt))

    try:
        gain1 = hdul1_input[ext].header["CCDGAIN"]
    except:
        print("*** Error: GAIN keyword not found in header ({}); quitting...".\
            format(fname1_input))
        exit(1)

    gain2 = hdul2_input[ext].header["CCDGAIN"]

    gain = 0.5 * (gain1 + gain2)

    # The divisor of 2.0 inside the square root accounts for the
    # image-differencing of two independent bias frames.

    read_noise = gain * math.sqrt(var / 2.0) / (2**16)
    #read_noise = gain * math.sqrt(var) / (2**16)           # Uncomment this line and comment above line to test less robust single-image method.

    if debug == 1:
        print("extname,Gain (electrons/DN),Read noise (electrons) = {},{},{}".\
            format(ext,gain,read_noise))

    return read_noise

def main():

    """
    Main program.
    """

    n_sigma = 3.0    # Number of sigmas for outlier rejection.

    cf = compute_clip_corr(n_sigma)

    exts_list = ['GREEN_AMP1','GREEN_AMP2','GREEN_AMP3','GREEN_AMP4','RED_AMP1','RED_AMP2','RED_AMP3','RED_AMP4']


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


    # Query for bias frames.

    mjdobs_list = []
    filename_list = []
    count = 0

    query = "select mjdobs,filename,checksum from l0files where imtype='Bias' and dateobs > '2022-12-01' order by mjdobs;"

    cur.execute(query)
    for record in cur:

        if record is not None:
            mjdobs = record[0]
            filename = record[1]
            checksum = record[2]


            # See if file exists.

            isExist = os.path.exists(filename)

            if isExist == False:
                continue

            if debug == 1:
                print('File,existence = {},{}'.format(filename,isExist))


            # Compute checksum and compare with database value.

            cksum = md5(filename)

            if debug == 1:
                print('cksum = {}'.format(cksum))

            if cksum == checksum:
                if debug == 1:
                    print("File checksum is correct...")
            else:
                print("*** Error: File checksum is incorrect; skipping...")
                continue

            count = count + 1

            print("count =",count)

            mjdobs_list.append(mjdobs)
            filename_list.append(filename)

            if count >= countmax:
                break;


    # Close database cursor and then connection.

    try:
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        exit_code = 2
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')

    n = len(filename_list)

    if n % 2 == 1:             # Ensure pairs of bias frames to difference.
        n = n - 1

    print("n = ",n)

    outfile = open('read_noise_datasec.txt', 'w')

    h = "mjdobs,GREEN_AMP1,GREEN_AMP2,GREEN_AMP3,GREEN_AMP4,RED_AMP1,RED_AMP2,RED_AMP3,RED_AMP4"
    print(h)
    outfile.write(h + "\n")

    for i in range(0,n-1,2):

        print("i = ",i)

        mjdobs = 0.5 * (mjdobs_list[i+1] + mjdobs_list[i])

        fits1 = filename_list[i]
        fits2 = filename_list[i+1]

        print(fits1,fits2)

        s = ""
        for extname in exts_list:

            read_noise = compute_read_noise(extname,fits1,fits2,n_sigma,cf)

            if read_noise is not None:
                s = s + "," + str(read_noise)
            else:
                s = s + "," + "null"

        s = str(mjdobs) + s

        print(s)
        outfile.write(s + "\n")

    outfile.close()


if __name__ == "__main__":
    exit_code = 0

    main()
    exit(exit_code)
