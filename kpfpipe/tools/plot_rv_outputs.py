import numpy as np
import argparse
import glob

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time

# grab output directory from command line
parser = argparse.ArgumentParser(description='Plot test recipe outputs.')
parser.add_argument('output_dir', help='relative path to pipeline outputs')
args = parser.parse_args()

# find all level 2 fits files in user-specified output dir
level2_files = glob.glob('{}/*L2.fits'.format(args.output_dir))

n_files = len(level2_files)
rvs = np.empty(n_files)
epochs = np.empty(n_files)
units = []

# grab RVs from each file
for i, rv_file in enumerate(level2_files):


    hdul = fits.open(rv_file)
    rv = hdul[-1].header['CCF-RVC']

    rv_comment = hdul[-1].header.comments['CCF-RVC'] 

    if 'km/s' in rv_comment:
        units.append('km/s')
    elif 'cm/s' in rv_comment:
        units.append('cm/s')
    else:
        units.append('m/s')

    epoch = hdul[-1].header['CCFJDSUM']

    rvs[i] = rv
    epochs[i] = epoch
    hdul.close()

for i in range(n_files - 1):
    assert units[i] == units[i+1], "units between files differ."

delta_time_day = epochs - epochs[0]
delta_time_min = delta_time_day * 60 * 24

# plot the RVs
fig, ax = plt.subplots(dpi=250, figsize=(5,5))
plt.scatter(
        delta_time_min, rvs, s=25, fc='white', ec='k', label='RV std: {:.2f} {}'.format(np.std(rvs), units[0])
)
plt.ylabel('RV [{}]'.format(units[0]))
plt.xlabel('CCF JD Sum (relative to first image) [min]')
plt.legend()
plt.savefig('{}/rv_summary_plot.png'.format(args.output_dir))
