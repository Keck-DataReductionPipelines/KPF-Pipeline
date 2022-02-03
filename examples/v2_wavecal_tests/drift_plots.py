import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.animation import FuncAnimation
import glob

from modules.wavelength_cal.src.alg import calcdrift_polysolution

"""
First, plot our wavelength solutions compared to the NEID team's
"""

path_prefix = '/data/KPF-Pipeline-TestData/DRP_V2_Testing/NEID_HD73344/output_wavecal/'
file_dirs = np.sort(glob.glob('{}*/'.format(path_prefix)))[0:13]
obs_nums = [f.split('/')[-2] for f in file_dirs]

wlpixelfile1 = '{}/Etalon_20210221T{}.npy'.format(file_dirs[0], obs_nums[0])

obstimes = np.zeros(len(file_dirs))
first_file = fits.open('{}neidL1_20210221T{}_L1_wave.fits'.format(path_prefix, obs_nums[0]))
obstimes[0] = first_file['PRIMARY'].header['OBSJD']

for j, obnum in enumerate(obs_nums):

    # read in our solution
    our_file = fits.open('{}neidL1_20210221T{}_L1_wave.fits'.format(path_prefix, obs_nums[j]))
    our_wls = our_file['CALWAVE'].data
    obstimes[j] = our_file['PRIMARY'].header['OBSJD']

    # read in NEID team solution
    neid_file = fits.open('/data/KPF-Pipeline-TestData/DRP_V2_Testing/NEID_HD73344/L1/neidL1_20210221T{}.fits'.format(obs_nums[j]))
    neid_wls = neid_file['CALWAVE'].data

    # make a difference plot
    fig, ax = plt.subplots(2, 1, figsize=(15,7))
    plt.subplots_adjust(hspace=0)
    for i in np.arange(len(our_wls)):
        if np.median(our_wls[i]) > 0:
            ax[0].plot(our_wls[i] - neid_wls[i], color='grey', alpha=0.5)
            pixel_size = neid_wls[i][1:] - neid_wls[i][:-1]
            ax[1].plot((our_wls[i][:-1] - neid_wls[i][:-1]) / pixel_size, color='grey', alpha=0.5)

    ax[1].set_xlabel('pixel')
    ax[0].set_ylabel('Our WLS - NEID WLS [$\\rm \AA$]')
    ax[1].set_ylabel('Our WLS - NEID WLS [pix]')

    plt.savefig('{}/us_vs_NEID.png'.format(file_dirs[j]), dpi=250)

"""
Next, make a gif showing the order-by-order drift over the three exposures
"""

fig, ax = plt.subplots(dpi=250)
ax.axhline(0, ls='--', color='k', alpha=0.5)
drift_line = plt.plot([0], [0], 'ro', ls='--', color='k')
plt.xlim(55, 86)
plt.ylim(-1000, 1000)
plt.xlabel('order')
plt.ylabel('drift [cm/s]')
date_text = plt.text(78, 900, '')
avg_drift = np.zeros(len(file_dirs) - 1)

def AnimationFunction(i):

    # calculate drift from saved wl-pixel files
    wlpixelfile2 = '{}Etalon_20210221T{}.npy'.format(file_dirs[i + 1], obs_nums[i + 1])

    drift_all_orders = calcdrift_polysolution(wlpixelfile1, wlpixelfile2)
    avg_drift[i] = np.mean(drift_all_orders[:,1])

    drift_line[0].set_data(drift_all_orders[:,0], drift_all_orders[:,1])
    date_text.set_text('$\Delta$ time = {:.2f} hr'.format((obstimes[i+1] - obstimes[0]) * 24))

animation = FuncAnimation(fig, AnimationFunction, frames=len(file_dirs) - 1, interval=300)
animation.save('{}drift.gif'.format(path_prefix))

# """
# Finally, plot the average drift in cm/s over the course of all exposures
# """

plt.figure()
plt.plot((obstimes[1:] - obstimes[0]) * 24, avg_drift, 'ro', ls='--', color='k')
plt.xlabel('$\Delta$ time [hr]')
plt.ylabel('order-averaged drift [cm/s]')
plt.savefig('{}drift.png'.format(path_prefix), dpi=250)


