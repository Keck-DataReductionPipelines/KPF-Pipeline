import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

#######################################
# LFC, running for first time
# filename = 'neidL1_20220127T135910'
# savedir = 'LFC_noprev'
# ylims_pix = [-.15, .15]

# LFC, not for first time
# filename = 'neidL1_20220130T135641'
# savedir = 'LFC_yesprev'
# ylims_pix = [-.15, .15]

# ThAr
filename = 'neidL1_20220126T235959'
savedir = 'ThAr'
ylims_pix = [-1.5, 1.5]
#######################################


output_path = '/data/KPF-Pipeline-TestData/DRP_V2_Testing/NEID-cals/output_wavecal/{}'.format(savedir)

# read in our solution
our_file = fits.open('{}/{}_L1_wave.fits'.format(output_path, filename))
our_wls = our_file['CALWAVE'].data

# read in NEID team solution
neid_file = fits.open('/data/KPF-Pipeline-TestData/DRP_V2_Testing/NEID-cals/{}.fits'.format(filename))
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
ax[1].set_ylim(*ylims_pix)
plt.savefig('{}/us_vs_NEID.png'.format(output_path), dpi=250)