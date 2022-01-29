import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

filename = 'neidL1_20220127T135910'

# read in our solution
our_file = fits.open('/code/KPF-Pipeline/outputs/{}_L1_wave.fits'.format(filename))
our_wls = our_file['SCIWAVE'].data

# read in NEID team solution
neid_file = fits.open('/data/KPF-Pipeline-TestData/DRP_V2_Testing/NEID-cals/{}.fits'.format(filename))
neid_wls = neid_file['SCIWAVE'].data

# make a difference plot
fig, ax = plt.subplots(2, 1, figsize=(15,5))
for i in np.arange(len(our_wls)):
    if np.median(our_wls[i]) > 0:
        ax[0].plot(our_wls[i] - neid_wls[i], color='grey', alpha=0.5)
        pixel_size = neid_wls[i][1:] - neid_wls[i][:-1]
        ax[1].plot((our_wls[i][:-1] - neid_wls[i][:-1]) / pixel_size, color='grey', alpha=0.5)

ax[1].set_xlabel('pixel')
ax[0].set_xlabel('pixel')
ax[0].set_ylabel('Our WLS - NEID Team WLS [ang]')
ax[1].set_ylabel('Our WLS - NEID Team WLS [pix]')
plt.tight_layout()
plt.savefig('/code/KPF-Pipeline/outputs/LFC_vs_NEID.png', dpi=250)