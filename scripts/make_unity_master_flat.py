####################################################################################################################
# Generate unity master flat for GREEN_CCD and RED_CCD.
####################################################################################################################

import numpy as np
from astropy.io import fits

fname_master_flat = "kpf_20230626_master_flat.fits"
fname_unity_master_flat = "kpf_20230626_unity_master_flat.fits"

hdul_master_flat = fits.open(fname_master_flat)

ffis = ["GREEN_CCD","RED_CCD"]

ny = 4080
nx = 4080
unity_image = np.zeros(shape=(ny, nx))
unity_image[:] = 1.0                                             # Initialize 2-D array of ones

for ffi in ffis:
    ffi_flat = ffi
    data_master_flat = hdul_master_flat[ffi_flat].data

    data = data_master_flat
    print("ffi,data[13,2077],data[12,2077],data[11,2077] = ",ffi,data[13,2077],data[12,2077],data[11,2077])

    hdul_master_flat[ffi].data = unity_image

hdul_master_flat.writeto(fname_unity_master_flat,overwrite=True,checksum=True)

