####################################################################################################################
# Generate smooth lamp patterns for master flat.
#
# For GREEN_CCD and RED_CCD make a fixed lamp pattern made by the 7x7-pixel local median of all
# stacked-image data within the orderlet mask from a specific observation date (e.g., 100 Flatlamp
# frames, 30-second exposures each, were acquired on 20230628).  The fixed smooth lamp pattern enables
# the flat-field correction to remove dust and debris signatures on the optics of the instrument and
# telescope.  The local median filtering minimizes undesirable effects at the orderlet edges.
#
# The fast method does NOT properly handle NaNs (https://github.com/scipy/scipy/issues/4800)
# The slow method works fine, but takes about two hours to complete.
####################################################################################################################

import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter

fname_order_mask = "order_mask_G4-3_2_R4-3_2_20230717.fits"
fname_stack_average = "kpf_20230628_master_flat.fits"
fname_smooth_lamp = "kpf_20230628_smooth_lamp_made20230717.fits"

hdul_order_mask = fits.open(fname_order_mask)
hdul_stack_average = fits.open(fname_stack_average)

ffis = ["GREEN_CCD","RED_CCD"]

hdu_list = []
fast_method = False    # Always set to False.
window = 7             # 7x7-pixel local median filter

empty_data = None
hdu_list.append(fits.PrimaryHDU(empty_data))

for ffi in ffis:
    ffi_stack = ffi + "_STACK"
    data_order_mask = hdul_order_mask[ffi].data
    data_stack_average = hdul_stack_average[ffi_stack].data

    np_om_ffi = np.array(np.rint(data_order_mask)).astype(int)           # Ensure rounding to nearest integer.
    data = np.where(np_om_ffi > 0,data_stack_average,np.nan)             # Set to NaN outside orderlet regions.

    # Local median filter data.

    if fast_method:
        median_image = median_filter(data, size=window)
    else:
        hwin = int((window - 1) / 2)
        ny = 4080
        nx = 4080
        median_image = np.zeros(shape=(ny, nx))
        median_image[:] = np.nan
        for i in range(0,ny):
            for j in range(0,nx):

                if np.isnan(data[i, j]): continue

                data_list = []
                for ii in range(i-hwin,i+hwin+1):
                    if ((ii < 0) or (ii >= ny)): continue
                    for jj in range(j-hwin,j+hwin+1):
                        if ((jj < 0) or (jj >= ny)): continue

                        datum = data[ii, jj]

                        #print(datum)

                        if not np.isnan(datum):
                            data_list.append(datum)

                if len(data_list) > 0:
                    med = np.median(np.array(data_list))
                    median_image[i, j] = med
                    #print(j,i,med)

    hdu = fits.ImageHDU(median_image)
    hdu.scale(type='float32')
    hdu.header['EXTNAME'] = ffi
    hdu.header['WINDOW'] = (window, "Size of median-filter kernel")
    hdu_list.append(hdu)

hdu = fits.HDUList(hdu_list)
hdu.writeto(fname_smooth_lamp,overwrite=True,checksum=True)

