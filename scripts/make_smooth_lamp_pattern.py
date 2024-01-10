####################################################################################################################
# Generate smooth lamp patterns for master flat.
#
# For GREEN_CCD and RED_CCD make a fixed lamp pattern made by the sliding-window clipped mean of all
# stacked-image data within the orderlet mask from a specific observation date (e.g., 100 Flatlamp
# frames, 30-second exposures each, were acquired on 20230628).  The fixed smooth lamp pattern enables
# the flat-field correction to remove time-evolving dust and debris signatures on the optics of the
# instrument and telescope.  A smoothing kernel 200-pixels wide (along dispersion dimension) by
# 1-pixel high (along cross-dispersion dimension) is used for computing the clipped mean, with
# 3-sigma, double-sided outlier rejection.  The kernel is centered on the pixel of interest.
#
# The implemented method is slow and takes many hours to complete.
####################################################################################################################

import numpy as np
import numpy.ma as ma
from astropy.io import fits

#fname_order_mask = "kpf_20230716_order_mask_untrimmed_made20230719.fits"

# Used to make smooth lamp pattern for 20230624 through 20230730.
#fname_stack_average = "kpf_20230628_master_flat.fits"
#fname_smooth_lamp = kpf_20230628_smooth_lamp_made20230803_float32.fits

# Used to make smooth lamp pattern for 20230801 and onward.
#fname_stack_average = "kpf_20230804_master_flat.fits"
#fname_smooth_lamp = "kpf_20230804_smooth_lamp_made20230808_float32.fits"

# Used to make smooth lamp pattern for 20230622 and before.
#fname_stack_average = "kpf_20230619_master_flat.fits"
#fname_smooth_lamp = "kpf_20230619_smooth_lamp_made20230817_float32.fits"

# Used to make smooth lamp pattern for 20230919.
fname_stack_average = "kpf_20230919_master_flat.fits"
fname_smooth_lamp = "kpf_20230919_smooth_lamp_made20240105_float32.fits"

#hdul_order_mask = fits.open(fname_order_mask)
hdul_stack_average = fits.open(fname_stack_average)

ffis = ["GREEN_CCD","RED_CCD"]

hdu_list = []
x_window = 200         # Approximately along dispersion dimension.
y_window = 1           # Approximately along cross-dispersion dimension.
n_sigma = 3            # 3-sigma, double-sided outlier rejection

empty_data = None
hdu_list.append(fits.PrimaryHDU(empty_data))

for ffi in ffis:
    ffi_stack = ffi + "_STACK"
    #data_order_mask = hdul_order_mask[ffi].data
    data_stack_average = hdul_stack_average[ffi_stack].data

    #np_om_ffi = np.array(np.rint(data_order_mask)).astype(int)           # Ensure rounding to nearest integer.
    #data = np.where(np_om_ffi > 0,data_stack_average,np.nan)             # Set to NaN outside orderlet regions.
    data = data_stack_average
    print("ffi,data[13,2077],data[12,2077],data[11,2077] = ",ffi,data[13,2077],data[12,2077],data[11,2077])

    x_hwin = int((x_window - 1) / 2)
    y_hwin = int((y_window - 1) / 2)
    ny = 4080
    nx = 4080
    smooth_image = np.zeros(shape=(ny, nx))
    smooth_image[:] = np.nan                                              # Initialize 2-D array of NaNs
    for i in range(0,ny):
        for j in range(0,nx):

            if np.isnan(data[i, j]): continue

            data_list = []
            for ii in range(i - y_hwin, i + y_hwin + 1):
                if ((ii < 0) or (ii >= ny)): continue
                for jj in range(j - x_hwin, j + x_hwin + 1):
                    if ((jj < 0) or (jj >= ny)): continue

                    datum = data[ii, jj]

                    #print(datum)

                    if not np.isnan(datum):
                        data_list.append(datum)

            if len(data_list) > 0:

                a = np.array(data_list)

                med = np.median(a)
                p16 = np.percentile(a,16)
                p84 = np.percentile(a,84)
                sigma = 0.5 * (p84 - p16)
                mdmsg = med - n_sigma * sigma
                b = np.less(a,mdmsg)
                mdpsg = med + n_sigma * sigma
                c = np.greater(a,mdpsg)
                mask = b | c
                mx = ma.masked_array(a, mask)
                avg = ma.getdata(mx.mean())

                smooth_image[i, j] = avg.item()
                if smooth_image[i, j] <= 0.0:           # Avoid division by zero and no negative values.
                    smooth_image[i, j] = 1.0

    hdu = fits.ImageHDU(smooth_image.astype(np.float32))
    hdu.header['EXTNAME'] = ffi
    hdu.header['XWINDOW'] = (x_window, "X clipped-mean kernel size (pix)")
    hdu.header['YWINDOW'] = (y_window, "Y clipped-mean kernel size (pix)")
    hdu.header['NSIGMA'] = (n_sigma, "Number of sigmas for data-clipping")
    hdu_list.append(hdu)

hdu = fits.HDUList(hdu_list)
hdu.writeto(fname_smooth_lamp,overwrite=True,checksum=True)

