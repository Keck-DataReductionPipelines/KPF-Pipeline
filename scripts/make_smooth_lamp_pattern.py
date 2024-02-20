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
from numpy.lib.stride_tricks import as_strided

# Used to make smooth lamp pattern for 20240211.
fname_stack_average = "kpf_20240211_master_flat.fits"
fname_smooth_lamp = "kpf_20240211_smooth_lamp_made20240212.fits"

def sliding_window_2d(arr, window_shape):
    """Generate a 2D sliding window view of the input array."""
    r, c = window_shape
    new_shape = (arr.shape[0] - r + 1, arr.shape[1] - c + 1, r, c)
    new_strides = arr.strides + arr.strides
    return as_strided(arr, shape=new_shape, strides=new_strides)

# Load your data
hdul_stack_average = fits.open(fname_stack_average)

ffis = ["GREEN_CCD", "RED_CCD"]
x_window = 200  # Kernel width (along dispersion dimension)
y_window = 1    # Kernel height (along cross-dispersion dimension)
n_sigma = 3     # Sigma for clipping

hdu_list = [fits.PrimaryHDU()]

for ffi in ffis:
    print(ffi)
    ffi_stack = ffi + "_STACK"
    data_stack_average = hdul_stack_average[ffi_stack].data

    # Extend data to apply sliding window at the edges
    extended_data = np.pad(data_stack_average, ((x_window//2, x_window//2), (y_window//2, y_window//2)), mode='reflect')
    
    # Generate sliding window views
    windowed_data = sliding_window_2d(extended_data, (x_window, y_window))
    
    # Compute the median and standard deviation for each window
    median = np.median(windowed_data, axis=(2, 3))
    p16 = np.percentile(windowed_data, 16, axis=(2, 3))
    p84 = np.percentile(windowed_data, 84, axis=(2, 3))
    sigma = 0.5 * (p84 - p16)
    
    # Apply clipping
    clipped_data = np.clip(windowed_data, median - n_sigma * sigma[:, :, None, None], median + n_sigma * sigma[:, :, None, None])
    
    # Compute the mean of the clipped data
    smooth_image = np.mean(clipped_data, axis=(2, 3))
    
    # Ensure no division by zero or negative values
    smooth_image[smooth_image <= 0.0] = 1.0

    # Create new HDU for the smoothed data
    hdu = fits.ImageHDU(smooth_image.astype(np.float32))
    hdu.header['EXTNAME'] = ffi
    hdu_list.append(hdu)

# Save the new HDU list to a FITS file
hdu = fits.HDUList(hdu_list)
hdu.writeto(fname_smooth_lamp, overwrite=True, checksum=True)
