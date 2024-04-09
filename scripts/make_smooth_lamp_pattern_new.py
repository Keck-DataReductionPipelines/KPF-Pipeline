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
# The implemented method, unless multi-threaded, is slow and takes many hours to complete.
####################################################################################################################

import os
import sys
import numpy as np
from astropy.io import fits
from concurrent.futures import ProcessPoolExecutor, as_completed

# Get input and output files from command-line arguments.

fname_stack_average = (sys.argv)[1]
fname_smooth_lamp = (sys.argv)[2]

print("Input file: fname_stack_average =",fname_stack_average)
print("Output file: fname_smooth_lamp =",fname_smooth_lamp)


def apply_sliding_window_line(data_line, kernel_width, n_sigma):
    """Apply a 1D sliding window operation on a line of data with dynamic kernel adjustment near edges."""
    result_line = np.zeros_like(data_line)
    line_length = len(data_line)
    half_kernel_width = int((kernel_width - 1) / 2)

    for i in range(line_length):
        # Dynamically adjust window start and end to stay within data bounds
        start = max(i - half_kernel_width, 0)
        end = min(i + half_kernel_width + 1, line_length)

        # Extract the window
        window = data_line[start:end]
        # Use only non-NaN values from the window for calculations
        valid_window = window[~np.isnan(window)]

        if valid_window.size > 0:  # Ensure there are valid values to process
            median = np.median(valid_window)
            # Calculate the 16th and 84th percentiles for the spread
            p16, p84 = np.percentile(valid_window, [16, 84])
            spread = 0.5 * (p84 - p16)
            lower_bound = median - n_sigma * spread
            upper_bound = median + n_sigma * spread
            # Clip the window based on bounds and calculate the mean
            clipped_window = valid_window[(valid_window >= lower_bound) & (valid_window <= upper_bound)]
            result_line[i] = np.mean(clipped_window) if clipped_window.size > 0 else np.nan
        else:
            result_line[i] = np.nan  # Set to NaN if no valid values are found

    return result_line

def process_image(data_stack_average, kernel_width, n_sigma, num_cores=None):
    """Process a single CCD image using multiple processes, with progress tracking."""
    ny, _ = data_stack_average.shape
    if num_cores is None:
        num_cores = os.cpu_count()  # Use all available cores if not specified

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all tasks to the executor and store the futures in a list
        futures = [executor.submit(apply_sliding_window_line, data_stack_average[i, :], kernel_width, n_sigma) for i in range(ny)]
        
        # Initialize an array to hold the results, filled with NaNs as placeholders
        smooth_image = np.full((ny, data_stack_average.shape[1]), np.nan)

        # Iterate over completed futures and update progress
        for i, future in enumerate(as_completed(futures)):
            index = futures.index(future)  # Find the original index/order of the completed future
            smooth_image[index, :] = future.result()  # Store the result in the corresponding place
            print(f"Completed: {i+1}/{ny} lines", end='\r')  # Print progress, '\r' returns cursor to start of line

    return smooth_image


if __name__ == '__main__':

    # Load your data
    hdul_stack_average = fits.open(fname_stack_average)

    ffis = ["GREEN_CCD", "RED_CCD"]
    x_window = 200  # Kernel width
    y_window = 1
    n_sigma = 3     # Sigma for clipping
    num_cores = 90

    hdu_list = [fits.PrimaryHDU()]

    for ffi in ffis:
        print(ffi)
        ffi_stack = ffi + "_STACK"
        data_stack_average = hdul_stack_average[ffi_stack].data

        smooth_image = process_image(data_stack_average, x_window, n_sigma, num_cores=num_cores)

        # Ensure no division by zero or negative values
        smooth_image[smooth_image <= 0.0] = 1.0

        # Create new HDU for the smoothed data
        hdu = fits.ImageHDU(smooth_image.astype(np.float32))
        hdu.header['EXTNAME'] = ffi
        hdu.header['XWINDOW'] = (x_window, "X clipped-mean kernel size (pix)")
        hdu.header['YWINDOW'] = (y_window, "Y clipped-mean kernel size (pix)")
        hdu.header['NSIGMA'] = (n_sigma, "Number of sigmas for data-clipping")
        hdu_list.append(hdu)

    # Save the new HDU list to a FITS file
    hdu = fits.HDUList(hdu_list)
    hdu.writeto(fname_smooth_lamp, overwrite=True, checksum=True)
