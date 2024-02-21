import numpy as np
from astropy.io import fits
from concurrent.futures import ProcessPoolExecutor, as_completed

def smooth_row(row_index, data, x_window, n_sigma, nx):
    x_hwin = int((x_window - 1) / 2)
    smooth_row = np.zeros_like(data[row_index])
    smooth_row[:] = np.nan  # Initialize the row with NaNs

    for j in range(nx):
        if np.isnan(data[row_index, j]):
            continue

        data_list = []
        for jj in range(max(j - x_hwin, 0), min(j + x_hwin + 1, nx)):
            datum = data[row_index, jj]
            if not np.isnan(datum):
                data_list.append(datum)

        if data_list:
            a = np.array(data_list)
            med = np.median(a)
            p16 = np.percentile(a, 16)
            p84 = np.percentile(a, 84)
            sigma = 0.5 * (p84 - p16)
            mdmsg = med - n_sigma * sigma
            mdpsg = med + n_sigma * sigma
            clipped_window = a[(a >= mdmsg) & (a <= mdpsg)]

            if clipped_window.size > 0:
                avg = np.mean(clipped_window)
                smooth_row[j] = max(avg, 1.0)  # Ensure non-zero positive values

    return row_index, smooth_row

def multi_threaded_smoothing(data, x_window, n_sigma, max_workers=None):
    ny, nx = data.shape
    smooth_image = np.zeros_like(data)
    smooth_image[:] = np.nan  # Initialize the image with NaNs

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {executor.submit(smooth_row, i, data, x_window, n_sigma, nx): i for i in range(ny)}

        for future in as_completed(future_to_row):
            row_index = future_to_row[future]
            try:
                _, result_row = future.result()
                smooth_image[row_index, :] = result_row
                print(f"Row {row_index+1}/{ny} processed.", end="\r")
            except Exception as exc:
                print(f"Row {row_index+1} generated an exception: {exc}")

    return smooth_image

# Example usage
fname_stack_average = "kpf_20240211_master_flat.fits"
fname_smooth_lamp = "kpf_20240211_smooth_lamp_made20240220_mt.fits"
ffis = ["GREEN_CCD", "RED_CCD"]  # List of FFIs to process

x_window = 200  # Kernel width
n_sigma = 3     # Sigma for clipping

hdul_stack_average = fits.open(fname_stack_average)
hdu_list = [fits.PrimaryHDU()]  # Initialize HDU list with a primary HDU

for ffi in ffis:
    print(ffi)
    data_stack_average = hdul_stack_average[ffi + "_STACK"].data  # Adjust extension name as needed
    smooth_image = multi_threaded_smoothing(data_stack_average, x_window, n_sigma, max_workers=96)

    # Create new HDU for the smoothed data and add to the list
    hdu = fits.ImageHDU(smooth_image.astype(np.float32), name=ffi)
    hdu_list.append(hdu)

# Save the new HDU list to a FITS file
hdul = fits.HDUList(hdu_list)
hdul.writeto(fname_smooth_lamp, overwrite=True)


# hdu_old = fits.open('/data/masters/20240211/kpf_20240211_smooth_lamp_made20240220_small.fits')
# hdu_old = fits.open('/data/masters/20240211/kpf_20240211_smooth_lamp_made20240220_mt.fits')
hdu_old = fits.open('/data/reference_fits/kpf_20240211_smooth_lamp_made20240212.fits')
diff = hdu_old[1].data - hdul[1].data
hdu_old[1].data = diff
print(diff)
hdu_old.writeto('difference.fits', overwrite=True)