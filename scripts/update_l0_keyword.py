from astropy.io import fits
import pandas as pd

'''
Purpose
-------
Update the L0 fits header 

Input:
------
.csv file with l0 filename, full path, keyword, new_value


Output:
-------
Updated header of L0 file with 

Notes:
------
L0, hdu[0] keywords to update to change the starname
   TARGNAME= '10700 ' / KPF Target name
   FULLTARG= '10700 ' / Full Target name from kpfconfig 
   OBJECT = '10700 ' / Object 
L0, hdu[1] keywords:
   TARGNAME= 10700
   OBJECT = 10700  
   FULLTARG= 10700 

'''

df = pd.read_csv('example_update_l0_keyword.csv')
file = df['filename'][0]
kw_to_change = df['kw2update'][0]
new_value = df['newvalue'][0]

with fits.open(file, mode='update') as hdul:
    print("Updating file: ", file)
    # Access the header of HDU[0]
    header = hdul[0].header
    old_value = header.get(kw_to_change)
    print("Original "+kw_to_change + " in HDU[0]: ",old_value)
    print("New value: ", new_value)
    # Modify the OBJECT keyword
    header[kw_to_change] = new_value
    # Save changes
    hdul.flush()  # Write changes back to the file
    print("OBJECT updated successfully in HDU[0].")
    print("New value: in ",kw_to_change,' : ',header[kw_to_change])

