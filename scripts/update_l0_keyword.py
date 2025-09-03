from astropy.io import fits
import pandas as pd
import os

'''
Purpose
-------
Update the L0 fits header 

Input:
------
.csv file with l0 filename(fullpath), keyword, new_value
Example first two rows
    filename,kw2update,newvalue
    /data/kpf/L0/20250101/KP.20250101.000000.fits,OBJECT,10700

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

# Read CSV file
df = pd.read_csv('header_keyword_change_2025feb18.csv')

# Loop through each row in the DataFrame
for index, row in df.iterrows():
    file = row['filename']
    kw_to_change = row['kw2update']
    new_value = row['newvalue']
    
    if not os.path.exists(file):  # Check if the file exists
        print(f"Skipping {file}: File not found.")
        continue

    try:
        with fits.open(file, mode='update') as hdul:
            print(f"Updating file: {file}")

            # Access the header of HDU[0]
            header = hdul[0].header

            old_value = header.get(kw_to_change, "NOT FOUND")
            print(f"Original {kw_to_change} in HDU[0]: {old_value}")
            print(f"New value: {new_value}")

            # Update the header keyword
            header[kw_to_change] = new_value

            # Save changes
            hdul.flush()  
            print(f"{kw_to_change} updated successfully in HDU[0].")
            print(f"New value in {kw_to_change}: {header[kw_to_change]}")
    
    except Exception as e:
        print(f"Error processing {file}: {e}")

