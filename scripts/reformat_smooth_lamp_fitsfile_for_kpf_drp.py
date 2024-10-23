import sys
import numpy as np
from astropy.io import fits


#=========================================================================
# Early tests:
#
# Test 1:
#input_file_1 = "/Users/laher/Folks/kpf/kpf_20240321_smooth_lamp.fits"
#input_file_2 = "/Users/laher/Folks/kpf/kpf_20240206_master_flat.fits"
#output_reformatted_file = "kpf_20240321_smooth_lamp_refomatted.fits"
#
#Test 2:
#input_file_1 = "/masters/20240929/kpf_20240929_smooth_lamp.fits"
#input_file_2 = "/masters/20240929/kpf_20240929_master_flat.fits"
#output_reformatted_file = "kpf_20240929_smooth_lamps.fits"
#=========================================================================


# Get command-line arguments.

input_file_1 = (sys.argv)[1]
input_file_2 = (sys.argv)[2]
output_reformatted_file = (sys.argv)[3]

print("input_file_1 =",input_file_1)
print("input_file_2 =",input_file_2)
print("output_reformatted_file =",output_reformatted_file)


# Read in the FITS files and reformat the smooth-lamp-pattern FITS file.

hdul_1 = fits.open(input_file_1)
hdul_2 = fits.open(input_file_2)

ffis = ["GREEN_CCD","RED_CCD"]

hdu_list = []

empty_data = None


# Use primary header of master flat.

primaryHDU = hdul_2[0]
hdu_list.append(primaryHDU)


# Use RECEIPT extension of master flat.

receiptHDU = hdul_2["RECEIPT"]
hdu_list.append(receiptHDU)


# Use CONFIG extension of master flat.

configHDU = hdul_2["CONFIG"]
hdu_list.append(configHDU)

for ffi in ffis:


    # Load the smooth-lamp-pattern data into the output HDU.

    data_1 = hdul_1[ffi].data

    reformatted_data = data_1

    hdu = fits.ImageHDU(reformatted_data.astype(np.float32))


    # Preserve other useful keywords.

    hdu.header['EXTNAME'] = ffi

    ncards = len(hdul_1[ffi].header)


    # This messy code is apparently necessary for preserving the keyword comment.

    for i in range(0, ncards):
            card = hdul_1[ffi].header.cards[i]
            #print("card = ",card)
            if card.keyword == "XWINDOW":
                hdu.header['XWINDOW'] = (card.value,card.comment)
            elif card.keyword == "YWINDOW":
                hdu.header['YWINDOW'] = (card.value,card.comment)
            elif card.keyword == "NSIGMA":
                hdu.header['NSIGMA'] = (card.value,card.comment)

    #hdu.header['XWINDOW'] = hdul_1[ffi].header['XWINDOW']
    #hdu.header['YWINDOW'] = hdul_1[ffi].header['YWINDOW']
    #hdu.header['NSIGMA'] = hdul_1[ffi].header['NSIGMA']

    hdu_list.append(hdu)

hdu = fits.HDUList(hdu_list)
hdu.writeto(output_reformatted_file,overwrite=True,checksum=True)

print("Terminating with exitcode = 0")
exit(0)

