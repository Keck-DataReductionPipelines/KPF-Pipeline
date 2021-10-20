from astropy.io import fits

myfits = fits.open('/code/KPF-Pipeline/outputs/neidL1_20210719T232216_L1_wave.fits')
print(myfits['CALWAVE'].data[95])
print(myfits['CALWAVE'].data[96])
print(myfits['CALWAVE'].data[97])