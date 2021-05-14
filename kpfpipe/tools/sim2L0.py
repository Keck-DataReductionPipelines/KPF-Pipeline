# test kpf_analysis script

import numpy as np
import matplotlib.pylab as plt
import os
import glob
from astropy.io import fits

class convert_frame():
	"""
	Class to ingest full frame data (e.g. simulated data) and reformat to quadrant format and save

	"""
	def __init__(self,msmt_indx,path='.',ftype='.FITS', save=True,savepath='.',ser_oscn=156,par_oscn=160):
		"""
		inputs:
			-------
		msmt_indx: integer
			Description: index of file to take from glob of path for all files with extension in ftypes, heritage from KPF CCD analysis code
		path: string, optional
			  Description: path to data folder, points from current working directory
			  default: '.'
		ftypes: string, optional
			Description: the extension of data files to glob, case sensitive, Steve saves files as .FITS
			default: 'FITS'
		save: Boolean, optional
			Description: If true, will save the output of the conversion
			default: True
		savepath: string, optional
			  Description: path to save folder, points from current working directory
			  default: '.'
		ser_oscn: Integer, optional
			Description: Number of serial overscan pixels to add to data
			default: 156 (this is STA value)
		par_oscn: Integer, optional
			Description: Number of serial overscan pixels to add to data
			default: 160 (this is STA value)

		outputs:
		-------
		None
		"""
		cwd = os.getcwd()
		self.files = glob.glob(cwd + os.sep + path + os.sep + '*' + ftype)
		nFiles = len(self.files)
		if nFiles == 0:
			print("No matching files in %s"%(cwd + os.sep + path + os.sep));
			return
		if np.abs(msmt_indx) > nFiles:
			raise TypeError('msmt_indx set to value greater that number of files in path')

		# pick file
		self.file = self.files[msmt_indx]

		self.SIM_to_quad(ser_oscn, par_oscn)
		savename = self.file.split(os.sep)[-1].strip(ftype) + '_quad.fits'
		self.save_quad(savename)
		#self.noise_check()

		return

	def SIM_to_quad(self,ser_oscn, par_oscn):
		"""
		convert FFI to quadrants

		ser_oscn: Integer, optional
			Description: Number of serial overscan pixels to add to data
			default: 156 (this is STA value)
		par_oscn: Integer, optional
			Description: Number of serial overscan pixels to add to data
			default: 160 (this is STA value)
		"""
		self.bias=1000

		f = fits.open(self.file)
		npix, nlines = f[0].header['NAXIS1'], f[0].header['NAXIS2']
		self.data = f[0].data[::-1,:] # flip verticallly to fix orientation. blue is the brighter bit, should be on the right

		prescan = 4 # KPF set detector prescan # pixels
		self.ser_oscn, self.par_oscn, self.prescan = ser_oscn, par_oscn, prescan
		self.npix, self.nlines = npix, nlines 

		# fill into quadrant structure
		ser_extra = ser_oscn + prescan
		par_extra = par_oscn
		L2 = self.bias*np.ones((4, nlines//2 + par_extra, npix//2 + ser_extra),dtype=np.float32)   # npix and nlines defined per quadrant (halved above)
		nlines_L2, npix_L2 = np.shape(L2)[1], np.shape(L2)[2]

		L2[0, 0:nlines//2, prescan  :npix//2 + prescan]	   += self.data[0:nlines//2, 0:npix//2] # top left
		L2[1, 0:nlines//2, ser_oscn :npix//2 + ser_oscn]   += self.data[0:nlines//2, npix//2:]  # top right
		L2[2, par_oscn:,   prescan  :npix//2 + prescan]    += self.data[nlines//2: , 0:npix//2]  # bottom left	
		L2[3, par_oscn:,   ser_oscn :npix//2 + ser_oscn]   += self.data[nlines//2: , npix//2:]   # bottom right

		self.L2 = L2

	def save_quad(self,savename):
		"""
		save quadrant version of simulated data
		"""
		hdr = fits.Header()
		hdr['NOSCN_s'] = self.ser_oscn
		hdr['NOSCN_p'] = self.par_oscn
		hdr['NPSCN']   = self.prescan
		hdu1 = fits.PrimaryHDU(header=hdr)
		hdu2 = fits.ImageHDU(data=self.L2[0],name='quad1') # change to float 32
		hdu3 = fits.ImageHDU(data=self.L2[1],name='quad2')
		hdu4 = fits.ImageHDU(data=self.L2[2],name='quad3')
		hdu5 = fits.ImageHDU(data=self.L2[3],name='quad4')
		hdu6 = fits.ImageHDU(data=np.zeros((self.nlines,self.npix),dtype=np.float32),name='FFI') # 4080x4080 is KPF image area for full frame

		new_hdul = fits.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5, hdu6])
		new_hdul.writeto(savename, overwrite=True)
		

	def bindat(self,x,y,nbins):
		"""
		Bin Data

		Inputs:
		------
		x, y, nbins

		Returns:
		--------
		arrays: bins, mean, std  [nbins]
		"""
		# Create bins (nbins + 1)?
		n, bins = np.histogram(x, bins=nbins)
		sy, _ = np.histogram(x, bins=nbins, weights=y)
		sy2, _ = np.histogram(x, bins=nbins, weights=y*y)

		# Calculate bin centers, mean, and std in each bin
		bins = (bins[1:] + bins[:-1])/2
		mean = sy / n
		std = np.sqrt(sy2/n - mean*mean)

		# Return bin x, bin y, bin std
		return bins, mean, std

	def noise_check(self, sub_lines=list(range(903,906)), sub_pix=list(range(1500,3250)), watts=500):
		"""
		quick check of noise in a random subset

		defaults for file:
		KPF_rays-1.0E+05_orders- 71-103_cal-incandescent_sci-incandescent_sky-incandescent_normalized_159_Red.FITS'

		otherwise must redefine subset bounds
		"""
		#
		# take sum of flux
		dat_rcut  = self.data[sub_lines,:]
		dat_crcut = dat_rcut[:,sub_pix]
		dat_sub   = dat_crcut/watts

		x = np.arange(len(dat_sub[0]))
		vert_sum = np.sum(dat_sub,axis=0)
		plt.figure()
		plt.plot(x,vert_sum,'o')
		out = self.bindat(x,vert_sum,nbins=30)
		plt.errorbar(*out,zorder=100)
		
		# plot std vs mean again but in bins - show get same thing
		plt.figure()
		plt.plot(out[1],out[2],'o')
		plt.xlabel('Mean')
		plt.ylabel('STD')
		plt.plot(out[1],out[2])
		plt.plot(out[1],np.sqrt(out[1]))

		#take out linear trend, plot histogram with std marked
		m,b = np.polyfit(x,vert_sum,deg=1)
		flat = np.mean(vert_sum) * vert_sum/(x*m+b)
		mu = np.mean(flat)
		std = np.std(flat)
		exp = np.sqrt(mu)

		plt.figure()
		plt.hist(flat)
		plt.vlines([mu - std/2, mu + std/2],0,550,color='r',ls='--',label='observed',zorder=150)
		plt.vlines([mu - exp/2, mu + exp/2],0,550,color='k',ls='-',label='expected',zorder=150)
		plt.ylim(0,550)
		plt.xlabel('True Flux (# rays)')
		plt.ylabel('Counts')
		plt.legend()


	def noise_check_old():
		"""
		old noise check by taking std of whole image subset
		"""
		L2_sub = (self.L2[0, 1945:1955, 250:2000] - 1000)/500
		x = np.arange(len(L2_sub[0]))
		vert_sum = np.sum(L2_sub,axis=0)
		vert_mean = np.mean(L2_sub,axis=0)
		vert_std  = np.std(L2_sub,axis=0)

		plt.figure()
		plt.plot(vert_mean, vert_std,'o')

		bin_mean, bin_std, sum_mean = [],[], []
		# step through
		nstep = 100
		for j in x[::nstep]:
			if j < len(L2_sub[0]) - nstep:
				L2_subsub = L2_sub[:, j:j+nstep]
				#L2_subsub = vert_sum[j:j+nstep]
				sum_mean.append(np.sum(L2_subsub))
				bin_mean.append(np.mean(L2_subsub))
				bin_std.append(np.std(L2_subsub))

		bin_mean, bin_std, sum_mean = np.array(bin_mean), np.array(bin_std), np.array(sum_mean)

		plt.plot(bin_mean,bin_std,'o',label='binned mean + std')

		# determine expectation
		expect_std = np.sqrt(np.arange(np.min(vert_mean), np.max(vert_mean)))
		plt.plot(np.arange(np.min(vert_mean), np.max(vert_mean)), expect_std,
			'k-',label='18*sqrt(counts)')
		plt.xlabel('Mean Counts')
		plt.ylabel('Standard Deviation')
		plt.legend()


if __name__=='__main__':
	path = '.'

	# convert frame, i set index=2 to pick the file used for noise calcs, will depend on directory contents
	cf = convert_frame(2,path = path)

	# can do plt.imshow(cf.data,aspect='auto') to figure out desired lines and pixels subset
	# default is some science trace, watts defaulted at 500
	cf.noise_check() 
	plt.show()

	# plot calibration trace noise plots, watts set to 111
	cf.noise_check(sub_lines=list(range(3668,3669)), 
					sub_pix=list(range(3200,4000)), 
					watts=111) 
	plt.show()

	# run through and convert all files in folder
	for i in range(4):
		cf = convert_frame(i,path=path)









