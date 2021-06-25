# test kpf_analysis script

import numpy as np
import matplotlib.pylab as plt
import os
import glob
import datetime
from astropy.io import fits

plt.ion()

class ConvertFrame():
	""" Convert Frame Class
	
	This class ingests full frame data (e.g. simulated data from Zemax) and reformats it to quadrant format
	with a constant bias level and saves the final file

	Args:
		msmt_indx (integer): index of file to take from glob of path for all files with extension in ftypes, 
			heritage from KPF CCD analysis code
		path (str): path to data folder, points from current working directory, default: '.'
		ftypes (str): the extension of data files to glob, case sensitive, Steve saves files as .FITS, 
			default: 'FITS'
		save (bool): If true, will save the output of the conversion, default:True
		savepath (str): path to save folder, points from current working directory, default: '.'
		ser_oscn (int): Number of serial overscan pixels to add to data, default: 156 (this is STA value)
		par_oscn (int): Number of serial overscan pixels to add to data, default: 160 (this is STA value)

	Attributes:
		files (list): List of files to consider
		file (str): name of file to convert chosen by msmt_indx

	Raises:
		TypeError: if msmt_indx is greater than nFiles in path

	"""
	def __init__(self,greenfile, redfile, dtype, ser_oscn=156,par_oscn=160, prescan=4, save=True, plot=False):
		green_quads = self.sim_to_quad(greenfile, ser_oscn, par_oscn, prescan,plot=plot)
		red_quads   = self.sim_to_quad(redfile, ser_oscn, par_oscn, prescan,plot=plot)

		if save:
			savename = '%s/KPF_simulated_L0_%s.fits' %(dtype, dtype)
			hdr = self.make_header(ser_oscn, par_oscn, prescan)
			self.save_quad(savename, green_quads, red_quads, hdr)

		return

	def sim_to_quad(self,filename, ser_oscn, par_oscn, prescan,plot=False):
		"""
		convert FFI to quadrants

		Args:
			filename (str): name of file to convert including path
			ser_oscn (int): Number of serial overscan pixels to add to data
			par_oscn (int): Number of serial overscan pixels to add to data
			prescan (int):  Numper of prescan pixels, KPF has 4

		Attributes:
			bias (int): bias level in counts to be added to frame
			data (numpy.ndarray): 2D full frame image, nlines by npix
			npix
			nlines
			quads
		"""
		# define constant bias level to add to frame (could change to add noise here)
		self.bias=1000 

		f = fits.open(filename)
		npix, nlines = f[0].header['NAXIS1'], f[0].header['NAXIS2']
		self.npix, self.nlines = npix, nlines
		data    = f[0].data[::-1,:] # flip verticallly to fix orientation. blue is the brighter bit, should be on the right

		# fill into quadrant structure
		ser_total = npix//2 + ser_oscn + prescan
		par_total = nlines//2 + par_oscn
		quads = self.bias*np.ones((4, par_total, ser_total),dtype=np.float32)  

		quads[0, 0:nlines//2, prescan  :npix//2 + prescan]	  += data[0:nlines//2, 0:npix//2] # top left
		quads[1, 0:nlines//2, ser_oscn :npix//2 + ser_oscn]   += data[0:nlines//2, npix//2:]  # top right
		quads[2, par_oscn:,   prescan  :npix//2 + prescan]    += data[nlines//2: , 0:npix//2]  # bottom left	
		quads[3, par_oscn:,   ser_oscn :npix//2 + ser_oscn]   += data[nlines//2: , npix//2:]   # bottom right

		if plot:
			plt.figure(); plt.imshow(data); plt.title('simulated data frame')
			fig,ax = plt.subplots(nrows=2,ncols=2)
			q=0
			for r in list(range(2)):
				for c in list(range(2)):
					ax[r,c].imshow(quads[q])
					q+=1

		return quads

	def save_quad(self, savename, green_quads, red_quads, hdr):
		"""
		save quadrant version of simulated data to file

		Args:
			savename (str): name of the output file
			green_quads (2D numpy.array): 2D full frame image of green ccd
			red_quads (2D numpy.array): 2D full frame image of red ccd
			hdr (astropy.io.fits.header.Header): header object
		"""
		hdu1 = fits.PrimaryHDU(header=hdr)
		hdu2 = fits.ImageHDU(data=green_quads[0],name='GREEN-AMP1') # change to float 32
		hdu3 = fits.ImageHDU(data=green_quads[1],name='GREEN-AMP2')
		hdu4 = fits.ImageHDU(data=green_quads[2],name='GREEN-AMP3')
		hdu5 = fits.ImageHDU(data=green_quads[3],name='GREEN-AMP4')
		hdu6 = fits.ImageHDU(data=np.zeros((self.nlines,self.npix),dtype=np.float32),name='GREEN-CCD') # 4080x4080 is KPF image area for full frame
		hdu7 = fits.TableHDU(data=None, name='GREEN-RECEIPT')

		hdu8 = fits.ImageHDU(data=red_quads[0],name='RED-AMP1') # change to float 32
		hdu9 = fits.ImageHDU(data=red_quads[1],name='RED-AMP2')
		hdu10 = fits.ImageHDU(data=red_quads[2],name='RED-AMP3')
		hdu11 = fits.ImageHDU(data=red_quads[3],name='RED-AMP4')
		hdu12 = fits.ImageHDU(data=np.zeros((self.nlines,self.npix),dtype=np.float32),name='RED-CCD') # 4080x4080 is KPF image area for full frame
		hdu13 = fits.TableHDU(data=None, name='RED-RECEIPT')

		hdu14 = fits.ImageHDU(data=np.zeros((100,100)), name='CA-HK')
		hdu15 = fits.ImageHDU(data=np.zeros((100,100)), name='EXPMETER')
		hdu16 = fits.ImageHDU(data=np.zeros((100,100)), name='GUIDECAM')
		hdu17 = fits.ImageHDU(data=np.zeros((100,100)), name='SOLAR-IRRADIANCE')

		new_hdul = fits.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5, hdu6])
		new_hdul.writeto(savename, overwrite=True)
		

	def make_header(self, ser_oscn, par_oscn, prescan):
		"""
		Args:
			ser_oscn (int):  Number of serial overscan pixels to add to data
			par_oscn (int):  Number of serial overscan pixels to add to data
			prescan (int):  Numper of prescan pixels, KPF has 4
		"""
		typemap = {'bool':bool, 'float':float, 'int':int,'string':str,'double':np.double,'DateTime':str,'':str} #datetime.datetime? for datetime? not worth it r.n.
		hdr = fits.Header()
		hdr['SOSCN'] = (str(ser_oscn), 'serial overscan')
		hdr['POSCN'] = (str(par_oscn), 'parallel overscan')
		hdr['PSCN']  = (str(prescan), 'parallel overscan')

		# load header
		f = np.loadtxt('./KPF FITS Header Keywords - kpf_header.tsv', delimiter='	',dtype=str)
		keys   = f[:,1]
		values = f[:,3]
		types  = f[:,5]

		for i, key in enumerate(keys):
			if (len(key) > 0) & (i > 3):
				try:
					val, com = values[i].split('/')[0].replace("'","").strip(), values[i].split('/')[1].strip()
					hdr[key] = (typemap[types[i]](val), com)
				except IndexError:
					try:
						hdr[key] =typemap[types[i]](values[i].split('/')[0].replace("'","").strip())
					except ValueError:
						val= values[i].split('/')[0].replace("'","").strip()
						hdr[key] = 0
				except ValueError:
						com= values[i].split('/')[0].replace("'","").strip()
						hdr[key] = (0,com)


		return hdr


if __name__=='__main__':
	# convert frame, i set index=2 to pick the file used for noise calcs, will depend on directory contents
	greenfile='./flat/KPF_rays-1.0E+05_orders-103-138_cal-incandescent_sci-incandescent_sky-incandescent_normalized_154_Green.FITS'
	redfile = './flat/KPF_rays-1.0E+05_orders- 71-103_cal-incandescent_sci-incandescent_sky-incandescent_normalized_159_Red.FITS'
	cf = ConvertFrame(greenfile,redfile,'flat',plot=False)

	# convert frame, i set index=2 to pick the file used for noise calcs, will depend on directory contents
	greenfile='./science/KPF_rays-1.0E+05_orders-103-138_cal-lfc_20GHz_sci-solar_Planck_sky-solar_Planck_RV_30000ms_normalized_193_Green.FITS'
	redfile = './science/KPF_rays-1.0E+05_orders- 71-103_cal-lfc_20GHz_sci-solar_Planck_sky-solar_Planck_RV_30000ms_normalized_10018_Red.FITS'
	cf = ConvertFrame(greenfile,redfile,'science',plot=True)










