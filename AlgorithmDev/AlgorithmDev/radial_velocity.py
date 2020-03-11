"""This module finds radial velocities by cross correlating star spectra with stellar template mask"""

from __future__ import print_function
from astropy.io import fits
from astropy.coordinates import Angle
from astropy.modeling import models, fitting
import warnings
import numpy as np
import time
import datetime
from barycorrpy import get_BC_vel
from astropy.utils import iers
import os.path
import csv

#STEP_INDEX = [-82, 82]
STEP_INDEX = [-80.0, 81.0]
LIGHT_SPEED = 299792.458   # light speed in km/s
LIGHT_SPEED_M = 299792458. # light speed in m/s
X1 = 500
X2 = 3500
SEC_TO_JD = 1.0/86400.0
FIT_G = fitting.LevMarLSQFitter()
MORE_FOR_ANALYSIS = 3

# key constant in config
STAR_RV = 'star_rv'
OBSLON = 'obslon'
OBSLAT = 'obslat'
OBSALT = 'obsalt'
RA = 'ra2000'
DEC = 'de2000'
PM_RA = 'pm_ra'
PM_DEC = 'pm_dec'
PARALLAX = 'parallax'
STEP = 'step'
MASK_WID = 'mask_width'

class RadialVelocity:
    """Class for finding radial velocity of the star.

    Parameters:
        config (dict):
            containing setting for velocity calculation, including setting like:
            starname, star_rv, obslon, obslat, obsalt, ra2000, dev2000, pm_ra, pm_dec, parallax,
            step, mask_width
        spectrum_path (str):the path of spectrum fits
        mask_path (str): mask file path
    """

    def __init__(self, config, spectrum_order, mask_path):
        self.config = config
        self.mask_path = mask_path
        self.velocity_loop = None     # loop of velocities for rv finding
        self.velocity_steps = None    # total number in velocity_loop
        self.mask_line = None
        self.zb_long = None
        self.spectrum_order = spectrum_order
        self.init_calc = False
        if 'overwrite' not in self.config:
            self.config['overwrite'] = True


    def init_calculation(self):
        """ initial calulation based on configuration setting """

        must_config_keys = (STAR_RV, OBSLON, OBSLAT, OBSALT, RA, DEC, PM_RA, PM_DEC, PARALLAX, STEP, MASK_WID)

        if self.config and set(must_config_keys).issubset(self.config):
            iers.Conf.iers_auto_url.set('ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all')
            self.get_velocity_loop()
            self.get_velocity_steps()
            self.get_zb_long()
            self.get_mask_line()
            return True
        else:
            return False

    def get_velocity_loop(self):
        if self.velocity_loop is None:
            self.velocity_loop = np.arange(STEP_INDEX[0], STEP_INDEX[1]) * self.config[STEP] + self.config[STAR_RV]
        return self.velocity_loop

    def get_velocity_steps(self):
        if self.velocity_steps is None:
            vel_loop = self.get_velocity_loop()
            self.velocity_steps = len(vel_loop)
        return self.velocity_steps

    def get_BC_corr_RV(self, start_jd, days_period=None):
        bc_corr = list()

        bc_file = None
        if days_period is None or days_period == 0:
            jds = [start_jd]
        else:
            bc_file = '../test_data/rv_test/bc_corr'+str(start_jd)+'_'+str(days_period)+'.csv';
            if os.path.isfile(bc_file):
                with open(bc_file) as bc_csv:
                    bc_row = csv.reader(bc_csv)
                    for row in bc_row:
                        bc_corr.append(float(row[0]))

        if len(bc_corr) == 0:
            jds = np.arange(days_period, dtype=float) + start_jd if days_period is not None else [start_jd]

            bc_corr = [get_BC_vel(JDUTC=jd,
                                 ra = self.config[RA],
                                 dec = self.config[DEC],
                                 pmra = self.config[PM_RA],
                                 pmdec = self.config[PM_DEC],
                                 px = self.config[PARALLAX],
                                 lat = self.config[OBSLAT],
                                 longi = self.config[OBSLON],
                                 alt = self.config[OBSALT],
                                 rv = self.config[STAR_RV])[0][0]/LIGHT_SPEED_M for jd in jds]

            if bc_file is not None:
                with open(bc_file, mode='w') as bc_csv:
                    result_writer = csv.writer(bc_csv)
                    for bc in bc_corr:
                        result_writer.writerow([bc])
        return np.array(bc_corr)

    def get_zb_long(self):
        if self.zb_long is None:
            rv_list = self.get_BC_corr_RV(2458591.5, 380) #self.get_BC_corr_RV(2458591.5, 380)
            self.zb_long = np.array([min(rv_list), max(rv_list)])
        return self.zb_long


    def get_mask_line(self):
        """calculate mask line center, start and end"""
        if self.mask_line is None:
            mask_width = self.config[MASK_WID]
            line_center, line_weight = np.loadtxt(self.mask_path, dtype=float, unpack=True) # load mask file
            line_mask_width = line_center * (mask_width/LIGHT_SPEED)

            self.mask_line = {'start': line_center - line_mask_width,
                              'end':    line_center + line_mask_width,
                              'center': line_center,
                              'weight': line_weight}

            v_loop = self.get_velocity_loop()
            zb_long = self.get_zb_long()
            dummy_start = self.mask_line['start'] * ((1.0 + (v_loop[0]/LIGHT_SPEED))/(zb_long[1] + 1.0))
            dummy_end = self.mask_line['end'] * ((1.0 + (v_loop[-1]/LIGHT_SPEED))/ (zb_long[0] + 1.0))
            self.mask_line.update({'bc_corr_start': dummy_start,
                                   'bc_corr_end': dummy_end})

        return self.mask_line


    def get_wavecalib_poly_params(self, hdr, start_order):
        CAL_TH_COEFF = 'HIERARCH ESO DRS CAL TH COEFF LL'
        p_degree = hdr['HIERARCH ESO DRS CAL TH DEG LL']

        wcalib_coeffs_orders = []

        for ord in range(start_order, self.spectrum_order+start_order):
            coeff_base = int(p_degree+1)*ord
            ll_coeffs_order = [hdr[CAL_TH_COEFF+str(coeff_base+i)] for i in range(p_degree, -1, -1)]
            wcalib_coeffs_orders.append( ll_coeffs_order )

        return wcalib_coeffs_orders


    def get_rv_on_spectrum_fits(self, spectrum_fits, weigh_ccf, hdu_in_calib=None, BC_corr_key=None, \
                                start_x=None, end_x=None, start_order=None, end_order=None):
        spectrum, hdr = fits.getdata(spectrum_fits, header=True)
        return self.get_rv_on_spectrum(spectrum, hdr, weigh_ccf, hdu_in_calib, BC_corr_key, start_x, end_x, start_order, end_order)


    def get_wavecal_by_poly_from_hdr(self, hdr, spectrum_x, start_order):
        wavecalib_coeffs = self.get_wavecalib_poly_params(hdr, start_order)
        wavecals = np.zeros((self.spectrum_order, np.size(spectrum_x)))

        for ord in range(self.spectrum_order):
            wavecals[ord, :] = np.polyval(np.poly1d(wavecalib_coeffs[ord]), spectrum_x) # calibrate pixel to wavelength

        return wavecals

    def get_wavecal_by_map(self, calib_file, hdu_index, spectrum_x, start_order):
        calib_hdu = fits.open(calib_file)
        if (hdu_index is None):
            hdu_index = 0

        calib_map = calib_hdu[hdu_index].data
        end_order = start_order + self.spectrum_order
        new_calibs = calib_map[start_order:end_order, spectrum_x]

        return new_calibs

    def get_rv_on_spectrum(self, spectrum, hdr, weigh_ccf=None, wavelength_calib_file=None, hdu_in_calib=None,
                           start_x = None, end_x = None, start_order=None, end_order=None, BC_corr=None,
                           order_diff = 0):
        """
        compute radial velocity of all orders based on 2D spectrum

        Parameters
            spectrum (array): spectrum of 2D array
            hdr (dict): header of spectrum fits
            weigh_ccf (array): cross correlation results of some reference for final scaling

        Returns:
            out (array): 2D array containing the correlation result at each velocity step of all orders
                         size of 2D array: 73 x pixels along x axis.
                         The first 70 (0-69) rows stores cross correlation of all orders
                         row 70: blank
                         row 71 reserved for velocity steps
                         row 72 reserved for rv summation from order 1-69 done by analyze_ccf()
        """

        if BC_corr is None:
            obsjd = hdr['MJD-OBS'] + 2400000.5 + hdr['EXPTIME'] * SEC_TO_JD/2
            zb = self.get_BC_corr_RV(obsjd)[0]
        else:
            zb = BC_corr

        s_x = X1 if start_x is None else start_x
        e_x = X2 if end_x is None else end_x
        spectrum_x = np.arange(np.shape(spectrum)[1])[s_x:e_x]

        s_order = 0 if start_order is None else start_order
        e_order = self.spectrum_order+s_order-1 if end_order is None else end_order

        new_spectrum = spectrum[s_order:e_order+1, s_x:e_x]
        new_w_ccf = None if weigh_ccf is None else weigh_ccf[s_order:e_order+1, s_x:e_x]

        result_ccf = np.zeros([self.spectrum_order+MORE_FOR_ANALYSIS, self.velocity_steps])

        if wavelength_calib_file is None:
            wavecal_all_orders = self.get_wavecal_by_poly_from_hdr(hdr, spectrum_x, s_order+order_diff)
        else:
            wavecal_all_orders = self.get_wavecal_by_map(wavelength_calib_file, hdu_in_calib, spectrum_x, s_order+order_diff)

        for ord in range(self.spectrum_order):
            print(ord, ' ', end="")
            wavecal = wavecal_all_orders[ord, :]

            if np.any(wavecal != 0.0):
                w_ccf = weigh_ccf[ord, :] if new_w_ccf is not None else None
                result_ccf[ord, :] = self.cross_correlate_by_mask_shift(wavecal, new_spectrum[ord, :], zb, w_ccf)
            else:
                print("all wavelength zero")

        print("\n")
        result_ccf[~np.isfinite(result_ccf)] = 0.

        return result_ccf

    def fit_ccf(self, velocities, ccf, rv_guess, velocity_cut=100.0):
        g_init = models.Gaussian1D(amplitude=-1e7, mean=rv_guess, stddev=5.0)
        #i_cut = (velocities >= rv_guess - velocity_cut) & (velocities <= rv_guess + velocity_cut)
        i_cut = (rv_guess - velocity_cut) < velocities < (rv_guess + velocity_cut)
        g_x = velocities[i_cut]
        g_y = ccf[i_cut] - np.nanmedian(ccf)
        gaussian_fit = FIT_G(g_init, g_x, g_y)
        rv_mean = gaussian_fit.mean.value

        return gaussian_fit, g_x, g_y, rv_mean

    def cross_correlate_by_mask_shift(self, wavecal, spectrum, zb, weigh_ccf_ord):
        line = self.mask_line
        line_index = np.where ((line.get('bc_corr_start') > np.min(wavecal)) & (line.get('bc_corr_end') < np.max(wavecal)))[0]
        nline_index = len(line_index)

        v_steps = self.velocity_steps
        ccf = np.zeros(v_steps)
        if nline_index == 0:
            return ccf

        n_xpixel = np.shape(wavecal)[0]
        pix1 = 10
        pix2 = n_xpixel - 11

        new_line_start = line['start'][line_index]
        new_line_end = line['end'][line_index]
        new_line_center = line['center'][line_index]
        new_line_weight = line['weight'][line_index]

        xpixel_wavestart = (wavecal + np.roll(wavecal,1))/2.0   # w[0]-(w[1]-w[0])/2, (w[0]+w[1]).....
        xpixel_waveend = np.roll(xpixel_wavestart, -1)          # (w[0]+w[1])/2,      (w[1]+w[2])/2....
        #xpixel_waveend = (wavecal + np.roll(wavecal,-1))/2.0

        #xpixel_wavestart[0] = wavecal[0]
        #xpixel_waveend[-1] = wavecal[-1]

        # fix
        xpixel_wavestart[0] = wavecal[0] - (wavecal[1]-wavecal[0])/2.0
        xpixel_waveend[-1] = wavecal[-1] + (wavecal[-1]-wavecal[-2])/2.0

        shift_lines_by = (1.0 + (self.velocity_loop / LIGHT_SPEED)) / (1.0 + zb) #Shifting mask in redshift space

        total_match = 0.
        total_cc = 0.
        for c in range(v_steps):
            line_dopplershifted_start =  new_line_start * shift_lines_by[c]
            line_dopplershifted_end =  new_line_end * shift_lines_by[c]
            line_dopplershifted_center =  new_line_center * shift_lines_by[c]

            #closestmatch = np.sum((xpixel_wavestart - line_dopplershifted_center[:,np.newaxis] <= 0.), axis=1)
            closestmatch = np.sum((xpixel_waveend - line_dopplershifted_start[:,np.newaxis] < 0.), axis=1)
            closestmatch_next = np.sum((xpixel_wavestart - line_dopplershifted_end[:, np.newaxis] < 0.), axis=1)
            maskspectra_dopplershifted = np.zeros(n_xpixel)

            for k in range(nline_index):
                #closest_xpixel = closestmatch[k] - 1    # fix: closest index before line_dopplershifted_center
                closest_xpixel = closestmatch[k]       # before fix
                closest_xpixel_next = closestmatch_next[k]
                line_startwave = line_dopplershifted_start[k]
                line_endwave = line_dopplershifted_end[k]
                line_weight = new_line_weight[k]

                if (closest_xpixel_next <= pix1 or closest_xpixel >= pix2):
                    continue
                else:
                    for n in range(closest_xpixel, closest_xpixel_next):
                        if n >= pix2:
                            break
                        if n <= pix1:
                            continue
                        # if there is overlap
                        if xpixel_wavestart[n] <= line_endwave and xpixel_waveend[n] >= line_startwave:
                            wavestart = max(xpixel_wavestart[n], line_startwave)
                            waveend = min(xpixel_waveend[n], line_endwave)
                            maskspectra_dopplershifted[n] = line_weight * (waveend - wavestart)/(xpixel_waveend[n]-xpixel_wavestart[n])
            ccf[c] = np.nansum(spectrum * maskspectra_dopplershifted)

        #print('ccf ', ccf)  #total_match averagely higher than original code??
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if weigh_ccf_ord is None:
                weigh_ccf_ord = ccf.copy()
            ccf *=  np.nanmean(weigh_ccf_ord / ccf)
        return ccf

    def weigh_with_other_ccf(self, other_ccf, ccf):
        order = np.shape(ccf)[0]

        for ord in range(order):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                ccf[ord, :] *=  np.nanmean(other_ccf[ord,:] / ccf[ord,:])

        ccf[~np.isfinite(ccf)] = 0.
        return ccf

    def analyze_ccf(self, ccf):
        ccf[self.spectrum_order+1, :] = self.velocity_loop
        ccf[self.spectrum_order + MORE_FOR_ANALYSIS-1, :] = np.sum(ccf[1:self.spectrum_order, :], axis=0) # no use order 0
        return ccf

    def output_ccf_to_fits(self, ccf, out_fits, ref_head, mean ):
        hdu = fits.PrimaryHDU(ccf)
        for key in ref_head:
            if key in hdu.header or key == 'COMMENT':
                continue
            else:
                if key == 'Date':
                    hdu.header[key] = str(datetime.datetime.now())
                elif 'ESO' in key:
                    if  'ESO DRS CCF RVC' in key:
                        hdu.header['CCF-RVC'] = (str(mean), ' Baryc RV (km/s)')
                else:
                    hdu.header[key] = ref_head[key]

        hdu.writeto(out_fits, overwrite=True)

    def fit_ccf(self, result_ccf, velocity_cut=100.0):
        rv_guess = self.config[STAR_RV]
        g_init = models.Gaussian1D(amplitude=-1e7, mean=rv_guess, stddev=5.0)
        velocities = result_ccf[self.spectrum_order+1, :]
        ccf = result_ccf[self.spectrum_order+MORE_FOR_ANALYSIS-1, :]
        i_cut = (velocities >= rv_guess - velocity_cut) & (velocities <= rv_guess + velocity_cut)
        g_x = velocities[i_cut]
        g_y = ccf[i_cut]-np.nanmedian(ccf)
        gaussian_fit = FIT_G(g_init, g_x, g_y)
        rv_result = gaussian_fit.mean.value
        return gaussian_fit, rv_result, g_x, g_y