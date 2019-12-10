
# -- Template fittig algorithm arguments --
# This file stores all data structures used by the template fitting 
# algorithm. This include input arguments, output arguments, and 
# debugging results. 

import macro as mc

import copy
import typing as tp
from astropy.io import fits
from astropy.time import Time
import numpy as np
import scipy.ndimage as img

import matplotlib.pyplot as plt

class TFASpec:
    '''
    Argument object class that contains data from the echelle spectrum 
    Can be initialize to shape [72, 4096] of zeros by Spec()
    Can read from a .fits file by Spec(filename: str)
    can be inialized by 
    '''

    def __init__(self, data: mc.EchellePair_TYPE=None, 
                       filename: str=None,
                       jd: Time=None) -> None:
        '''
        Initializer of the Spec Class 
        '''

        # The shape of wave and data is initialized with dimension 
        # defined by the global
        self._wave = np.zeros(mc.ECHELLE_SHAPE)
        self._spec = np.zeros(mc.ECHELLE_SHAPE)
        self.NOrder = mc.ECHELLE_SHAPE[0]
        self.NPixel = mc.ECHELLE_SHAPE[1]

        # If the data is generated from a .fits file, 
        # self.filename contains the .fits file destination 
        # self.header contains all headers of that  file 
        self.filename = None
        self.header = None

        # Now initialize the class based on the given input
        if filename != None and data != None: 
            # error! Cannot read from both a file and a defined spectrum
            # --TODO-- 
            # Make this more informative
            print('what')
        elif filename != None:
            # a file is specified, so read data from that file 
            # prioritize data from file, so overlook any xy data 
            self.read_from(filename)
            try: 
                # Each file must have a julian date 
                self.julian_day = Time(self.header['eso drs bjd'], format='jd')
            except: 
                print('what')
                exit(-1)
            self.flag['from_file'] = True
        elif data != None:
            # no filename is specified and a set of data is provided

            if data[0].shape != data[1].shape:
                # size of wave data must be same as flux data
                msg = 'size of data[0] (wave) and data[1] must be same, \
                        but have size {}, {}'.format(
                            data[0].size, data[1].size) 
                raise ValueError(msg)
            else:
                # Success!
                self._wave = data[0]
                self._spec = data[1]
                self.NOrder, self.NPixel = self._wave.shape
                self.NPixel = mc.ECHELLE_SHAPE[1]
                self.julian_day = jd
                self.flag['from_array'] = True
        else: 
            # in this case nothing is given 
            # we leave the data field blank 
            pass

    def __eq__(self, other: type) -> bool:
        '''
        Comparison == between two Spec Class
        return true if and only if both wavelength and flux 
        are exactly the same
        '''
        return np.logical_and(
            np.all(self._spec == other._spec),
            np.all(self._wave == other._wave)
        )

    # public methods
    def copy(self) -> type:
        '''
        returns a deep copy of self, with a new class ID
        '''
        dup = copy.deepcopy(self)
        return dup
    
    def get_order(self, order: int) -> mc.EchellePair_TYPE:
        '''
        returns a tuple of 2 np.ndarray representing 
        wave and flux of the specified order
        '''
        return (self._wave[order], self._spec[order])

    def read_from(self, fname: str, HDU: str='primary') -> None:
        ''' '''
        if fname.endswith('.fits') == False:
            # Can only read from .fits files
            msg = 'input files must be .fits files'
            raise IOError(msg)
        
        self.filename = fname
        with fits.open(fname) as hdu_list:
            # First record relevant header information 
            self.header = hdu_list[HDU].header
            self._spec = hdu_list[HDU].data
            self.NOrder, self.NPixel = self._spec.shape
            # Generate wavelength values for each order
            for order in range(self.NOrder):
                self._wave[order] = self._gen_wave(order)

    def write_to(self, fname: str, deg: int) -> None:
        '''
        Take the current data and write to a .fits file
        '''
        if self.flag['from_file'] != True and self.flag['from_array'] != True:
            msg = 'Can only write to file when not empty!'
            raise ValueError(msg)
        if fname.endswith('.fits') == False:
            msg = 'Can only write to .fits files!'
            raise IOError(msg)

        # Initialize a new instance of HDU and save data to primary
        hdu = fits.PrimaryHDU(self._spec)
        hdu_header = hdu.header

        # Record relevat headers
        hdu_header.set('axis2', self.NOrder)
        hdu_header.set('axis1', self.NPixel)

        # degree of interpolation for wavelength
        deg_key = 'hierarch eso drs cal th deg ll'
        hdu_header.set(deg_key, deg)

        hdu_header.set('hierarch eso drs berv', 0)
        hdu_header.set('hierarch eso drs bjd', self.julian_day.jd)

        # Record polynomial interpolation results to headers
        for order in range(self.NOrder):
            c = np.polyfit(np.arange(self.NPixel), self._wave[order], deg)
            c = np.flip(c)
            for i, ci in enumerate(c):
                key = 'hierarch eso drs cal th coeff ll' + str((deg+1)*order+i)
                hdu_header.set(key, ci)

        hdul = fits.HDUList([hdu])
        hdul.writeto(fname, overwrite=True)

    def shift(self, a: mc.ALPHA_TYPE, order: int) -> None:
        ''' '''
        # Create flux normalization polynomials
        c = int(self._wave[order].size/2)
        am = np.flip(a[1:])
        px = self._wave[order] - self._wave[order, c]
        norm = np.polyval(am, px)

        # alpha_v shift for spectrum 
        av = np.divide(1, a[0])

        # shift wavelength and flux accordingly 
        self._wave[order] *= av
        self._spec[order] *= norm

    def resample(self, resolution: float) -> None:
        ''' '''
        # img.zoom use Cubic spline interpolation on default
        for order in self.NOrder:
            self._wave[order] = img.zoom(self._wave[order], resolution)
            self._spec[order] = img.zoom(self._spec[order], resolution)
        self.NPixel *= resolution

    def plot(self, order: int, 
                   comment: str='', 
                   color: str='g') ->plt.Figure:
        ''' '''
        # fig = plt.figure()
        plt.plot(self._wave[order], self._spec[order], 
                 label=comment,
                 color=color,
                 linewidth=0.5)
        plt.legend()
        # return fig

    def shift_wave(self, a: float, order: int) -> None: 
        ''' 
        shift the wavelength of this spectrum only
        used in barycenter correction 
        '''
        self._wave[order] *= a

    # private helper functions:
    def _gen_wave(self, order: int) -> mc.EchelleData_TYPE:
        ''' generate wavelength for flux of specified order '''
        opower = self.header['eso drs cal th deg ll']
        a = np.zeros(opower+1)
        for i in range(0, opower+1, 1):
            keyi = 'eso drs cal th coeff ll' + str((opower+1)*order+i)
            a[i] = self.header[keyi]
        wave = np.polyval(
            np.flip(a),
            np.arange(self.NPixel, dtype=np.float64)
        )
        return wave

def flatten(lst: np.ndarray) -> np.ndarray:
    '''
    flatten a nested numpy array
    '''
    return sum( ([x] if not isinstance(x, np.ndarray) else flatten(x)
            for x in lst), [])

class TFAResult:
    # An Argument class used to store results from the 
    # template fitting algorithms 

    def __init__(self, header):
        '''
        constructor
        '''
        self.header = header
        self.res_df = pd.DataFrame(columns=self.header)

    def append(self, order, inter_res):
        '''
        add to the bottom of the result
        '''
        flat_res = flatten(inter_res)
        assert(len(flat_res) == len(self.res_df.columns))
        self.res_df.at[order] = flat_res

class TFADebugRes(TFAResult): 
    # an argument class that stores debug information 
    # and value for each order.
    # all significant parameters from each step of the
    # newton solver is recorded.
    # the result member is formatted as:
    # [a, da, err, converge, k, X2]

    def __init__(self, obs_name: str, temp_name: str, order: int, m: int):
        '''
        constructor
        '''
        self.obs_name = obs_name
        self.temp_name = temp_name
        self.order = order
        self.exit_msg = None

        # order specific values to be recorded
        # result is stored as pandas dataframe
        # add a header for each value of alpha 

        # top level header [alpha, dalpha, error...]
        top = ['alpha' for n in range(m+2)] + ['d_alpha' for n in range(m+2)]
        top += ['error', 'convergence', 'kappa', 'Chi^2']
        # sub level header [alpha[-1], alpha[0], ...]
        # alpha[-1] is alpha_v (for easy numbering)
        sub = ['alpha[{}]'.format(n-1) for n in range(m+2)] \
                + ['d_alpha[{}]'.format(n-1) for n in range(m+2)]
        sub += ['', '', '', ''] 
        #overall header
        header = [np.array(top), np.array(sub)]
        TFAResult.__init__(self, header)

        self.weight = None
        self.ran = False
    
    def log_exit(self, exit_msg: str) -> None:
        '''
        order process exited with a failed messge
        '''
        self.exit_msg = exit_msg
    
    def append_weight(self, weight: np.ndarray) -> None:
        '''
        add weight 
        '''
        if type(self.weight) == type(None):
            self.weight = weight
        else:
            self.weight = np.vstack([self.weight, weight])

    def record(self, xlsx_writer, w_path):
        '''
        record to an .xlsx file
        each sheet in this file represnet the result of a order
        the structure of the file is 
            exit message 
            table of data
        '''
        # name of the excel sheet we are writing to 
        sheet_name = str(self.order)
        self.final_res = pd.DataFrame(columns=['alpha[v]', 'Error', 'success', 'iteration', 'Message'])
        self.files = pd.DataFrame(columns=['name'])

        a = self.res_df['alpha']['alpha[-1]'].values[-1]
        e = self.res_df['error'].values[-1]
        iteration = len(self.res_df.index)

        # first row is header
        if self.exit_msg == None:
            # exit message never updated, so this order processed successfully
            msg = 'order computed successfully'
            success = True
        else:
            msg = self.exit_msg
            success = False
          
        # write all data to the sheet 
        self.final_res.at[self.order] = [a, e, success, iteration, msg]
        self.final_res.to_excel(xlsx_writer, sheet_name=sheet_name, startrow=0)
        self.res_df.to_excel(xlsx_writer, sheet_name=sheet_name, startrow=3)

        # write the weight to a .dat file
        np.savetxt(w_path, self.weight, delimiter=',')

class TFAOrderResult(TFAResult):
    # An Argument class used to store results from the 
    # template fitting algorithms 

    def __init__(self, m, jd):
        '''
        constructor
        initialize relevant result members
        '''
        
        # File specific information
        self.m = m
        self.julian_day = jd

        # order specific results from TFAs
        top = ['RV[km/s]']+ (m+2)*['alpha'] + ['Error', 'success', 'iteration']
        sub = []
        for i in range(m+2):
            sub += ['alpha[{}]'.format(i-1)]
        bottom = [''] + sub + ['', '', '']
        header = [np.array(top), np.array(bottom)]
        TFAResult.__init__(self, header)

        self.rmo = rm.RemoveOutlier()

    def append(self, order, inter_res):
        '''
        add to the bottom of the result
        '''
        # [a[0], e, s, i]
        flat_res = flatten(inter_res)
        flat_res = [(1-flat_res[0])*mc.C_SPEED] + flat_res
        assert(len(flat_res) == len(self.res_df.columns))
        self.res_df.at[order] = flat_res
    
    def average(self):
        '''
        flat average 
        '''
        #[RV, Error, s_rate, iteration]

        rv = np.mean(self.res_df['RV[km/s]'].to_numpy())
        error = np.mean(self.res_df['Error'].to_numpy())
        return [rv, error]
    
    def weighted_average(self):
        '''
        weighted average
        '''
        rv = self.res_df['RV[km/s]'].to_numpy()
        weight = np.divide(1, np.square(self.res_df['Error'].to_numpy()))
        
        mu_e = np.sum(np.multiply(rv, weight)) / np.sum(weight)
        error = np.mean(self.res_df['Error'].to_numpy())
        ret_val = [mu_e, error]

        return ret_val
    def write_to_final(self):
        '''

        '''
        date = self.julian_day.isot
        jd = self.julian_day.jd
        result = np.asarray(self.average())
        converge = np.mean(self.res_df['iteration'])
        success_rate = np.mean(self.res_df['success'])
        return np.asarray([date, jd, result, converge, success_rate])

class TFAFinalResult(TFAResult):
    
    def __init__(self):
        '''
        constructor
        '''

        header = ['Date', 'Julian Date', 'RV[km/s]', 'Error', 'converge', 'success_rate']
        TFAResult.__init__(self, header)

    def to_csv(self, f_path: str):
        '''
        write to a .csv file
        '''
        self.res_df.to_csv(f_path)
    
    def convert_to_ms(self):
        self.res_df['RV[km/s]'] *= 1000.0
        self.res_df['Error'] *= 1000.0
        self.res_df.rename(columns={'RV[km/s]': 'RV[m/s]'})

if __name__ == '__main__':
    pass