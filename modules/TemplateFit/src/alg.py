
# -- Template fitting module core algorithm --
# This file contains the core of the template fitting algorithm

# import python built-in packages 
import sys, os
import logging

# import external dependencies

import numpy as np
import scipy.interpolate as ip
import configparser as cp
import pandas as pd


# import local template fitting modules 
import modules.TemplateFit.src.macro as mc
import modules.TemplateFit.src.primitives as prim
from modules.TemplateFit.src.argument import spec as arg
from modules.TemplateFit.src.argument import tfa_res as tr

# Opterations necessary for the entire template fitting algorithm
def prob_for_prelim(flist: list) -> str:
    ''' 
    find the file with the highest mean flux
    this file is used as the preliminary template
    '''
    best_file = None, 
    best_val = 0
    for f in flist:
        S = arg.Spec(filename=f)
        mean = np.mean(S._spec)
        if mean > best_val:
            best_val = mean
            best_file = f
    return best_file

class Debugger:
    '''
    A class used for debugging purposes by the SingleTFA
    This means that each file should have its own debugger
    '''

    def __init__(self, m) -> None:
        '''
        Initializer
        '''
        self.m = m              # Degree of flux normalization polynomial
        self.running = False    # Only run the debugger if the config says so
        self.xlsx_writer = None # writting debug results to xlsx files, so this
                                # is the writer instance 
    
    def __del__(self) -> None:
        '''
        Deconstructor, called when this instance is deleted
        '''
        if self.xlsx_writer != None:
            # Save and close the xlsx file, if we have one
            self.xlsx_writer.save()
            self.xlsx_writer.close()
    
    def start(self, fname: tuple, debug_path: str, order: list) -> None: 
        '''
        start the debugger class 
        '''
        self.obs_name, self.temp_name = fname
        self.path = debug_path
        self.xlsx_writer = pd.ExcelWriter(self.path + '.xlsx', engine='xlsxwriter')
        self.result = {}
        for o in order:
            self.result[o] = arg.TFADebugRes(self.obs_name, self.temp_name, o, self.m)
        self.running = True
        
    def append_result(self, order: int, iteration:int, result: list) -> None:
        '''
        record the intermediate results
        '''
        if self.running:
            assert(len(result) == 6)
            self.result[order].append(iteration, result)
    
    def append_weight(self, order: int, weight: np.ndarray) -> None:
        '''
        record weight 
        '''
        if self.running:
            self.result[order].append_weight(weight)
    
    def log_exit(self, order: int, msg:str) -> None:
        '''
        exit status and message for a specific order
        '''
        if self.running:
            self.result[order].log_exit(msg)
    
    def record(self, order: int, name:str) -> None:
        '''
        
        '''
        if self.running:
            self.result[order].record(self.xlsx_writer, self.path + '.dat')

class Logger: 
    '''
    A class used for logging for the TFA Module
    '''
    def __init__(self):
        '''
        Initializer
        '''
        self.running = False
    
    def get_level(self, lvl:str):
        '''
        read the logging level (string) from config file and return 
        the corresponding logging level
        '''
        if lvl == 'debug': return logging.DEBUG
        elif lvl == 'info': return logging.INFO
        elif lvl == 'warning': return logging.WARNING
        elif lvl == 'error': return logging.ERROR
        elif lvl == 'critical': return logging.CRITICAL
        else: return logging.NOTSET

    def start(self, log_path: str, lvl: str, verbose: bool):
        '''
        starting the logging instance 
        '''
        # basic logger instance
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.get_level(lvl))
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        f_handle = logging.FileHandler(log_path, mode='w') # logging to file
        f_handle.setLevel(self.get_level(lvl))
        f_handle.setFormatter(formatter)
        self.logger.addHandler(f_handle)
        if verbose: 
            # also print to terminal 
            s_handle = logging.StreamHandler()
            s_handle.setLevel(self.get_level(lvl))
            s_handle.setFormatter(formatter)
            self.logger.addHandler(s_handle)
        self.running = True
    
    def log(self, msg: str, lvl: str):
        '''
        log message
        '''
        if self.running: 
            if lvl == 'debug':
                self.logger.debug(msg)
            elif lvl == 'info':
                self.logger.info(msg)
            elif lvl == 'warning':
                self.logger.warning(msg)
            elif lvl == 'error':
                self.logger.error(msg)
            elif lvl == 'critical':
                self.logger.critical(msg)
            else:
                raise ValueError('Logger does not recognize level: {}'.format(lvl))

class SingleTFA:
    ''' 
    The template fitting algorithm
    '''
    def __init__(self, temp: arg.Spec, obs: arf.Spec, 
                 cfg: cp.ConfigParser, log:logging.Logger) -> None:
        '''
        Initializer
        '''
        # the template and observation spectrum 
        self.temp = temp
        self.obs = obs
        # range of orders to consider
        # This is specified in the macro file
        # --TODO--
        # Set this to a configuration?
        self.ord_range = mc.ord_range

        # default parameter values, unless modified by 
        # a configuration file, if provided
        self.m = 3         # blaze polynomial order
        self.max_iter = 50 # maximum allowed steps 

        self.logger = log # object for logging. Should have been initialized on top level
        self.debugger = Debugger(self.m) # object for debugging purposes

        # each file is identified by their julian date in normal mode
        # in debug mode each file name is also saved

        self.res = arg.TFAOrderResult(self.m, obs.julian_day)
        self.outlier = prim.RemoveOutlier()

        # a configuration file is provided
        # set up any values that are not default
        if cfg != None:
            self.parse_config(cfg)

    def parse_config(self, config:cp.ConfigParser) -> None:
        '''
        Used to parse the configuration object passed from top level
        '''
        try: # set parameters
            self.m = int(config.get('PARAM', 'blaze_deg'))
            self.max_iter = int(config.get('PARAM', 'max_nstep'))
        except: 
            pass

        try: # update debug related configs
            if config.getboolean('DEBUG', 'debug'):
                names = (self.obs.filename, self.temp.filename)
                Norder = range(72)
                self.debug_path = config.get('DEBUG', 'debug_path')
                name = os.path.basename(self.obs.filename)
                self.debug_path += '/{}'.format(os.path.splitext(name)[0])
                self.debugger.start(names, self.debug_path, Norder)
                self.logger.log('debug files saved to {}'.format(
                    self.debug_path
                ), 'info')
        except:
            print(sys.exc_info())    

    def correct(self, w):
        '''any nonvalid weight is set to zero'''
        w[np.where(np.isfinite(w) == 0)] = 0
        w[np.where(w <= 0)] = 0
        return w

    def solve_step(self, order: int,
                   a: mc.ALPHA_TYPE,
                   w0: np.ndarray) -> mc.ALPHA_TYPE:
        '''
        One step in the Chi^2 minimizer (effectively a 
        Newton's method for optimization).
        Returns the updating step 
        '''
        # Reference data
        tlamb, tspec = self.temp.get_order(order)
        # Observed data
        flamb ,fspec = self.obs.get_order(order)

        av_lamb = np.multiply(a[0], tlamb)
        # overlapping interval between tspec (F) and observed(f)
        # we can only compare the two in this interval
        # print(av_lamb.size, flamb.size)
        lamb, w= mc.common_range(flamb, av_lamb, w0)
        tckF = ip.splrep(av_lamb, tspec)
        tckf = ip.splrep(flamb,fspec)
        F_av = ip.splev(lamb, tckF)
        f = ip.splev(lamb, tckf)

        # create f[lamb]*sum(a_m * (lamb - lamb_c)) in eqn 1
        am = a[1:]            # polynomial coefficients [a_0, a_1, ... a_m]
        c = int(lamb.size/2)  # index for center wavelength of each order
        px = lamb - lamb[c]
        # np.polyval is setup as:
        #    polyval(p, x) = p[0]*x**(N-1) + ... + p[N-1]
        # Thus we need to reverse am
        amf = np.flip(am)

        # f_coor = f[lamb]*sum(a_m * (lamb - lamb_c)) 
        poly = np.polyval(amf, px)
        f_corr = np.multiply(f, poly)
        
        # Final form of eqn 1
        R = F_av - f_corr
        X2 = np.sum(np.multiply(w, np.square(R)))

        ## calculate partial derivatives
        # eqn 3-4
        dR_dm = []
        grad = np.gradient(F_av,lamb)
        grad = np.nan_to_num(grad, nan=0)

        dF = np.multiply(lamb, grad)
        dR_dm.append(dF)
        for i in np.arange(self.m+1):
            dR_dm.append(-np.multiply(f, np.power(px, i)))

        ## setup hessian and eqn 8:
        #  summing all of pixels (res * 4096 * 76) in matrix
        A_lk = np.zeros((self.m+2, self.m+2))
        b_l = np.zeros((self.m+2, 1))
        # eqn 6 & 15
        for i in np.arange(0, self.m+2):
            for j in np.arange(0, self.m+2):
                A_lk[i, j] = np.sum(np.multiply(w,
                             np.multiply(dR_dm[i], dR_dm[j])))
            b_l[i] = -np.sum(np.multiply(w, np.multiply(dR_dm[i], R)))

        da = np.linalg.solve(A_lk, b_l)
        return da, R, A_lk, X2

    def solve_order(self, order: int): 
        '''
        Apply the template fitting algorithm on a single order
        '''
        _, flux = self.temp.get_order(order)
        flux = self.correct(flux)

        # w = np.sqrt(flux)
        w = np.ones_like(w)
        w = self.correct(w)
        # average flux of yje prder
        f_mean = np.sqrt(np.mean(flux))
        success = True

        ## Initial values
        a = np.asarray([1,1] + [0] *self.m, dtype=np.float64)
        da = np.asarray([np.inf]*(self.m+2), dtype=np.float64)
        err_v = 1
        err_prev = np.inf
        # Keep track of number of iterations to convergence
        iteration = 0
        # Convergence criteria
        con_val = np.asarray([np.inf]*(self.m+2), dtype=np.float64)
        converge = False
        success = True
        # values for record
        k = 1
        X2 = np.inf
        exit_msg = ''
        # log initial conditions:
        result = np.asarray([a, da, err_v, con_val[0], k, X2])
        self.debugger.append_result(order, 0, result)
        self.debugger.append_weight(order, w)
        
        # convergence criteria specified in 2.1.5
        while converge != True and success == True:
            iteration += 1
            w = self.correct(w)

            ## solve
            try:
                da, R, A_lk, X2 = self.solve_step(order, a, w)
            except(np.linalg.LinAlgError):
                # this happens if the next step failed to compute
                success = False
                exit_msg = '[{}] LinAlg error encountered when solving step {}'.format(order, iteration)

            if iteration > self.max_iter:
                # infinite loop (potentialy)
                success = False
                exit_msg = '[{}] maximum allowed iteration reached: {}'.format(order, self.max_iter)

            ## update
            # only update if operation is successful
            if success:
                # update alpha
                da = np.reshape(da, a.shape)
                da[0] *= -1.0 # still have no idea why we negate this
                a += da

                # update weights
                ## Try to update the noise only after the first iteration 
                if iteration > 1: 
                    R_sig = np.std(R)
                    k = np.divide(R_sig, np.sqrt(f_mean))  # kappa in 2.1.2
                    # w = np.reciprocal(np.multiply(np.square(k), flux))

                    # remove outlier
                    bad = self.outlier.sigma_clip(R, 3)
                    # w[bad] = 0

                # compute errors and convergence
                error = np.sqrt(np.linalg.inv(A_lk).diagonal())
                err_v = np.multiply(error[0], mc.C_SPEED)
                con_val = abs(da * mc.C_SPEED)

                # converge criteria 
                converge = con_val[0] < 1e-7
                # converge = abs(err_v -err_prev) < 1e-6
                # err_prev = err_v

            ## record:
            if success: 
                result = np.asarray([a, da, err_v, con_val[0], k, X2])
                self.debugger.append_result(order, iteration, result)
                self.debugger.append_weight(order, w)
                msg = '[{}] finished iteration {} successfully. X2 = {:.1f}'.format(order, iteration, X2)
                self.logger.log(msg, 'debug')
            else: 
                self.debugger.log_exit(order, exit_msg)
                self.logger.log(exit_msg, 'warning')
                break

        # Out of the while loop now
        # record final result
        if success: 
            result = np.asarray([a, da, err_v, con_val[0], k, X2])
            self.debugger.append_result(order, 'final', result)
            self.debugger.append_weight(order, w)
            msg = '[{}] finished order {} successfully. Final X2 = {:.1f}'.format(order, order, X2)
            self.logger.log(msg, 'debug')
        return a, err_v, success, iteration

    def run(self) -> arg.TFAOrderResult:
        '''
        Run the template fitting algorithm on all orders
        '''
        
        for i, order in enumerate(mc.ord_range): 
            a, err, s, it = self.solve_order(order)
            # only record alpha_v in final result
            inter_res = np.asarray([a, err, s, it])
            self.res.append(order, inter_res)
            self.debugger.record(order, '{}'.format)
        return self.res

class TFAPrimitive(KPF_Primitive):
    '''
    The template fitting module
    '''
    def __init__(self, dirpath: str ,config_file: str) -> None:
        '''
        Initializer
        '''
        self.logger = Logger()
        # Consider all fits files in the provided folder a candidate
        self.flist = mc.findfiles(dirpath, '.fits')
        # Read the config file with config parser
        self.cfg = cp.ConfigParser(comment_prefixes='#')
        self.cfg.read(config_file)
        # Now parse the config 
        self.parse_config(self.cfg)
        
        # Pre and post process
        self.pre = prim.ProcessSpec()
        self.post = prim.PostProcess()

        self.res = arg.TFAFinalResult()

        KPF_Primitive.__init__(self)

    def parse_config(self, config: cp.ConfigParser) -> None:
        '''
        get all logging related configurations
        '''
        try: # logging related configs
            if config.getboolean('LOGGER', 'log'):
                log_path = config.get('LOGGER', 'log_path')
                log_level = config.get('LOGGER', 'log_level')
                verbose = config.getboolean('LOGGER', 'log_verbose')
                self.logger.start(log_path, log_level, verbose)
                # by this point the logger should have started, so we 
                # can start logging 
                msg = 'Beginning logger instance. log_path = {}, log_vele = {}'.format(
                    log_path, log_level
                )
                self.logger.log(msg, 'info')
        except:
            pass

    def make_template(self, prelim:str, flist:list) -> arg.Spec:
        # Initialize the preliminary as the template

        name = prelim.split('/')[-1]
        self.logger.log('beginning to create template', 'info')
        self.logger.log('preliminary file used: {}'.format(name), 'info')
        SP = sp.Spec(filename=prelim)
        SP = self.pre.run(SP) 

        n_files = len(flist)
        # get the wavelength and specs of the preliminary  
        # as foundation to the template
        twave = SP._wave
        tspec = SP._spec

        # Currently just a average of all spectrum
        # should also be taking care of the outliers (3-sigma clipping)
        for i, file in enumerate(flist):
            name = file.split('/')[-1]
            self.logger.log('({}/{}) processing {}'.format(i, len(flist), name), 'info')
            S = sp.Spec(filename=file)
            SS = self.pre.run(S)
            T = SingleTFA(SP, SS, None, self.logger)
            res = T.run()
            rel = res.res_df[['alpha', 'success']].to_records()
            for order in rel:
                if order[2]: #success
                    flamb, fspec = SS.get_order(order[0])
                    fspec2 = np.interp(twave[order[0], :], flamb, fspec)
                    tspec[order[0], :] += fspec2
                else: 
                    n_files -= 1
        tspec = np.divide(tspec, n_files)
        self.logger.log('finised making templated', 'info')
        return sp.Spec(data=(twave, tspec), jd=SP.julian_day)

    def calc_velocity(self, temp: str, flist:list) -> arg.TFAFinalResult:
        '''

        '''
        self.logger.log('beginning to calculate radial velocity', 'info')
        for i, file in enumerate(flist):
            name = file.split('/')[-1]
            self.logger.log('({}/{}) processing {}'.format(i, len(flist), name), 'info')
            S = sp.Spec(filename=file)
            SS = self.pre.run(S)

            T = SingleTFA(temp, SS, self.cfg, self.logger)
            r = T.run()
            self.res.append(i, r.write_to_final())
        self.logger.log('finised calculating velocity', 'info')
        return self.res

if __name__ == '__main__':
    pass