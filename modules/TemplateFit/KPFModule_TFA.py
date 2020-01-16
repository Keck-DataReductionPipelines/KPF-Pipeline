

# Standard dependencies
import configparser as cp
import logging

# External dependencies
from keckdrpframework.primitives.base_primitive import BasePrimitive
from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext

# Local dependencies
import modules.TemplateFit.src.macro as mc
# import modules.TemplateFit.src.alg
# import modules.TemplateFit.src.arg

def get_level(lvl:str) -> int:
    '''
    read the logging level (string) from config file and return 
    the corresponding logging level (technically of type int)
    '''
    if lvl == 'debug': return logging.DEBUG
    elif lvl == 'info': return logging.INFO
    elif lvl == 'warning': return logging.WARNING
    elif lvl == 'error': return logging.ERROR
    elif lvl == 'critical': return logging.CRITICAL
    else: return logging.NOTSET

def start_logger(logger_name: str, log_config: dict) -> logging.Logger:

    log_path = log_config.get('log_path')
    log_lvl = log_config.get('level')
    log_verbose = log_config.getboolean('verbose')

    # basic logger instance
    logger = logging.getLogger(logger_name)
    logger.setLevel(get_level(log_lvl))

    formatter = logging.Formatter('[%(name)s] - %(levelname)s - %(message)s')
    f_handle = logging.FileHandler(log_path, mode='w') # logging to file
    f_handle.setLevel(get_level(log_lvl))
    f_handle.setFormatter(formatter)
    logger.addHandler(f_handle)

    if log_verbose: 
        # also print to terminal 
        s_handle = logging.StreamHandler()
        s_handle.setLevel(get_level(log_lvl))
        s_handle.setFormatter(formatter)
        logger.addHandler(s_handle)
    return logger

class KPFModule_TFA(BasePrimitive):
    '''
    The template fitting module
    '''
    def __init__(self, 
            action: Action, 
            context: ProcessingContext) -> None:
        '''
        Initializer
        args: 
            arg: keckDRP defined object that contains input
            context: kcekDRP defined object that keeps track of 
        '''

        # --TODO-- Implement in precondition check
        # we assume that a ConfigParser class is included in the argument
        BasePrimitive.__init__(self, action, context)

        log_config = arg.config
        # The logger has the same name as the class
        self.logger = start_logger(self.__class__.__name__, log_config)
        # Consider all fits files in the provided folder a candidate
        self.flist = mc.findfiles(dirpath, '.fits')
        # Read the config file with config parser
        self.cfg = cp.ConfigParser(comment_prefixes='#')
        self.cfg.read(config_file)
        # Now parse the config 
        self.parse_config(self.cfg)
        
        # # Pre and post process
        # self.pre = prim.ProcessSpec()
        # self.post = prim.PostProcess()

        # self.res = arg.TFAFinalResult()

        self._perform()

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

    # def make_template(self, prelim:str, flist:list) -> arg.Spec:
    #     # Initialize the preliminary as the template

    #     name = prelim.split('/')[-1]
    #     self.logger.log('beginning to create template', 'info')
    #     self.logger.log('preliminary file used: {}'.format(name), 'info')
    #     SP = sp.Spec(filename=prelim)
    #     SP = self.pre.run(SP) 

    #     n_files = len(flist)
    #     # get the wavelength and specs of the preliminary  
    #     # as foundation to the template
    #     twave = SP._wave
    #     tspec = SP._spec

    #     # Currently just a average of all spectrum
    #     # should also be taking care of the outliers (3-sigma clipping)
    #     for i, file in enumerate(flist):
    #         name = file.split('/')[-1]
    #         self.logger.log('({}/{}) processing {}'.format(i, len(flist), name), 'info')
    #         S = sp.Spec(filename=file)
    #         SS = self.pre.run(S)
    #         T = SingleTFA(SP, SS, None, self.logger)
    #         res = T.run()
    #         rel = res.res_df[['alpha', 'success']].to_records()
    #         for order in rel:
    #             if order[2]: #success
    #                 flamb, fspec = SS.get_order(order[0])
    #                 fspec2 = np.interp(twave[order[0], :], flamb, fspec)
    #                 tspec[order[0], :] += fspec2
    #             else: 
    #                 n_files -= 1
    #     tspec = np.divide(tspec, n_files)
    #     self.logger.log('finised making templated', 'info')
    #     return sp.Spec(data=(twave, tspec), jd=SP.julian_day)

    # def calc_velocity(self, temp: str, flist:list) -> arg.TFAFinalResult:
    #     '''

    #     '''
    #     self.logger.log('beginning to calculate radial velocity', 'info')
    #     for i, file in enumerate(flist):
    #         name = file.split('/')[-1]
    #         self.logger.log('({}/{}) processing {}'.format(i, len(flist), name), 'info')
    #         S = sp.Spec(filename=file)
    #         SS = self.pre.run(S)

    #         T = SingleTFA(temp, SS, self.cfg, self.logger)
    #         r = T.run()
    #         self.res.append(i, r.write_to_final())
    #     self.logger.log('finised calculating velocity', 'info')
    #     return self.res
    
    def _perform(self):
        print('-----------YaY-----------')
        self.logger.info('YAY! TFAMain found!')

if __name__ == "__main__":
    pass

        