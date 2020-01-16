# An example pipeline that is used to test the template fitting 
# algorithm module. 
import os
import importlib
import configparser as cp
import logging

# KeckDRPFramework dependencies
from keckdrpframework.pipelines.base_pipeline import BasePipeline
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.action import Action
from keckdrpframework.models.processing_context import ProcessingContext

def get_level(lvl:str) -> int:
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

def start_logger(pipe_name: str, log_config: dict) -> logging.Logger:

    log_path = log_config.get('log_path')
    log_lvl = log_config.get('level')
    log_verbose = log_config.getboolean('verbose')
    # basic logger instance
    logger = logging.getLogger(pipe_name)
    logger.setLevel(get_level(log_lvl))

    formatter = logging.Formatter('[%(name)s]%(levelname)s: %(message)s')
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


class TestPipeline(BasePipeline):
    """
    Pipeline to Process KPF data using the KeckDRPFramework

    Attributes:
        event_table (dictionary): table of actions known to framework. All primitives must be registered here.
    """
    # Modification: 
    name = 'KPF-Pipe'
    event_table = {
        # action_name: (name_of_callable, current_state, next_event_name)
        'evaluate_recipe': ('evaluate_recipe', 'evaluating_recipe', None), 
        'exit': ('exit_loop', 'exiting...', None)
        }
    

    def __init__(self, context):
        BasePipeline.__init__(self, context)

    def start(self, config: cp.ConfigParser) -> None:
        '''
        Initialize the customized pipeline.
        Customized in that it sets up logger and configurations differently 
        from how the BasePipeline does. 
        Args: 
            config: a ConfigParser object containing pipeline configuration 
        '''
        ## setup logger
        try: 
            self.logger = start_logger(self.name, config['LOGGER'])
        except KeyError: 
            raise IOError('cannot find [LOGGER] section in config')
        self.logger.info('Logger started. Level={}')
        ## setup pipeline configuration 
        # Technically the pipeline's configuration is stored in self.context as 
        # a ConfigClass() defined by keckDRP. But we will be using configParser
        try:
            self.config =  config['PIPELINE']
        except KeyError:
            raise IOError('cannot find [PIPELINE] section in config')

        ## Initialize event table and callables
        self._populate_tables('modules') # It is required that modules be stored here

        ## Initialize argument
        # The way we are using the event table means that arguments actually cannot 
        # be passed from one event to another (since our 'next_event' is always none)
        # So the arguments are stored on the pipeline level, as part of the context 
        # --TODO-- This is actually pretty bad. Is there another way?
        self.args = Arguments()
    
    def _populate_tables(self, search_path: str):
        '''
        Populate event table and callables by searching through the module tree
        There is a strict restiction on how the modules are named and where they are,

        '''
        # initailize the callables: 
        self.module_handles = {
            # action_name: action_handle
            'evaluate_recipe': self.evaluate_recipe, 
            'exit': self.exit_loop
        }
        # First loop through the search path for all modules
        for mod_folder in os.listdir(search_path):
            # loop through all files in this folder. 
            # If it's a .py that satisfies the naming convention, try to import it
            relative_path = '{}/{}'.format(search_path, mod_folder) # relative to pipeline repo
            
            if os.path.isdir(relative_path): # ignore all files at this level
                for modpy in os.listdir(relative_path): # look in all folders 
                    if self._check_name(modpy):
                        # This file is a valid KPF module 
                        mod_name = modpy.split('.')[0]
                        # import the .py file relative the pipeline repo
                        import_path = '{}.{}.{}'.format('modules', mod_folder, mod_name)
                        module = importlib.import_module(import_path)
                        # the name of the primitvie class is the same as the file name
                        primitive_handle = getattr(module, mod_name)
                        # Populate event table
                        self.event_table[mod_name] = (mod_name, mod_name, None)
                        self.module_handles[mod_name] = primitive_handle
    
    def _check_name(self, mod: str) -> bool:
        '''
        A helper function for deciding whether a path is a valid KPF module
        '''
        success = mod.endswith('.py')
        success &= mod.startswith('KPFModule_')
        return success

    def evaluate_recipe(self, action, context):
        """
        Evaluates the recipe file (Python file) specified in context.config.run.recipe.
        All actions are executed consecutively in the high priority queue

        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.ProcessingContext.ProcessingContext): Keck DRPF ProcessingContext object
        """
        try: 
            recipe_file = action.args.recipe
            context.logger.info('Executing recipe file {}'.format(recipe_file))
            f = open(recipe_file)
            fstr = f.read()
            exec(fstr)
            f.close()
        except:
            print(sys.exit_info())

    def exit_loop(self, action, context):
        """
        Force the Keck DRP Framework to exit the infinite loop

        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.ProcessingContext.ProcessingContext): Keck DRPF ProcessingContext object
        """
        context.logger.info("Goodbye")
        os._exit(0)

    def get_action(self, action_name):
        try:
            primitive = self.module_handles[action_name]
            if isinstance(primitive, type):
                action = Action((None , None, None), None)
                instance = primitive(action, self.context)
                return instance
            else:
                return self.module_handles[action_name]
        except KeyError:
            # Dummpy place holder
            def f(action, context):
                return True
            return f
