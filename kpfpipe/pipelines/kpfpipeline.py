# An example pipeline that is used to test the template fitting 
# algorithm module. 
import os
import sys
import importlib
import configparser as cp
import keckdrpframework.config.framework_config as cfg
import logging

from kpfpipe.logger import start_logger

# AST recipe support
import ast
from kpfpipe.pipelines.kpf_parse_ast import KpfPipelineNodeVisitor

# KeckDRPFramework dependencies
from keckdrpframework.pipelines.base_pipeline import BasePipeline
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.action import Action
from keckdrpframework.models.processing_context import ProcessingContext


class KPFPipeline(BasePipeline):
    """
    Pipeline to Process KPF data using the KeckDRPFramework
    
    Args:
        context (ProcessingContext): context class provided by the framework
    
    Attributes:
        event_table (dictionary): table of actions known to framework.
        All primitives must be registered here.
    
    Note: the correct operation of the recipe visitor depends on action.args being KpfArguments, which
    is an extension (class derived from) the Keck DRPF Arguments class.  All pipeline primitives must use
    KpfArguments rather than simply Arguments for their return values.  They will also get input arguments
    packaged as KpfArguments. 
    """

    # Modification: 
    name = 'KPF-Pipe'
    event_table = {
        # action_name: (name_of_callable, current_state, next_event_name)
        'start_recipe': ('start_recipe', 'starting recipe', None), 
        'resume_recipe': ('resume_recipe', 'resuming recipe', None),
        'to_fits': ('to_fits', 'processing', 'resume_recipe'),
        'kpf0_from_fits': ('kpf0_from_fits', 'processing', 'resume_recipe'),
        'kpf1_from_fits': ('kpf1_from_fits', 'processing', 'resume_recipe'),
        'kpf2_from_fits': ('kpf2_from_fits', 'processing', 'resume_recipe'),
        'exit': ('exit_loop', 'exiting...', None)
        }
    

    def __init__(self, context: ProcessingContext):
        BasePipeline.__init__(self, context)

    def start(self, config: cp.ConfigParser) -> None:
        '''
        Initialize the customized pipeline.
        Customized in that it sets up logger and configurations differently 
        from how the BasePipeline does.

        :Args: config (ConfigParser): containing pipeline configuration
        '''
        ## setup pipeline configuration 
        # Technically the pipeline's configuration is stored in self.context as 
        # a ConfigClass() defined by keckDRP. But we will be using configParser
    

        self.logger = start_logger(self.name, config)
        self.logger.info('Logger started')
        ## Setup argument
        try: 
            cfg_obj = cp.ConfigParser()
            # cfg_obj = cfg.ConfigClass()
            cfg_obj.read(config)
            arg = cfg_obj._sections['ARGUMENT']
        except KeyError:
            raise IOError('cannot find [ARGUMENT] section in config')
        self.context.arg = arg

        ## Dave's experiment
        # print("kpfpipeline: about to assign into pipeline.config")
        # self.config = cfg_obj
        # print(f"Experiment: type of self.config is {type(self.config)}")

        ## Setup primitive-specific configs:
        self.context.config_path = cfg_obj._sections['MODULES']
        self.logger.info('Finished initializing Pipeline')

    def start_recipe(self, action, context):
        """
        Starts evaluating the recipe file (Python syntax) specified in context.config.run.recipe.
        All actions are executed consecutively in the high priority queue

        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.ProcessingContext.ProcessingContext): Keck DRPF ProcessingContext object
        """
        try: 
            recipe_file = action.args.recipe
            f = open(recipe_file)
            fstr = f.read()
            f.close()
            self._recipe_ast = ast.parse(fstr)
            self._recipe_visitor = KpfPipelineNodeVisitor(pipeline=self, context=context)
            self._recipe_visitor.visit(self._recipe_ast)
        except:
            print(sys.exc_info())
        return Arguments()

    def exit_loop(self, action, context):
        """
        Force the Keck DRP Framework to exit the infinite loop

        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.ProcessingContext.ProcessingContext): Keck DRPF ProcessingContext object
        """
        self.logger.info("exiting pipeline...")
        # os._exit(0)

    # reentry after call

    def resume_recipe(self, action: Action, context: ProcessingContext):
        """
        Continues evaluating the recipe started in start_recipe().

        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.ProcessingContext.ProcessingContext): Keck DRPF ProcessingContext object
        """
        # pick up the recipe processing where we left off
        self._recipe_visitor.returning_from_call = True
        self._recipe_visitor.awaiting_call_return = False
        self._recipe_visitor.call_output = action.args # framework put previous output here
        self._recipe_visitor.visit(self._recipe_ast)
        return Arguments()  # nothing to actually return, but meet the Framework requirement
