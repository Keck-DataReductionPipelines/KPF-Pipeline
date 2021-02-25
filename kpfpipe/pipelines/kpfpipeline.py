# An example pipeline that is used to test the template fitting 
# algorithm module. 
import os
import sys
import importlib
import configparser as cp
import logging
import glob
from dotenv.main import load_dotenv

from kpfpipe.logger import start_logger

# AST recipe support
import ast
from kpfpipe.pipelines.kpf_parse_ast import KpfPipelineNodeVisitor
import kpfpipe.config.pipeline_config as cfg

# KeckDRPFramework dependencies
from keckdrpframework.pipelines.base_pipeline import BasePipeline
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.action import Action
from keckdrpframework.models.processing_context import ProcessingContext


class KPFPipeline(BasePipeline):
    """
    Pipeline to Process KPF data using the KeckDRPFramework

    Overview:
        The KPFPipeline is a mechanism conforming to the KeckDRPFramework Pipeline definitions, but uses the
        Framework mechanisms in a novel way to provide needed additional flexibility through the use of
        "recipes". In the Framework, the focus is to enqueue and run a sequence of data reduction "primitives".
        The Framework accomplishes this through event queues (two of them, with differing priorities) and
        the event_table. The event_table provides a mechanism to map a primitive definition into actual
        python code in a module.

        In order to support recipes with the flexibility to do conditional processing and loops, the
        KPFPipeline implements two special primities, "start_recipe" and "resume_recipe".  "start_recipe"
        is always the first primitive run by the framework. It parses the recipe file into an Abstract Syntax
        Tree (AST), and walks through the nodes of the tree, which are syntax elements of the recipe, acting
        appropriately on each node in turn.  When the AST walk hits a "call" node that references an actual
        data processing primitive, it enqueues the primitive as the next event, and returns from "start_recipe".
        The KPFPipeline uses the Framework's "next_event" mechanism when it enqueues the primitive to ensure
        that the Framework runs "resume_recipe" immediately after each data reduction primitive.
        "resume_recipe" continues walking the AST nodes tree, picking up at the "call" node that triggered the
        running of the data reduction primitive, including handling return values from that primitive.

        In addition to "start_recipe" and "resume_recipe", the KPFPipeline prepopulates the event table with
        primitives to write and read FITS files.  These are "to_fits", "kpf0_from_fits", "kpf1_from_fits" and
        "kpf2_from_fits". Other built-in functions that are available for use within recipes without external
        calls to the Framework are discussed in register_recipe_builtins() below.

    Note:
        Supported Recipe Syntax Elements:

            mod = Module(stmt* body, type_ignore* type_ignores)

            stmt = FunctionDef(identifier name, arguments args,
                            stmt* body, expr* decorator_list, expr? returns,
                            string? type_comment)
                                    string? type_comment)

                | Assign(expr* targets, expr value, string? type_comment)

                -- use 'orelse' because else is a keyword
                | For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
                | If(expr test, stmt* body, stmt* orelse)

                | ImportFrom(identifier? module, alias* names, int? level)

                | Expr(expr value)

                -- col_offset is the byte offset in the utf8 string the parser uses
                attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

            expr = BinOp(expr left, operator op, expr right)
                | UnaryOp(unaryop op, expr operand)
                -- need sequences for compare to distinguish between
                -- x < 4 < 3 and (x < 4) < 3
                | Compare(expr left, cmpop* ops, expr* comparators)
                | Call(expr func, expr* args, keyword* keywords)
                | Constant(constant value, string? kind)

                -- the following expression can appear in assignment context
                | Attribute(expr value, identifier attr, expr_context ctx)
                | Subscript(expr value, expr slice, expr_context ctx)
                | Name(identifier id, expr_context ctx)
                | List(expr* elts, expr_context ctx)
                | Tuple(expr* elts, expr_context ctx)

            operator = Add | Sub | Mult | Div

            unaryop = Not | UAdd | USub

            cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn

            arguments = (arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs,
                        expr* kw_defaults, arg? kwarg, expr* defaults)

            arg = (identifier arg, expr? annotation, string? type_comment)
                attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

            -- keyword arguments supplied to call (NULL identifier for **kwargs)
            keyword = (identifier? arg, expr value)
                    attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)

            -- import name with optional 'as' alias.
            alias = (identifier name, identifier? asname)

    Args:
        context (ProcessingContext): context class provided by the framework

    Attributes:
        event_table (dictionary): Values are tuples, e.g. "action_name: (name_of_callable, current_state, next_event_name)"
    """

    # Modification: 
    name = 'KPF-Pipe'
    event_table = {
        """
        event_table (dictionary): table of actions known to framework. All primitives must be registered here.
        Data reduction primitives are registered into the event_table as part of the processing
        of the "from ... import" statement at the top of a recipe.

        The format of entries is:
            action_name: (name_of_callable, current_state, next_event_name)
        """
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
        load_dotenv()
    
    def register_recipe_builtins(self):
        """
        register_recipe_builtins() registers some built-in functions for the recipe to use
        without having to invoke them through the Framework's queue.  If additional built-in
        functions are needed, this is the place to add them.
        
        The supported built-ins are:

        int:
            Same behavior as in Python

        float:
            Same behavior as in Python

        str:
            Same behavior as in Python

        len:
            Same behavior as in Python

        find_files:
            Same behavior as glob.glob in Python, which returns a list of files
            that match the string pattern given as its argument.  In particular,
            * expansion is supported.

        split:
            Same behavior as os.path.split in Python. It returns two strings,
            the first representing the directories of a file path, and the second
            representing the simple file name within the directory.

        split_ext:
            Same behavior as os.path.splitext in Python. It returns two strings,
            the second being the file extension of a file path, including the dot,
            and the first being everything else.

        dirname:
            Same behavior as os.path.dirname.  It returns the directory portion of
            a file path, excluding the file name itself, with no trailing separator.

        """
        self._recipe_visitor.register_builtin('int', int, 1)
        self._recipe_visitor.register_builtin('float', float, 1)
        self._recipe_visitor.register_builtin('str', str, 1)
        self._recipe_visitor.register_builtin('len', len, 1)
        self._recipe_visitor.register_builtin('find_files', glob.glob, 1)
        self._recipe_visitor.register_builtin('split', os.path.split, 1)
        self._recipe_visitor.register_builtin('splitext', os.path.splitext, 1)
        self._recipe_visitor.register_builtin('dirname', os.path.dirname, 1)

    def preload_env(self):
        """
        preload_env() preloads environment variables using dotenv """
        """
        env_values = dotenv_values()
        for key in env_values:
            self.context.logger.debug(f"_preload_env: {key} <- {env_values.get(key)}")
            self._recipe_visitor.load_env_value(key, env_values.get(key))
        """
        for key in os.environ:
            self.context.logger.debug(f"_preload_env: {key} <- {os.environ.get(key)}")
            self._recipe_visitor.load_env_value(key, os.environ.get(key))

    def start(self, configfile: str) -> None:
        '''
        Initialize the customized pipeline.
        Customized in that it sets up logger and configurations differently 
        from how the BasePipeline does.

        Args: 
            config (ConfigParser): containing pipeline configuration
        '''
        ## setup pipeline configuration 
        # Technically the pipeline's configuration is stored in self.context as 
        # a ConfigClass() defined by keckDRP. But we will be using configParser

        self.logger = start_logger(self.name, configfile)
        self.logger.info('Logger started')

        ## Setup argument
        try: 
            self.config = cfg.ConfigClass()
            self.config.read(configfile)
            arg = self.config._sections['ARGUMENT']
        except KeyError:
            raise IOError('cannot find [ARGUMENT] section in config')
        self.context.arg = arg

        ## Setup primitive-specific configs:
        self.context.config_path = self.config._sections['MODULES']
        self.logger.info('Finished initializing Pipeline')

    def start_recipe(self, action, context):
        """
        Starts evaluating the recipe file (Python syntax) specified in context.config.run.recipe.
        All actions are executed consecutively in the Framework's high priority queue.

        Before starting processing the recipe, built-in functions available to recipes without
        having to enqueue them to the Framework are registered, and values defined in the environment
        are imported so that they are also available to recipes.

        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.ProcessingContext.ProcessingContext): Keck DRPF ProcessingContext object
        """
        recipe_file = action.args.recipe
        with open(recipe_file) as f:
            fstr = f.read()
            self._recipe_ast = ast.parse(fstr)
        self._recipe_visitor = KpfPipelineNodeVisitor(pipeline=self, context=context)
        self.register_recipe_builtins()
        ## set up environment
        try:
            self.preload_env()
        except Exception as e:
            self.logger.error(f"KPF-Pipeline couldn't load environment due to exception {e}")
        
        self._recipe_visitor.visit(self._recipe_ast)
        return Arguments(name="start_recipe_return")

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
        Continues evaluating the recipe started in start_recipe().  resume_recipe() will run immediately
        after each data processing primitive, and makes return values from the previous primitive, stored in an
        Arguments class instance in action.args, available back to the recipe.

        Args:
            action (keckdrpframework.models.action.Action): Keck DRPF Action object
            context (keckdrpframework.models.ProcessingContext.ProcessingContext): Keck DRPF ProcessingContext object
        """
        # pick up the recipe processing where we left off
        self.logger.debug("resume_recipe")
        self._recipe_visitor.returning_from_call = True
        self._recipe_visitor.awaiting_call_return = False
        self._recipe_visitor.call_output = action.args # framework put previous output here
        self._recipe_visitor.visit(self._recipe_ast)
        return Arguments(name="resume_recipe_return")  # nothing to actually return, but meet the Framework requirement
