# kpf_parse_ast.py

from ast import NodeVisitor, parse
import _ast
from collections.abc import Iterable
from queue import Queue
import os

from keckdrpframework.models.action import Action
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.models.processing_context import ProcessingContext
import configparser as cp

class RecipeError(Exception):
    """
    RecipeError is raised whenever an error in the recipe or recipe processing is encountered.  The most
    common reason for raising RecipeError is an error in the recipe itself, such as a typo or undefined
    variable or function name, or the use of syntactic elements from Python that are not supported in
    recipes.
    """

class KpfPipelineNodeVisitor(NodeVisitor):
    """
    KpfPipelineNodeVisitor is a node visitor class derived from ast.NodeVisitor to convert KPF pipeline
    recipes expressed in python syntax into operations on the KPF Framework.  This class implements
    methods with names of the form visit_<syntax_element>, each of which handles the actual processing
    of <syntax_element> instances found in a recipe.  ast.NodeVisitor implements visit(), which is the
    code that calls the appropriate visit_<syntax_element> method, or generic_visit() if no such method
    is found. For recipes containing only supported syntax elements, generic_visit() is never called.

    There are three class member flags that keep the state of node walking across calls to data processing
    primitives and loop iterations.  They are:

    self.awaiting_call_return:
        This flag is set just before the visit_Call() method returns, after queueing a data processing
        primitive to run in the Framework.  Any node visiting method that can have a Call within it must
        "return" immediately when this flag is set.  It is cleared by resume_recipe() after the Framework
        finishes executing the called primitive.

    self.returning_from_call:
        This flag is set by resume_recipe() just before it starts walking the recipe syntax tree, which
        starts from the top each time. (Each visit_<syntax_element> method is responsible for behaving
        correctly in the face of these multiple "duplicate" calls, either by redoing work quickly, or by
        saving state and results of work already done.  See the notes about the kpf_completed flag below.)
        visit_Call() uses this flag to trigger the processing of the return value(s) from the data processing
        primitive previously enqueued to the framework that resume_recipe() is following  This flag
        distinguishes returning from a finished call from preparing for a new call.

    self._reset_visited_states:
        This flag is set by reset_visited_states(), which is called from visit_For() and potentially other
        loop control methods to be implemented in the future.  It causes the visit_<syntax_element> methods
        to reset their saved state to prepare for another iteration of a loop.

    In addition, all visit_<syntax_element>() methods doing work that takes a significant amount of time
    use an internal attribute, kpf_completed, to indicate that the work represented by that node and any of its children
    has been completed, and as necessary the result has been stored in an attribute of the AST
    node itself. Some visit_<syntax_element>() methods store additional state and parameters.  Those
    behaviors are covered below in the method documentation.
    """

    def __init__(self, pipeline=None, context=None):
        """
        __init__() constructs an instance of the class that actually walks ("visits") the recipe nodes,
        after they have been parsed into an Abstract Syntax Tree (AST).  The actual work of the pipeline
        recipe is done by the various visit_<syntax_element> class methods.
        """
        NodeVisitor.__init__(self)
        # instantiate the parameters dict
        self._params = None
        # instantiate the environment dict
        self._env = {}
        # store and load stacks
        # (implemented as lists; use append() and pop())
        self._store = list()
        self._load = list()
        # KPF framework items
        self.pipeline = pipeline
        self.context = context
        # local state flags
        self.awaiting_call_return = False
        self.returning_from_call = False
        self._reset_visited_states = False
        # value returned by primitive executed by framework
        self.call_output = None
        self._builtins = {}
        self.subrecipe_depth = 0

    def register_builtin(self, key, func, nargs):
        """
        register_builtin() registers a function so that the function so registered can be
        called from within a recipe directly, without using the Framework's event queue
        mechanism.  This is useful for recipe support functions like converting floats to
        ints and splitting strings.

        Args:
            key: the name of the built-in function as a string
            func: the python function itself
            nargs: the number of input args the function expects

        Note:
            Functions that can take a variable number of arguments are not supported by
            the recipe parser.
        """
        self._builtins[key] = (func, nargs)

    def load_env_value(self, key, value):
        self._env[key] = value
    
    def visit_Module(self, node):
        """
        visit_Module() processes "module" node of a parsed recipe.
        A Module node is always at the top of an AST tree returned
        by ast.parse(), and there is only one per recipe file.
        The parameters dictionary self._params used to store the values of recipe variables
        is initialized or reset things when processing the module node, and cleaned up
        (releasing allocated memory) at the end.  Cleaning up has the
        effect of freeing storage used by variables from a previous pipeline recipe.

        The recipe parser supports invocation of sub-recipes through the
        special-case function name "invoke_subrecipe()" (see visit_Call()).
        Care is taken here to avoid reinitializing the parsing state when processing
        the "module" node of a sub-recipe.
        """
        if self._reset_visited_states:
            setattr(node, 'kpf_started', False)
            for item in node.body:
                self.visit(item)
            if self.subrecipe_depth == 0:
                self._params = None # let storage get collected
            return
        self.pipeline.logger.info(f"Module: subrecipe_depth = {self.subrecipe_depth}")
        if not getattr(node, 'kpf_started', False):
            if self.subrecipe_depth == 0:
                self._params = {}
            setattr(node, 'kpf_started', True)
        for item in node.body:
            self.visit(item)
            if self.awaiting_call_return:
                return
        if self.subrecipe_depth == 0:
            self._params = None # let allocated memory get collected

    def visit_ImportFrom(self, node):
        """
        visit_ImportFrom() processes the "from ... import" statement in a recipe.  It causes the imported
        function names to be added to the pipeline's event_table, and the paths in the "from" section to
        be added to the Framework's module search path.  In combination, these two actions allow data
        reduction pipeline primitives to be successfully run when they are invoked within a recipe.
        The actual queueing of a pipeline primitive onto the Framework's event queue occurs in
        visit_Call().

        Note:
            An "import" clause without a corresponding "from" clause is not supported in recipes, since it
            would not provide a mechanism to represent a module search path.  A bare "import" clause would
            want to call "visit_Import()", which this class does not implement, so generic_visit() would
            be called instead.  generic_visit() logs and raises an error.

        Note:
            Because "from ... import" statements in subrecipes could be imported more than once if the
            subrecipe is within a loop, inclusing of "from ... import" statements should only appear
            in the primary recipe, not subrecipes.
        """
        if self._reset_visited_states:
            setattr(node, 'kpf_completed', False)
            for name in node.names:
                self.visit(name)
            return
        if not getattr(node, 'kpf_completed', False):
            module = node.module
            # append the module path to the framework's primitive_path
            self.context.config.primitive_path = tuple([*self.context.config.primitive_path, module])
            loadQSizeBefore = len(self._load)
            for name in node.names:
                self.visit(name)
                if len(self._load) > loadQSizeBefore:
                    # import the named primitive
                    # This comes as a 2-element tuple from visit_alias
                    #
                    # just add the name to the event_table for now
                    # But we should ensure that the name exists in the module and is Callable
                    tup = self._load.pop()
                    # create an event_table entry that returns control
                    # to the pipeline after running
                    self.pipeline.event_table[tup[0]] = (tup[0], "Processing", "resume_recipe")
                    self.pipeline.logger.info(f"Added {tup[0]} from {module} to event_table")
            setattr(node, 'kpf_completed', True)

    def visit_alias(self, node):
        """
        visit_alias() processes an "as" alias node, which can only appear as part of
        an "from ... import ... as ..." construct.  The "as" clause is parsed, but ignored,
        and so should not be used in recipes.
        visit_alias() puts the name and asname on the _load stack as a tuple.
        visit_importFrom() handles the heavy lifting.
        Note: asname is currently not supported and is ignored.
        """
        if self._reset_visited_states:
            setattr(node, 'kpf_completed', False)
            return
        if not getattr(node, 'kpf_completed', False):
            self._load.append((node.name, node.asname))
            self.pipeline.logger.debug(f"alias: {node.name} as {node.asname}")
            setattr(node, 'kpf_completed', True)
    
    def visit_Name(self, node):
        """
        visit_Name() processes variable names encountered in a recipe.
        A Name can occur on either the left or right side of an assignment.
        If it's on the left side the context is Store, and the Name is the
        variable name into which to store a value.  visit_Name() pushes that
        variable name on the _store stack.
        If the Name is on the right side, e.g. as part of an expression,
        visit_Name() looks up the name in the params dict, and pushes the corresponding
        value on the _load stack.  There are three special cases to be aware of.
        If the name "None" is encountered, the value None is pushed on the _load stack.
        If the name "config" is encountered, the config object is pushed on the _load
        stack.  This object supports access to attributes, so a subsequent call to
        visit_attribute(), which is provided from the "." in an expression like
        "config.ARGUMENT", will work correctly to extract the ARGUMENT dictionary
        from the config.
        The name is looked up in an environment dictionary that has been preloaded
        from the shell environment that invoked the framework and pipeline.
        If none of the previous cases produced a successful match, the name is looked up
        in the internal _params dictionary, which stores by name the values of variables
        that have been previously encountered in the recipe on the left side of assignment
        statements.  If the name is not found in any of the above places, an error is
        logged and raised.
        Note that "config" and keywords from the environment will hide variables defined
        in recipes of the same name.

        Implementer note: The same instance of a Name can appear as different nodes
        in an AST, so nothing should be stored in the node as a node-specific attribute.
        """
        if self._reset_visited_states:
            return
        self.pipeline.logger.debug(f"Name: {node.id}")
        if isinstance(node.ctx, _ast.Store):
            self.pipeline.logger.debug(f"Name is storing {node.id}")
            self._store.append(node.id)
        elif isinstance(node.ctx, _ast.Load):
            if node.id == "None":
                value = None
            elif node.id == "config":
                if self.pipeline != None and hasattr(self.pipeline, "config"):
                    value = self.pipeline.config
                else:
                    self.pipeline.logger.error(f"Name: No context or context has no config attribute")
                    raise Exception(f"Name: No context or context has no config attribute")
            elif self._env.get(node.id):
                value = self._env.get(node.id)
            else:
                try:
                    value = self._params[node.id]
                except KeyError:
                    # self.pipeline.logger.error(
                    #     f"Name {node.id} on line {node.lineno} of recipe not defined.")
                    raise RecipeError(
                        f"Name {node.id} on line {node.lineno} of recipe not defined.  Recipe environment: {self._env}.  Python environment: {os.environ}")
            self.pipeline.logger.debug(f"Name is loading {value} from {node.id}")
            self._load.append(value)
        else:
            raise RecipeError(
                f"visit_Name: on recipe line {node.lineno}, ctx is unexpected type: {type(node.ctx)}")
    
    def visit_For(self, node):
        """
        visit_For() processes the "for" node of an AST.
        
        Handling looping correctly in a recipe is made more complex because of
        the need to support calls to data processing primitives within the loop,
        which cause start_recipe() or resume_recipe() to stop walking the AST tree
        nodes and return so that the Framework can run the primitive. When
        resume_recipe() picks up walking the parse tree after the primitive returns,
        it must be able to resume with the same state as when it was interrupted.
        Furthermore, state flags and results of operations stored on the various nodes
        within the loop must be appropriately reset for each iteration, and nesting of
        loops must work correctly.

        In addition to using node attributes as described in the documentation for
        the class as a whole, visit_for() also uses the following attributes stored
        on the AST node to keep track of the state of loop processing and loop
        parameters.

        kpf_started:
            Processing has started for this "for" node and it's children, but has not
            completed unless kpf_completed is also set.

        kpf_params:
            A local dictionary used to store the values of loop variables and assignment
            targets.  The principal stored parameters are "target", "args_iter" and
            "current_arg"
        """
        if self._reset_visited_states:
            setattr(node, 'kpf_completed', False)
            setattr(node, 'kpf_started', False)
            if hasattr(node, 'kpf_params'):
                delattr(node, 'kpf_params')
            self.visit(node.target)
            self.visit(node.iter)
            for subnode in node.body:
                self.visit(subnode)
            return
            
        if not getattr(node, 'kpf_completed', False):
            if not getattr(node, 'kpf_started', False):
                params = {}
                storeQSizeBefore = len(self._store)
                self.visit(node.target)
                if self.awaiting_call_return:
                    return
                target = self._store.pop() if len(self._store) > storeQSizeBefore else None
                params['target'] = target 
                loadQSizeBefore = len(self._load)
                self.visit(node.iter)
                args = []
                if len(self._load) - loadQSizeBefore == 1:
                    item = self._load.pop()
                    if isinstance(item, Iterable):
                        self.pipeline.logger.debug(f"For: Popping first list item, {item}, of type {type(item)}")
                        args = item
                    else:
                        args.insert(0, item)
                while len(self._load) > loadQSizeBefore:
                    # pick up any additional items
                    item = self._load.pop()
                    self.pipeline.logger.debug(f"For: next list item is {item} of type {type(item)}")
                    args.insert(0, item)
                args_iter = iter(list(args))
                try:
                    current_arg = next(args_iter)
                    self.pipeline.logger.debug(f"For: first call to next returned {current_arg} of type {type(current_arg)}")
                except StopIteration:
                    current_arg = None
                params['args_iter'] = args_iter
                params['current_arg'] = current_arg
                setattr(node, 'kpf_params', params)
                setattr(node, 'kpf_started', True)
            else:
                params = getattr(node, 'kpf_params', None)
                assert(params is not None)
                target = params.get('target')
                args_iter = params.get('args_iter')
                current_arg = params.get('current_arg')
            while current_arg is not None:
                self.pipeline.logger.debug(f"For: in while loop with current_arg {current_arg}, type {type(current_arg)}")
                self._params[target] = current_arg
                for subnode in node.body:
                    self.visit(subnode)
                    if self.awaiting_call_return:
                        return
                # reset the node visited states for all nodes
                # underneath this "for" loop to set up for the
                # next iteration of the loop.
                self.pipeline.logger.debug("For: resetting visited states before looping")
                for subnode in node.body:
                    self.reset_visited_states(subnode)
                # iterate by updating current_arg (and the arg iterator)
                try:
                    current_arg = next(args_iter)
                    params['current_arg'] = current_arg
                except StopIteration:
                    break
                self.pipeline.logger.info(f"Starting For loop on recipe line {node.lineno} with arg {current_arg}")
            setattr(node, 'kpf_completed', True)

    
    def visit_Assign(self, node):
        """
        visit_Assign() processes the assignment of one or more constant or calculated
        values to named variables. The variable names to assign into come from the _store
        stack, while the values come from the _load stack.
        Calls to visit_call() may occur in the tree that is walked to generate the value to
        be assigned, and therefore may set self._awaiting_call_return, in which case visit_Assign()
        we need to immediately return, and pick up where we left off later, completing the
        assignment.  See also visit_call() and resume_recipe().
        """
        if self._reset_visited_states:
            setattr(node, 'kpf_completed', False)
            setattr(node, 'kpf_completed_targets', False)
            setattr(node, 'kpf_completed_values', False)
            if hasattr(node, 'kpf_storeQSizeBefore'):
                delattr(node, 'kpf_storeQSizeBefore')
            if hasattr(node, 'kpf_num_targets'):
                delattr(node, 'kpf_num_targets')
            for target in node.targets:
                self.visit(target)
            self.visit(node.value)
            return
        if not getattr(node, 'kpf_completed', False):
            loadQSizeBefore = len(self._load)
            storeQSizeBefore = len(self._store)
            if not getattr(node, 'kpf_completed_targets', False):
                setattr(node, 'kpf_storeQSizeBefore', storeQSizeBefore)
                for target in node.targets:
                    self.visit(target)
                    if self.awaiting_call_return:
                        return
                num_store_targets = len(self._store[storeQSizeBefore:])
                setattr(node, "kpf_num_targets", num_store_targets)
                setattr(node, 'kpf_completed_targets', True)
            else:
                num_store_targets = getattr(node, 'kpf_num_targets', 0)
            if not getattr(node, 'kpf_completed_values', False):
                self.visit(node.value)
                if self.awaiting_call_return:
                    return
                setattr(node, 'kpf_completed_values', True)
            while num_store_targets > 0 and len(self._load) > loadQSizeBefore:
                target = self._store.pop()
                self.pipeline.logger.debug(f"Assign: assignment target is {target}")
                if target == '_':
                    self._load.pop() # discard
                else:
                    self._params[target] = self._load.pop()
                    self.pipeline.logger.info(f"Assign: {target} <- {self._params[target]}, type: {self._params[target].__class__.__name__}")
                num_store_targets -= 1
            had_error = False
            while len(self._store) > storeQSizeBefore:
                had_error = True
                self.pipeline.logger.error(
                    f"Assign: unfilled target: {self._store.pop()} on line {node.lineno} of recipe.")
            while len(self._load) > loadQSizeBefore:
                had_error = True
                self.pipeline.logger.error(
                    f"Assign: unused value: {self._load.pop()} on line {node.lineno} of recipe.")
            if had_error:
                raise RecipeError(
                    f"Error during assignment on line {node.lineno} of recipe.  See log for details.")
            setattr(node, 'kpf_completed', True)

    # UnaryOp and the unary operators
    
    def visit_UnaryOp(self, node):
        """
        visit_UnaryOp() implements the UnaryOps "-x", "+x" and "not x".  The actual
        work is done in the operator visitor method, e.g. visit_UAdd or visit_USub.

        Implementor Note:
            This implementation doesn't support calls within unaryOp expressions,
            so we don't bother guarding for self.awaiting_call_return here, nor in
            the Unary Operator methods.
        """
        if self._reset_visited_states:
            self.visit(node.operand)
            self.visit(node.op)
            return
        self.pipeline.logger.debug(f"UnaryOp:")
        self.visit(node.operand)
        self.visit(node.op)

    # Unary Operators

    def _unary_op_impl(self, node, name, func):
        """ Helper function containing common implementation of unary operators. """
        if self._reset_visited_states:
            return
        self.pipeline.logger.debug(name)
        if len(self._load) == 0:
            raise RecipeError(
                f"Unary operator {name} invoked on recipe line {name.lineno} with no argument")
        self._load.append(func(self._load.pop()))

    def visit_UAdd(self, node):
        """ 
        visit_UAdd() implements the operator for unary +, as in "+x" by invoking the internal
        method _unary_op_impl() with an appropriate lambda function.
        See also visit_UnaryOp().
        """
        self._unary_op_impl(node, "UAdd", lambda x : x)

    def visit_USub(self, node):
        """ 
        visit_USub() implements the operator for unary -, as in "-x" by invoking the internal
        method _unary_op_impl() with an appropriate lambda function.
        See also visit_UnaryOp().
        """
        self._unary_op_impl(node, "USub", lambda x : -x)

    def visit_Not(self, node):
        """ 
        visit_Not() implements the operator for not, as in " not x" by invoking the internal
        method _unary_op_impl() with an appropriate lambda function.
        See also visit_UnaryOp().
        """
        self._unary_op_impl(node, "Not", lambda x : not x)

    # BinOp and the binary operators

    def visit_BinOp(self, node):
        """
        visit_BinOp() implements binary operations, i.e. "x + y", "x - y", "x * y", "x / y".
        The actual work is done in the operator visitor method, e.g. visit_Add or visit_Mult.

        Implementor Note:
            This implementation doesn't support calls within binOp expressions,
            so we don't bother guarding for self.awaiting_call_return here, nor in
            the binary operator methods.
        """
        if self._reset_visited_states:
            self.visit(node.right)
            self.visit(node.left)
            self.visit(node.op)
            return
        self.pipeline.logger.debug("BinOp:")
        # right before left because they're being pushed on a stack, so left comes off first
        self.visit(node.right)
        self.visit(node.left)
        self.visit(node.op)

    # binary operators

    def _binary_op_impl(self, node, name, func):
        """ Helper function containing common implementation of binary operators. """
        if self._reset_visited_states:
            return
        self.pipeline.logger.debug(name)
        if len(self._load) < 2:
            raise RecipeError(
                f"Binary operator {name} invoked on recipe line {node.lineno} " +
                f"with insufficient number of arguments {len(self._load)}")
        self._load.append(func(self._load.pop(), self._load.pop()))

    def visit_Add(self, node):
        """ 
        visit_Add() implements the binary addition operator by invoking the internal
        method _binary_op_impl() with an appropriate lambda function.
        See also visit_BinOp()
        """
        self._binary_op_impl(node, "Add", lambda x, y: x + y)

    def visit_Sub(self, node):
        """ 
        visit_Sub() implements the binary subtraction operator by invoking the internal
        method _binary_op_impl() with an appropriate lambda function.
        See also visit_BinOp()
        """
        self._binary_op_impl(node, "Sub", lambda x, y: x - y)
    
    def visit_Mult(self, node):
        """ 
        visit_Mult() implements the binary multiplication operator by invoking the internal
        method _binary_op_impl() with an appropriate lambda function.
        See also visit_BinOp()
        """
        self._binary_op_impl(node, "Mult", lambda x, y: x * y)
    
    def visit_Div(self, node):
        """ 
        visit_Div() implements the binary division operator by invoking the internal
        method _binary_op_impl() with an appropriate lambda function.
        See also visit_BinOp()
        """
        self._binary_op_impl(node, "Div", lambda x, y: x / y)
    
    # Compare and comparison operators

    def visit_Compare(self, node):
        """
        visit_Compare() implements comparison expressions, e.g. "x <= y".  It walks
        the AST subtrees for the left and right side of the comparison, pushing the
        corresponding values on the _load stack. It "visits" the comparison operator
        itself, which results in popping off the values for the two sides and pushing
        either "True" or "False" on the _load stack as a Bool.  See also the comparison
        operators, e.g. visit_Eq(), visit_NotEq(), visit_Lt(), visit_LtE(), visit_Is(),
        visit_IsNot(), visit_In().
        """
        if self._reset_visited_states:
            setattr(node, 'kpf_completed', False)
            for item in node.comparators:
                self.visit(item)
            self.visit(node.left)
            for op in node.ops:
                self.visit(op)
            return
        if not getattr(node, 'kpf_completed', False):
            self.pipeline.logger.debug(f"Compare")
            loadQSizeBefore = len(self._load)
            # comparators before left because they're going on a stack, so left can be pulled first
            for item in node.comparators:
                self.visit(item)
            self.visit(node.left)
            for op in node.ops:
                self.visit(op)
            setattr(node, 'kpf_completed', True)

    # Comparison operators

    def _compare_op_impl(self, node, name, func):
        """ Helper function containing common implementation of comparison operators. """
        if self._reset_visited_states:
            return
        self.pipeline.logger.debug(name)
        if len(self._load) < 2:
            raise RecipeError(
                f"Comparison operator {name} invoked on line {node.lineno} " +
                f"with less than two arguments: {len(self._load)}")
        self._load.append(func(self._load.pop(), self._load.pop()))

    def visit_Eq(self, node):
        """ 
        visit_Eq() implements the equality comparison operator by invoking
        the internal method _compare_op_impl() with an appropriate lambda function.
        See also visit_Compare().
        """
        self._compare_op_impl(node, "Eq", lambda x, y: x == y)
    
    def visit_NotEq(self, node):
        """ 
        visit_NotEq() implements the inequality comparison operator by invoking
        the internal method _compare_op_impl() with an appropriate lambda function.
        See also visit_Compare().
        """
        self._compare_op_impl(node, "NotEq", lambda x, y: x != y)
    
    def visit_Lt(self, node):
        """ 
        visit_Lt() implements the less than comparison operator by invoking
        the internal method _compare_op_impl() with an appropriate lambda function.
        See also visit_Compare().
        """
        self._compare_op_impl(node, "Lt", lambda x, y: x < y)
    
    def visit_LtE(self, node):
        """ 
        visit_LtE() implements the less than or equal comparison operator by invoking
        the internal method _compare_op_impl() with an appropriate lambda function.
        See also visit_Compare().
        """
        self._compare_op_impl(node, "LtE", lambda x, y: x <= y)
    
    def visit_Gt(self, node):
        """ 
        visit_Gt() implements the greater than comparison operator by invoking
        the internal method _compare_op_impl() with an appropriate lambda function.
        """
        self._compare_op_impl(node, "Gt", lambda x, y: x > y)
    
    def visit_GtE(self, node):
        """ 
        visit_GtE() implements the greater than or equal comparison operator by invoking
        the internal method _compare_op_impl() with an appropriate lambda function.
        See also visit_Compare().
        """
        self._compare_op_impl(node, "GtE", lambda x, y: x >= y)
    
    def visit_Is(self, node):
        """ 
        visit_Is() implements the "is" comparison operator by invoking
        the internal method _compare_op_impl() with an appropriate lambda function.
        See also visit_Compare().
        """
        self._compare_op_impl(node, "Is", lambda x, y: x is y)
    
    def visit_IsNot(self, node):
        """ 
        visit_Eq() implements the "is not" comparison operator by invoking
        the internal method _compare_op_impl() with an appropriate lambda function.
        See also visit_Compare().
        """
        self._compare_op_impl(node, "IsNot", lambda x, y: not (x is y))

    def visit_In(self, node):
        """ 
        visit_In() implements the "in" range comparison operator by invoking
        See also visit_Compare().
        the internal method _compare_op_impl() with an appropriate lambda function.
        """
        self._compare_op_impl(node, "In", lambda x, y: x in y)
    
    # TODO: implement visit_In and visit_NotIn.  Depends on support for Tuple and maybe others

    def visit_Call(self, node):
        """
        visit_Call() implements function call syntax.  The function can be one of several
        built-in functions, or can enqueue a data processing primitive to be run in the
        Framework.  In the latter case, it cooperates with return_recipe() to make available
        to the recipe values returned from data processing primitives.

        Whatever the function type, arguments are popped from the _load stack, and results
        of the call are pushed back onto the _load stack.

        If the function's name is the special case "invoke_subrecipe", the one expected
        argument is a path to a recipe file. If it has not already been, it is parsed into
        an abstract syntax tree (AST) and hung on the "call" tree node for future use.
        The subtree is then "visited" just like any other subtree.  The head of the
        subrecipe tree is a "module" node, as for all recipes.  See visit_module() for
        more details on how subrecipes are handled.

        If the name of the function to be called is not "invoke_subrecipe", the registry of
        built-in functions is checked.  (See the documentation for "register_recipe_builtins()"
        for a list of built-ins and their behavior.)

        If the function name is not found among the registered built-ins, it is presumed to be
        a data processing primitive to be run on the Keck framework.  The list of arguments
        popped from the _load stack is wrapped in an Arguments class object, an event is
        constructed containing the primitive name and the argument object.  That event is
        appended to the Framework's priority event queue. The awaiting_call_return flag is set,
        and visit_Call() immediately returns.  That flag causes an immediate return from each
        of the visit_<syntax_element>() functions all the way up the call stack, and the
        instance of start_recipe() or resume_recipe() at the top of the call stack also returns.
        The Framework is then free to run the next queued event, which is the primitive named
        in node being processed by this visit_Call().

        When the primitive has been run by the framework, the next primitive will be
        "resume_recipe", which will set the returning_from_call flag and start traversing the
        previously saved AST tree from the top again.  Because of the various kpf_completed
        attributes set on nodes of the tree, processing will quickly get back to here, with the
        state of the various tree nodes the same as before.  resume_recipe() will have placed
        the output of the primitive, which is an Arguments class object, on the class instance's
        call_output property and set the returning_from_call flag.  visit_Call() will extract each
        positional argument value from the Arguments object and push it onto the _load stack,
        becoming the results of the call.  (Any keyword arguments in the Arguments object returned
        from the primitive are ignored.)
        """
        if self._reset_visited_states:
            setattr(node, 'kpf_completed', False)
            for arg in node.args:
                self.visit(arg)
            return
        self.pipeline.logger.debug(f"Call: {node.func.id} on recipe line {node.lineno}; kpf_completed is {getattr(node, 'kpf_completed', False)}")
        if node.func.id == 'invoke_subrecipe':
            subrecipe = getattr(node, '_kpf_subrecipe', None)
            if not subrecipe:
                self.pipeline.logger.debug(f"invoke_subrecipe: opening and parsing recipe file {node.args[0].s}")
                # TODO: do some argument checking here
                with open(node.args[0].s) as f:
                    fstr = f.read()
                    subrecipe = parse(fstr)
                node._kpf_subrecipe = subrecipe
            else:
                self.pipeline.logger.debug(f"invoke_subrecipe: found existing subrecipe of type {type(subrecipe)}")
            saved_depth = self.subrecipe_depth
            self.subrecipe_depth = self.subrecipe_depth + 1
            self.visit(subrecipe)
            self.subrecipe_depth = saved_depth
            if self.awaiting_call_return:
                return
        elif not getattr(node, 'kpf_completed', False):
            if not self.returning_from_call:
                # Build and queue up the called function and arguments
                # as a pipeline event.
                # The "next_event" item in the event_table, populated
                # by visit_ImportFrom, will ensure that the recipe
                # processing will continue by making resume_recipe
                # the next scheduled event primative.
                # add keyword arguments
                kwargs = {}
                for kwnode in node.keywords:
                    self.visit(kwnode)
                    tup = self._load.pop()
                    kwargs[tup[0]] = tup[1]
                if node.func.id in self._builtins.keys():
                    # directly handle one of the registered built-in functions and push
                    # the results on the _load stack
                    func, nargs = self._builtins[node.func.id]
                    if len(node.args) != nargs:
                        self.pipeline.logger.error(f"Call to {node.func.id} takes exactly {nargs} args, got {len(node.args)} on recipe line {node.lineno}")
                        raise RecipeError(f"Call to {node.func.id} takes exactly one arg, got {len(node.args)} on recipe line {node.lineno}")
                    arglist = []
                    for ix in range(nargs-1, -1, -1): # down through range because _load is a LIFO stack
                        self.visit(node.args[ix])
                        arglist.append(self._load.pop())
                    results = func(*arglist, **kwargs)
                    if isinstance(results, tuple):
                        self.pipeline.logger.debug(f"Call (builtin): returned tuple, unpacking")
                        for item in results:
                            self.pipeline.logger.debug(f"Call (builtin): appending {item} of type {type(item)} to _load")
                            self._load.append(item)
                    else:
                        self.pipeline.logger.debug(f"Call (builtin): appending {results} of type {type(results)} to _load")
                        self._load.append(results)
                else:
                    # Prepare arguments for and enqueue a data processing primitive to be executed
                    # by the Framework, set the self.awaiting_call_return flag, and return.
                    event_args = Arguments(name=node.func.id+"_args", **kwargs)
                    # add positional arguments
                    for argnode in node.args:
                        self.visit(argnode)
                        event_args.append(self._load.pop())
                    self.context.append_event(node.func.id, event_args)
                    self.pipeline.logger.info(f"Queued {node.func.id} with args {str(event_args)}; awaiting return.")
                    #
                    self.awaiting_call_return = True
                    return
            else:
                # returning from a call (pipeline event):
                # Get any returned values, stored by resume_recipe() in self.call_output,
                # and push them on the _load stack for Assign (or whatever) to handle.
                self.pipeline.logger.debug(f"Call on recipe line {node.lineno} returned output {self.call_output}")
                if isinstance(self.call_output, Arguments):
                    # got output that we can deal with, otherwise, ignore the returned value
                    for ix in range(len(self.call_output)):
                        self._load.append(self.call_output[ix])
                self.call_output = None
                self.returning_from_call = False
            setattr(node, 'kpf_completed', True)

    def visit_keyword(self, node):
        """
        visit_keyword() implements the syntax of keyword arguments (as opposed to positional
        arguments) within function calls.  It does so by generating tuples of (keyword, value),
        generating the value by walking the corresponding AST subtree.  The resulting tuple is
        stored on the _load stack, replacing the keyword name item.
        """
        if self._reset_visited_states:
            return
        # let the value node put the value on the _load stack
        self.visit(node.value)
        val = self._load.pop()
        self.pipeline.logger.debug(f"keyword: {val}")
        self._load.append((node.arg, val))

    def visit_If(self, node):
        """
        visit_If() implements conditional execution triggered by an "if" or "if ... else"
        statement.
        Evaluate the test and visit one of the two branches, body or orelse.
        
        Note:
            The python "if" expression, e.g. x = a if <condition> else b, is not
            supported in recipes.
        """
        if self._reset_visited_states:
            setattr(node, 'kpf_completed', False)
            setattr(node, 'kpf_completed_test', False)
            if hasattr(node, 'kpf_boolResult'):
                delattr(node, 'kpf_boolResult')
            self.visit(node.test)
            for item in node.body:
                self.visit(item)
            for item in node.orelse:
                self.visit(item)
            return
        if not getattr(node, 'kpf_completed', False):
            if not getattr(node, 'kpf_completed_test', False):
                loadQSizeBefore = len(self._load)
                self.visit(node.test)
                if len(self._load) <= loadQSizeBefore:
                    raise RecipeError(
                        f"visit_If: on recipe line {node.lineno}, test didn't push a result on the _load stack")
                boolResult = self._load.pop()
                self.pipeline.logger.info(f"If condition on recipe line {node.lineno} was {boolResult}")
                setattr(node, 'kpf_boolResult', boolResult)
                setattr(node, 'kpf_completed_test', True)
            else:
                boolResult = getattr(node, 'kpf_boolResult')
            if boolResult:
                self.pipeline.logger.debug(
                    f"If on recipe line {node.lineno} pushing and visiting Ifso")
                for item in node.body:
                    self.visit(item)
                    if self.awaiting_call_return:
                        return
            else:
                self.pipeline.logger.debug(
                    f"If on recipe line {node.lineno} pushing and visiting Else")
                for item in node.orelse:
                    self.visit(item)
                    if self.awaiting_call_return:
                        return
            setattr(node, 'kpf_completed', True)

    def visit_List(self, node):
        """
        visit_List() implements the "[<list item>, <list item>, ...]" list syntax from
        Python.  It does this by visiting the tree node representing each list element in 
        turn.  Visiting each node results in a new item pushed onto the _load stack.  After
        visiting each subtree, the new items on the _load stack are popped and appended onto
        a list object.  Finally, the list object is pushed onto the _load stack.
        """
        if self._reset_visited_states:
            setattr(node, 'kpf_completed', False)
            for elt in node.elts:
                self.visit(elt)
            return
        self.pipeline.logger.debug(f"List")
        if not getattr(node, "kpf_completed", False):
            l = []
            loadDepth = len(self._load)
            for elt in node.elts:
                self.visit(elt)
                if len(self._load) > loadDepth:
                    l.append(self._load.pop())
            self._load.append(l)
            setattr(node, "kpf_completed", True)
    
    def visit_Tuple(self, node):
        """
        visit_Tuple() implements the "(<tuple item>, <tuple item>, ...)" tuple syntax from
        Python.  It does this by visiting the tree node representing each tuple element in 
        turn.  Visiting each node results in a new item pushed onto the _load stack.  After
        visiting each subtree, the new items on the _load stack are popped and appended onto
        a list object.  Finally, the list object is pushed onto the _load stack.

        Internally, the tuple is generated by calling visit_List() to build a list on the
        _load stack, and then converting it into a tuple.
        """
        self.pipeline.logger.debug(f"Tuple")
        self.visit_List(node)
        if self._reset_visited_states:
            return
        if not getattr(node, "kpf_completed", False):
            if not isinstance(self._load[len(self._load)-1], list):
                raise RecipeError("visit_Tuple() expected a list on the _load stack, "
                    f"but got {self._load[len(self._load)-s]}")
            self._load.append(tuple(self._load.pop()))
            setattr(node, "kpf_completed", True)
        
    def visit_NameConstant(self, node):
        """
        visit_NameConstant() implements python NameConstant syntax element by pushing the value
        on the _load stack.
        """
        if self._reset_visited_states:
            return
        self.pipeline.logger.debug(f"NameConstant: {node.value}")
        #ctx of NameConstant is always Load
        self._load.append(node.value)
    
    def visit_Num(self, node):
        """
        visit_Num() implements a numeric constant by pushing it on the _load stack.
        """
        if self._reset_visited_states:
            return
        self.pipeline.logger.debug(f"Num: {node.n}")
        # ctx of Num is always Load
        self._load.append(node.n)

    def visit_Str(self, node):
        """
        visit_Str() implements a string constant by pushing its value on the _load stack.
        Multiline strings delimited by three double-quotes will result in string values
        with embedded line breaks.  Multiple quoted strings with no separator character
        e.g. comma, are not automatically concatenated, as would be in Python.
        """
        if self._reset_visited_states:
            return
        self.pipeline.logger.debug(f"Str: {node.s}")
        # ctx of Str is always Load
        self._load.append(node.s)
    
    def visit_Expr(self, node):
        """
        visit_Expr() implements an expression by pushing the resulting value on the _load
        stack.  Expression syntax elements are produced by the AST parser only rarely in recipe
        situations, for example when an expression stands alone, not as part of an assignment
        statement.
        """
        if self._reset_visited_states:
            setattr(node, 'kpf_completed', False)
            self.visit(node.value)
            return
        if not getattr(node, 'kpf_completed', False):
            self.visit(node.value)
            if self.awaiting_call_return:
                return
            setattr(node, 'kpf_complted', True)
    
    def visit_Attribute(self, node):
        """
        visit_Attribute() implements the syntax e.g. of an object attribute access, e.g. "a.key".
        It does so by getting the name of the attribute, and then testing to see of the object at
        the top of the _load stack has an attribute of that name.  If it does, the value of the
        attribute replaces the original object at the top of the _load stack.  If no such
        attribute exists on the object, an message is logged and RecipeError is raised.
        """
        if self._reset_visited_states:
            return
        self.visit(node.value)
        obj = self._load.pop()
        if isinstance(node.ctx, _ast.Load):
            try:
                value = obj.getValue(node.attr)
                # print(f"Attribute: value is {type(value)}: {value}")
            except (KeyError, AttributeError):
                self.pipeline.logger.error(
                    f"Object {obj} on line {node.lineno} of recipe has no attribute {node.attr}.")
                raise RecipeError(
                    f"Object {obj} on line {node.lineno} of recipe has no attribute {node.attr}.")
            self.pipeline.logger.debug(f"Name is loading {value} from {node.attr}")
            self._load.append(value)
        elif isinstance(node.ctx, _ast.Store):
            self.pipeline.logger.error(
                f"Assigning to dictionary attribute on line {node.lineno} not supported")
            raise RecipeError(
                f"Assigning to dictionary attribute on line {node.lineno} not supported")
    
    def visit_Subscript(self, node):
        """
        visit_Subscript() implements subscript syntax of the form a[i], where the subscript value
        is a single index.  The behavior is to replace the object on the _load stack with the value
        of the indexed item.  See also visit_Index().

        Note:
            Slice subscripts of the form a[i:j] are not supported.  
        """
        if self._reset_visited_states:
            return
        if isinstance(node.ctx, _ast.Load):
            self.visit(node.value)
            value = self._load.pop()
            self.visit(node.slice)
            sliceName = self._load.pop()
            self._load.append(value[sliceName])
        elif isinstance(node.ctx, _astStore):
            self.pipeline.logger.error(
                f"Assigning to subscript {node.sliceName} on recipe line {node.lineno} not supported")
            raise RecipeError(
                f"Assigning to subscript {node.sliceName} on recipe line {node.lineno} not supported")
    
    def visit_Index(self, node):
        """
        visit_Index() doesn't do anything special.  It simply visits the subtree corresponding to the
        index value.  That (or those) nodes will have the result of pushing some appropriate value onto
        the _load stack.  visit_Index() doesn't need to alter that value in any way.  Typically
        visit_Subscript() will use that value to perform the actual indexing operation.
        """
        if self._reset_visited_states:
            return
        self.visit(node.value)

    def generic_visit(self, node):
        """
        generic_visit() is called if no explicit visitor function exists for a node.
        It logs a message in the pipeline logger noting that the recipe contained an
        unsupported syntax element, and then raises RecipeError with the same message.
        """
        self.pipeline.logger.error(
            f"generic_visit: got unsupported node {node.__class__.__name__}")
        raise RecipeError(
            f"Unsupported language feature: {node.__class__.__name__}")
    
    def reset_visited_states(self, node):
        """
        reset_visited_states() walks the tree below the node where the call is made after setting
        the class member self._reset_visited_states.  Each visit_<syntax_element>, when called with
        that flag set, should reset its state so that an enclosing loop can be properly executed again
        with new parameters.

        reset_visited_states() is called in visit_<> methods that control looping, such as visit_For()
        """
        self._reset_visited_states = True
        self.awaiting_call_return = False
        self.returning_from_call = False
        self.call_output = None
        self.visit(node)
        self._load.clear()
        self._store.clear()
        self._reset_visited_states = False