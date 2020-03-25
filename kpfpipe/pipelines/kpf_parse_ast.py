# kpf_parse_ast_1.py

from ast import NodeVisitor, iter_fields
import _ast
from collections.abc import Iterable
from collections import deque
from queue import Queue
from kpfpipe.pipelines.FauxLevel0Primitives import read_data, Normalize, NoiseReduce, Spectrum1D
from kpfpipe.models.kpf_arguments import KpfArguments
from keckdrpframework.models.action import Action
from keckdrpframework.models.processing_context import ProcessingContext

class KpfPipelineNodeVisitor(NodeVisitor):
    """
    Node visitor to convert KPF pipeline recipes expressed in python syntax
    into operations on the KPF Framework.

    Prototype version!
    """
    _indentStr = ""
    _indent = 0

    def __init__(self, pipeline=None, context=None):
        NodeVisitor.__init__(self)
        self._indentStr = "  "
        self._indent = 0
        # instantiate the parameters dict
        self._params = None
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
    
    def visit_Module(self, node):
        """
        Module node
        A Module node is always at the top of an AST tree returned
        by ast.parse(), and there is only one, so we initialize
        or reset things here, and clean up (releasing allocated
        memory) at the end.
        """
        if self._reset_visited_states:
            setattr(node, 'kpf_started', False)
            setattr(node, 'kpf_completed', False)
            for item in node.body:
                self.visit(item)
            self._params = None # let storage get collected
            return
        print("Module")
        if not getattr(node, 'kpf_started', False):
            self._params = {}
            setattr(node, 'kpf_started', True)
        if not getattr(node, 'kpf_completed', False):
            self._indent += 1
            for item in node.body:
                self.visit(item)
                if self.awaiting_call_return:
                    return
            self._params = None # let allocated memory get collected
            setattr(node, 'kpf_completed', True)

    def visit_ImportFrom(self, node):
        """
        import primitives and add them to the pipeline's event_table
        """
        if self._reset_visited_states:
            setattr(node, 'kpf_completed', False)
            for name in node.names:
                self.visit(name)
            return
        if not getattr(node, 'kpf_completed', False):
            module = node.module
            #TODO do something with module to extract a path
            loadQSizeBefore = len(self._load)
            for name in node.names:
                self.visit(name)
            while len(self._load) > loadQSizeBefore:
                # import the named primitive
                # This comes as a 2-element tuple from visit_alias
                #
                # just add the name to the event_table for now
                # But we should ensure that the name exists in the module and is Callable
                tup = self._load.pop()
                print(f"Import {tup[0]} from {module}")
                # create an event_table entry that returns control
                # to the pipeline after running
                self.pipeline.event_table[tup[0]] = (tup[0], "Processing", "resume_recipe")
            setattr(node, 'kpf_completed', True)

    def visit_alias(self, node):
        """
        alias node
        These only appear in import clauses
        Implement by putting name and asname on _load stack as tuple.
        ImportFrom handles the heavy lifting
        Note: asname is currently not supported and is ignored.
        """
        if self._reset_visited_states:
            setattr(node, 'kpf_completed', False)
            return
        if not getattr(node, 'kpf_completed', False):
            self._load.append((node.name, node.asname))
            print(f"{self._indentStr * self._indent}alias: {node.name} as {node.asname}")
            setattr(node, 'kpf_completed', True)
    
    def visit_Name(self, node):
        """
        Implementation of Name node
        A Name can occur on either the left or right side of an assignment.
        If it's on the left side the context is Store, and the Name is the
        variable name into which to store a value.  We push that variable name
        on the _store stack.
        If the Name is on the right side, e.g. as part of an expression,
        we look up the name in our params dict, and push the corresponding
        value on the _load stack.  If the name is not found, None is pushed
        on the _load stack.

        WARNING: The same instance of a Name can appear as different nodes in a AST,
        so nothing should be stored in the node as a node-specific attribute.
        """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}Name: {node.id}")
        if isinstance(node.ctx, _ast.Store):
            # print(f"Name is storing {node.id}")
            self._store.append(node.id)
        elif isinstance(node.ctx, _ast.Load):
            value = self._params.get(node.id)
            print(f"Name is loading {value} from {node.id}")
            self._load.append(value)
        else:
            raise Exception("visit_Name: ctx is unexpected type: {}".format(type(node.ctx)))
    
    def visit_For(self, node):
        """
        Implement the For node

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
        print("{}For: {} in ".format(self._indentStr * self._indent, node.target.id))
        if not getattr(node, 'kpf_completed', False):
            if not getattr(node, 'kpf_started', False):
                params = {}
                storeQSizeBefore = len(self._store)
                self.visit(node.target)
                if self.awaiting_call_return:
                    return
                if len(self._store) > storeQSizeBefore:
                    target = self._store.pop()
                    params['target'] = target             
                loadQSizeBefore = len(self._load)
                self.visit(node.iter)
                args = deque()
                while len(self._load) > loadQSizeBefore:
                    args.appendleft(self._load.pop())
                # TODO: can this be made simpler?
                args_iter = iter(list(args))
                current_arg = next(args_iter)
                params['args_iter'] = args_iter
                params['current_arg'] = current_arg
                setattr(node, 'kpf_params', params)
                setattr(node, 'kpf_started', True)
                print(f"\n\n\nFor: top of loop with arg {current_arg}")
            else:
                params = getattr(node, 'kpf_params', None)
                assert(params is not None)
                target = params.get('target')
                args_iter = params.get('args_iter')
                current_arg = params.get('current_arg')
            # TODO: how to loop?
            # (this doesn't seem to work right)
            while True:
                self._params[target] = current_arg
                for subnode in node.body:
                    self.visit(subnode)
                    if self.awaiting_call_return:
                        return
                # reset the node visited states for all nodes
                # underneath this "for" loop to set up for the
                # next iteration of the loop.
                for subnode in node.body:
                    self.reset_visited_states(subnode)
                # iterate by updating current_arg (and the arg iterator)
                try:
                    current_arg = next(args_iter)
                    params['current_arg'] = current_arg
                except StopIteration:
                    break
                print(f"\n\n\nFor: top of loop with arg {current_arg}")
            setattr(node, 'kpf_completed', True)

    
    def visit_Assign(self, node):
        """
        Assign one or more constant or calculated values to named variables
        The variable names come from the _store stack, while the values
        come from the _load stack.
        Calls to visit() may set self._awaiting_call_return, in which case
        we need to immediately return, and pick up where we left off later,
        completing the assignment.  This typically happens when there is
        a call to a processing primitive, which is queued up on the
        framework's event queue.  See also resume_recipe().
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
            print(f"{self._indentStr * self._indent}Assign:")
            self._indent += 1
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
                print(f"{self._indentStr * self._indent}Assign from:")
                self.visit(node.value)
                if self.awaiting_call_return:
                    return
                setattr(node, 'kpf_completed_values', True)
            while num_store_targets > 0 and len(self._load) > loadQSizeBefore:
                target = self._store.pop()
                self._params[target] = self._load.pop()
                num_store_targets -= 1
                print(f"{self._indentStr * self._indent}Assign: {target} <- {self._params[target]}")
            while len(self._store) > storeQSizeBefore:
                print(f"{self._indentStr * self._indent}Assign: unfilled target: {self._store.pop()}")
            while len(self._load) > loadQSizeBefore:
                print(f"{self._indentStr * self._indent}Assign: unused value: {self._load.pop()}")
            self._indent -= 1
            setattr(node, 'kpf_completed', True)

    # UnaryOp and the unary operators
    
    def visit_UnaryOp(self, node):
        """
        implement UnaryOp
        We don't support calls in unaryOp expressions, so we don't
        bother guarding for self.awaiting_call_return here, nor in
        the following Unary Operators
        """
        if self._reset_visited_states:
            self.visit(node.operand)
            self.visit(node.op)
            return
        print(f"{self._indentStr * self._indent}UnaryOp:")
        self.visit(node.operand)
        self.visit(node.op)

    # Unary Operators

    def visit_UAdd(self, node):
        """ implement UAdd """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}USub")
        if len(self._load) == 0:
            raise Exception("visit_UnaryOp: called with no argument")
        pass
        # it would be silly to do this:
        # self._load.append(+self._load.pop())

    def visit_USub(self, node):
        """ implement USub """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}USub")
        if len(self._load) == 0:
            raise Exception("visit_UnaryOp: called with no argument")
        self._load.append(-self._load.pop())

    def visit_UNot(self, node):
        """ implement USub """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}USub")
        if len(self._load) == 0:
            raise Exception("visit_UnaryOp: called with no argument")
        self._load.append(not self._load.pop())

    # BinOp and the binary operators

    def visit_BinOp(self, node):
        """
        BinOp
        """
        if self._reset_visited_states:
            self.visit(node.right)
            self.visit(node.left)
            self.visit(node.op)
            return
        print("{}BinOp:".format(self._indentStr * self._indent))
        self._indent += 1
        # right before left because they're being pushed on a stack, so left comes off first
        self.visit(node.right)
        self.visit(node.left)
        self.visit(node.op)
        self._indent -= 1

    # binary operators

    def visit_Add(self, node):
        """ implement the addition operator """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}Add")
        if len(self._load) < 2:
            raise Exception(f"Add called with insufficient number of arguments: {len(self._load)}")
        self._load.append(self._load.pop() + self._load.pop())

    def visit_Sub(self, node):
        """ implement the subtraction operator """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}Sub")
        if len(self._load) < 2:
            raise Exception(f"Sub called with insufficient number of arguments: {len(self._load)}")
        self._load.append(self._load.pop() - self._load.pop())
    
    def visit_Mult(self, node):
        """ implement the multiplication operator """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}Mult")
        if len(self._load) < 2:
            raise Exception(f"Mult called with insufficient number of arguments: {self._load.qsize()}")
        self._load.append(self._load.pop() * self._load.pop())
    
    def visit_Div(self, node):
        """ implement the division operator """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}Div")
        if len(self._load) < 2:
            raise Exception(f"Div called with insufficient number of arguments: {len(self._load)}")
        self._load.append(self._load.pop() / self._load.pop())
    
    # Comparison operators

    def visit_Eq(self, node):
        """ implement Eq comparison operator """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}Eq")
        if len(self._load) < 2:
            raise Exception(f"Eq called with less than two arguments: {self._load.qsize()}")
        self._load.append(self._load.pop() == self._load.pop())
    
    def visit_NotEq(self, node):
        """ implement NotEq comparison operator """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}NotEq")
        if len(self._load) < 2:
            raise Exception(f"NotEq called with less than two arguments: {len(self._load)}")
        self._load.append(self._load.pop() != self._load.pop())
    
    def visit_Lt(self, node):
        """ implement Lt comparison operator """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}Lt")
        if len(self._load) < 2:
            raise Exception(f"Lt called with less than two arguments: {len(self._load)}")
        self._load.append(self._load.pop() < self._load.pop())
    
    def visit_LtE(self, node):
        """ implement LtE comparison operator """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}LtE")
        if len(self._load) < 2:
            raise Exception(f"LtE called with less than two arguments: {len(self._load)}")
        self._load.append(self._load.pop() <= self._load.pop())
    
    def visit_Gt(self, node):
        """ implement Gt comparison operator """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}Gt")
        if len(self._load) < 2:
            raise Exception(f"Gt called with less than two arguments: {self._load.qsize()}")
        self._load.append(self._load.pop() > self._load.pop())
    
    def visit_GtE(self, node):
        """ implement GtE comparison operator """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}GtE")
        if len(self._load) < 2:
            raise Exception(f"GtE called with less than two arguments: {len(self._load)}")
        self._load.append(self._load.pop() >= self._load.pop())
    
    def visit_Is(self, node):
        """ implement Lt comparison operator """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}Is")
        if len(self._load) < 2:
            raise Exception(f"Is called with less than two arguments: {len(self._load)}")
        self._load.append(self._load.pop() is self._load.pop())
    
    def visit_IsNot(self, node):
        """ implement Lt comparison operator """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}IsNot")
        if len(self._load) < 2:
            raise Exception(f"IsNot called with less than two arguments: {len(self._load)}")
        self._load.append(not (self._load.pop() is self._load.pop()))
    
    # TODO: implement visit_In and visit_NotIn.  Depends on support for Tuple and maybe others

    def visit_Call(self, node):
        """
        Implement function call
        The arguments are pulled from the _load stack into a deque.
        Targets are put on the _store stack.
        After the call has been pushed to the framework's event queue,
        we set our awaiting_call_return flag and return.  That flag
        causes immediate returns all the way up the call stack.
        When the primitive has been run by the framework, the next
        primitive will be our "resume_recipe", which will set the
        returning_from_call flag and start traversing the AST tree
        from the top again.  Because of the various kpf_completed
        attributes set on nodes of the tree, processing will quickly
        get back to here, where the output of the primitive will be
        pushed on the _load stack, becoming the result of the call.

        TODO: It is awkward to support both positional and keyword
        arguments to the the called function, because the Arguments
        class only supports keyword arguments.  We currently handle
        positional arguments by naming them arg0, arg1, etc.  The
        names should probably be made less likely to clash with
        real keyword-based arguments.

        Even more difficult is the situation with returned values
        from the primitives.  The python syntax around a function
        call assumes positional return values, but the Arguments
        class only supports named values.

        We need to figure out a better way to deal with arguments
        and return values.
        """
        if self._reset_visited_states:
            setattr(node, 'kpf_completed', False)
            for arg in node.args:
                self.visit(arg)
            return
        if not getattr(node, 'kpf_completed', False):
            print(f"{self._indentStr * self._indent}Call: {node.func.id}")
            if not self.returning_from_call:
                loadSizeBefore = len(self._load)
                for arg in node.args:
                    self.visit(arg)
                """
                func_args = {}
                func_args['name'] = node.func.id+"_args"
                arg_num = 0
                while len(self._load) > loadSizeBefore:
                    foo = self._load.pop()
                    print(f"Call: processing arg {foo}")
                    func_args['arg'+str(arg_num)] = foo
                    arg_num += 1
                print(f"Call: func_args: {func_args}")
                args = Arguments(**func_args)
                """
                args = KpfArguments(name=node.func.id+"_args")
                while len(self._load) > loadSizeBefore:
                    # because the arguments come off the _load stack in
                    # reverse order, we need to insert each one at the front
                    args.insert(0, self._load.pop())
                # Build and queue up the called function and arguments
                # as a pipeline event.
                # The "next_event" item in the event_table, populated
                # by visit_ImportFrom, will ensure that the recipe
                # processing will continue by making resume_recipe
                # the next scheduled event primative.
                self.context.push_event(node.func.id, args)
                #
                self.awaiting_call_return = True
                return
            else:
                self.returning_from_call = False
                # print(f"Call: returning, output is {self.call_output}")

                """
                # This is truly ugly, since it requires primitive
                # implementors to name their return values names
                # that we recognize here so that we can pull them
                # out in a well-defined order to put them on the
                # _load stack.
                # Alternatives I've thought of:
                # peek at the _store stack to see the names that
                # return values will be assigned into.  But the
                # name used inside the primitive has no relation
                # to the variable name in the recipe.
                # There needs to be a way to handle positional
                # return values!
                arg_num = 0
                while True:
                    argname = "arg"+str(arg_num)
                    if not hasattr(self.call_output, argname):
                        break
                    self._load.append(getattr(self.call_output, argname))
                    arg_num += 1
                """
                assert(isinstance, self.call_output, KpfArguments)
                for ix in range(len(self.call_output)):
                    self._load.append(self.call_output[ix])
                self.call_output = None
            setattr(node, 'kpf_completed', True)

    
    def visit_Compare(self, node):
        """
        Implement Compare as follows:
        visiting "left" and "comparators" puts values on the _load stack.
        visiting "ops" evaluates some comparison operator, and puts the result
        on the _load stack as a Bool.
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
            print(f"{self._indentStr * self._indent}Compare")
            loadQSizeBefore = len(self._load)
            # comparators before left because they're going on a stack, so left can be pulled first
            for item in node.comparators:
                self.visit(item)
            self.visit(node.left)
            for op in node.ops:
                self.visit(op)
            print("{}Compare changed load qsize by {}".format(
                self._indentStr * self._indent,
                len(self._load)-loadQSizeBefore))
            setattr(node, 'kpf_completed', True)

    def visit_If(self, node):
        """
        Implementation of If
        Evaluate the test and visit one of the two branches, body or orelse.
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
            print(f"{self._indentStr * self._indent}If")
            self._indent += 1
            if not getattr(node, 'kpf_completed_test', False):
                print(f"{self._indentStr * self._indent}test: ")
                loadQSizeBefore = len(self._load)
                self.visit(node.test)
                if len(self._load) <= loadQSizeBefore:
                    raise Exception("visit_If: test didn't push a result on the _load stack")
                boolResult = self._load.pop()
                setattr(node, 'kpf_boolResult', boolResult)
                setattr(node, 'kpf_completed_test', True)
            else:
                boolResult = getattr(node, 'kpf_boolResult')
            if boolResult:
                print(f"{self._indentStr * self._indent}pushing and visiting Ifso")
                for item in node.body:
                    self.visit(item)
                    if self.awaiting_call_return:
                        return
            else:
                print(f"{self._indentStr * self._indent}pushing and visiting Else")
                for item in node.orelse:
                    self.visit(item)
                    if self.awaiting_call_return:
                        return
            self._indent -= 1
            setattr(node, 'kpf_completed', True)

    def visit_List(self, node):
        """
        List node
        """
        if self._reset_visited_states:
            setattr(node, 'kpf_completed', False)
            for elt in node.elts:
                self.visit(elt)
            return
        print(f"{self._indentStr * self._indent}List")
        if not getattr(node, "kpf_completed", False):
            for elt in node.elts:
                self.visit(elt)
            setattr(node, "kpf_completed", True)
    
    def visit_Tuple(self, node):
        """
        Tuple node
        """
        if self._reset_visited_states:
            setattr(node, 'kpf_completed', False)
            return
        print(f"{self._indentStr * self._indent}Tuple")
        if not getattr(node, "kpf_completed", False):
            for elt in node.elts:
                self.visit(elt)
            setattr(node, "kpf_completed", True)
        
    def visit_NameConstant(self, node):
        """
        NameConstant
        implement name constant by putting on the _load stack
        """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}NameConstant: {node.value}")
        #ctx of NameConstant is always Load
        self._load.append(node.value)
    
    def visit_Num(self, node):
        """
        Num
        implement numeric constant by putting it on the _load stack

        NB: An instance of Num can appear as different nodes in the same AST,
        so we can't store node specific information as an attribute.
        """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}Num: {node.n}")
        # ctx of Num is always Load
        self._load.append(node.n)

    def visit_Str(self, node):
        """
        Str node
        TODO: I'm not sure what to do with a multiline comment expressed as Str
        """
        if self._reset_visited_states:
            return
        print(f"{self._indentStr * self._indent}Str: {node.s}")
        # ctx of Str is always Load
        self._load.append(node.s)
    
    def visit_Expr(self, node):
        """
        Expr node
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

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        print("generic_visit: got {}".format(type(node)))
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, _ast.AST):
                        self.visit(item)
            elif isinstance(value, _ast.AST):
                self.visit(value)
    
    def reset_visited_states(self, node):
        """
        Resets kpf_completed and other attributes of this and all subnodes,
        e.g. so that a for loop can iterate with a fresh start.
        """
        self._reset_visited_states = True
        self.awaiting_call_return = False
        self.returning_from_call = False
        self.call_output = None
        self.visit(node)
        self._load.clear()
        self._store.clear()
        self._reset_visited_states = False