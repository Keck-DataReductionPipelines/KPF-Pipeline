# kpf_parse_ast.py

from ast import NodeVisitor, iter_fields
import _ast
from queue import LifoQueue, Queue
from _queue import Empty
from collections.abc import Iterable
from collections import deque
from FauxLevel0Primatives import read_data, Normalize, NoiseReduce, Spectrum1D

class KpfPipelineNodeVisitor(NodeVisitor):
    """
    Node visitor to convert KPF pipeline recipes expressed in python syntax
    into operations on the KPF Framework.

    Prototype version!
    """

    class PipelineState():
        """
        PipelineState is a place to hold an AST and keep track of where
        we are in the process of traversing it. It supports losing
        control because of a Call and regaining it by being called again.

        Unlike some uses of tree structures, we don't want to traverse
        every node in the tree.  Rather, we want to honor conditional
        tests and only visit those nodes that tests call for.

        recurse(node) should be called by visitor functions that begin
        traversing a subtree. Call it with the node or list that heads
        the subtree.
        """

        def __init__(self, parent=None):
            self._parent = parent
            self._tree = None
            self._stack = LifoQueue()
            self._top = None
            self._params = {}
        
        def set_tree(self, tree):
            self._tree = tree        

        def __next__(self):
            while True:
                if self._top is None:
                    raise StopIteration
                if isinstance(self._top, _ast.AST):
                    # print("next(_state): got AST {}".format(type(self._top)))
                    return (self._top, self._params)
                elif isinstance(self._top, list):
                    # convert list to list_iterator
                    self._top = iter(self._top)
                if isinstance(self._top, Iterable):
                    # print("next(_state): got Iterable {}".format(type(self._top)))
                    try:
                        # print("next(_state): trying next()")
                        return (next(self._top), self._params)
                    except StopIteration:
                        # print("next(_state): caught StopIteration")
                        pass
                    try:
                        # print("next(_state): trying get_nowait()")
                        if self._stack.empty():
                            self._top = None
                            self._params = {}
                        else:
                            self._top, self._params = self._stack.get_nowait()
                            # print("next(_state): got {}, looping".format(type(self._top)))
                        if self._parent is not None:
                            self._parent._indent -= 1
                        continue
                    except Empty:
                        pass
                    raise StopIteration
                else:
                    raise Exception("Unsupported element: {}".format(type(self._top)))

        def recurse(self, node, params={}):
            """
            Any node visitor function that explicitly recurses down a new tree
            branch should call recurse with the head of that branch so that
            interruptions in flow can be resumed.
            """
            print("recurse called with node {}, params {}".format(node, params))
            if not isinstance(node, (_ast.AST, Iterable)):
                raise Exception("recurse: arg type not supported: {}".format(type(node)))
            self._stack.put((self._top, self._params))
            self._top = node
            self._params = params
            if self._parent is not None:
                self._parent._indent += 1
            else:
                print("recurse: parent is None")

        # end of PipelineState subclass

    _indentStr = ""
    _indent = 0

    def __init__(self):
        NodeVisitor.__init__(self)
        self._indentStr = "  "
        self._indent = 0
        self._state = self.PipelineState(self)
        # instantiate the primative dict and some primitives
        self._level0_prims = {}
        self._level0_prims["read_data"] = read_data
        self._level0_prims["Normalize"] = Normalize
        self._level0_prims["NoiseReduce"] = NoiseReduce
        self._level0_prims["Spectrum1D"] = Spectrum1D
        # instantiate the parameters dict
        self._params = {}
        # store and load stacks
        self._store = LifoQueue()
        self._load = LifoQueue()
    
    def visit_Module(self, node):
        print("Module")
        self._indent += 1
        self._state.set_tree(node)
        self._state._top = None
        self._state._params = {}
        for item in node.body:
            self.visit(item)

    def visit_ImportFrom(self, node):
        names = node.names[0].name
        for name in node.names[1:]:
            names += ", " + name.name
        print("Import {} from {}".format(names, node.module))
        self.generic_visit(node)
    
    def visit_Name(self, node):
        print("{}Name: {}".format(self._indentStr * self._indent, node.id))
        if isinstance(node.ctx, _ast.Store):
            self._store.put(node.id)
        elif isinstance(node.ctx, _ast.Load):
            self._load.put(node.id)
        else:
            print("visit_Name: ctx is unexpected type: {}".format(type(node.ctx)))
    
    def visit_For(self, node):
        print("{}For: {} in ".format(self._indentStr * self._indent, node.target.id))
        #TODO: need to do something to iterate through the iters
        self.visit(node.target)
        if not self._store.empty():
            target = self._store.get_nowait()                
            self.visit(node.iter)
            while not self._load.empty():
                params = {}
                params[target] = self._load.get_nowait()
                self._state.recurse(iter(node.body), params)
        self.visit_next()
    
    def visit_Assign(self, node):
        print("{}Assign: ".format(self._indentStr * self._indent))
        self._indent += 1
        storeQSizeBefore = self._store.qsize()
        loadQSizeBefore = self._load.qsize()
        for target in node.targets:
            self.visit(target)
        print("{}Assign from:".format(self._indentStr * self._indent))
        self.visit(node.value)
        while self._store.qsize() > storeQSizeBefore and self._load.qsize() > loadQSizeBefore:
            target = self._store.get_nowait()
            self._params[target] = self._load.get_nowait()
            print("{}Assign: {} <- {}".format(self._indentStr * self._indent, target, self._params[target]))
        while self._store.qsize() > storeQSizeBefore:
            print("{}Assign: unfilled target: {}".format(self._indentStr * self._indent, self._store.get_nowait()))
        while self._load.qsize() > loadQSizeBefore:
            print("{}Assign: unused value: {}".format(self._indentStr * self._indent, self._load.get_nowait()))
        self._indent -= 1
    
    def visit_Call(self, node):
        print("{}Call: {}".format(self._indentStr * self._indent, node.func.id))
        loadSizeBefore = self._load.qsize()
        for arg in node.args:
            self.visit(arg)
        args = deque()
        while self._load.qsize() > loadSizeBefore:
            args.appendleft(self._load.get_nowait())
        print("{}Call: args list: {}".format(self._indentStr * self._indent, list(args)))
        # no call to visit here, so execution breaks

    def visit_alias(self, node):
        print("{}alias: {} as {}".format(self._indentStr * self._indent, node.name, node.asname))

    def visit_Num(self, node):
        print("{}Num: {}".format(self._indentStr * self._indent, node.n))
        # ctx of Num is always Load
        self._load.put(node.n)

    def visit_If(self, node):
        print("{}If".format(self._indentStr * self._indent))
        self._indent += 1
        print("{}test: ".format(self._indentStr * self._indent))
        self.visit(node.test)
        # evaluate the Compare, and push and visit either "body" or "orelse"
        print("{}pushing Else:".format(self._indentStr * self._indent))
        self._state.recurse(node.orelse, self._params)
        print("{}pushing and visiting Ifso: ".format(self._indentStr * self._indent))
        self._state.recurse(node.body, self._params)
        self.visit_next()
        self._indent -= 1
        self._indent -= 1

    def visit_List(self, node):
        print("{}List".format(self._indentStr * self._indent))
        for elt in node.elts:
            self.visit(elt)
    
    def visit_Tuple(self, node):
        print("{}Tuple".format(self._indentStr * self._indent))
        for elt in node.elts:
            self.visit(elt)

    def visit_next(self):
        try:
            node, self._params = next(self._state)
            self.visit(node)
        except StopIteration:
            print("visit_next: caught StopIteration")
        except Empty:
            print("visit_next: caught Empty")
    
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
    


class FauxFramework():
    """
    FauxFramework is a simple replacement for the Keck DRP Framework
    The purpose is to work out how to integrate our AST=based pipeline
    into such a framework.
    """

    _event_queue = Queue()

    def __init__(self, v):
        self._v = v
    
    def queue_push(self, item):
        self._event_queue.put(item)
    
    def execute(self):
        item = self._event_queue.get()

