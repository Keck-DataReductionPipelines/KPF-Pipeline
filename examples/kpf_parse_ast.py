# kpf_parse_ast.py

from ast import NodeVisitor

class KpfPipelineNodeVisitor(NodeVisitor):
    """
    Node visitor to convert KPF pipeline recipes expressed in python syntax
    into operations on the KPF Framework.

    Prototype version!
    """

    _indentStr = ""
    _indent = 0

    def __init__(self):
        NodeVisitor.__init__(self)
        self._indentStr = "  "
        self._indent = 0
    
    def visit_ImportFrom(self, node):
        names = node.names[0].name
        for name in node.names[1:]:
            names += ", " + name.name
        print("Import {} from {}".format(names, node.module))
        self.generic_visit(node)
    
    def visit_Name(self, node):
        print("{}Name: {}".format(self._indentStr * self._indent, node.id))
    
    def visit_For(self, node):
        print("{}For: {} in ".format(self._indentStr * self._indent, node.target.id))
        self._indent += 1
        self.visit(node.iter)
        print("{}For body: ".format(self._indentStr * self._indent))
        self._indent += 1
        for item in node.body:
            self.visit(item)
        self._indent -= 1
        self._indent -= 1
    
    def visit_Assign(self, node):
        print("{}Assign: ".format(self._indentStr * self._indent))
        self._indent += 1
        for target in node.targets:
            self.visit(target)
        print("{}Assign from:".format(self._indentStr * self._indent))
        self.visit(node.value)
        self._indent -= 1
    
    def visit_Call(self, node):
        print("{}Call: {}".format(self._indentStr * self._indent, node.func.id))
        self._indent += 1
        for arg in node.args:
            self.visit(arg)
        self._indent -= 1

    def visit_Num(self, node):
        print("{}Num: {}".format(self._indentStr * self._indent, node.n))

    def visit_If(self, node):
        print("{}If".format(self._indentStr * self._indent))
        self._indent += 1
        print("{}Ifso: ".format(self._indentStr * self._indent))
        for item in node.body:
            self.visit(item)
        print("{}Else:".format(self._indentStr * self._indent))
        for item in node.orelse:
            self.visit(item)
        self._indent -= 1
