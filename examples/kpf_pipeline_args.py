# kpf_pipeline_args.py

class KpfPipelineArgs():
    """
    Args class for primatives of a KPF Pipeline
    We need to hold the AST representing the recipe as well as the
    actual function arguments.

    Because the args of the next event / action are the result of the
    previous event / action, all Primitives must return an instance of
    PipeArgs with tree set.

    visitor:    The parent class instance
    tree:       An AST representing the recipe
    args:       The actual function arguments / output, as a tuple or list
    """

    def __init__(self, visitor, tree, args):
        self.visitor = visitor
        self.tree = tree
        self.args = args

