# kpf_arguments.py

"""
KpfArguments is a class derived from Arguments.
The purpose is to extend the capabilities of an arguments class to support
both keyword and positional arguments in a clean and convenient way.
The current Arguments class only supports keyword arguments, which is
especially limiting for return values from functions.  It means that
developers of pipelines and other functions must know the keywords used by
the developer of other functions internal to those functions.
"""

from keckdrpframework.models.arguments import Arguments

class KpfArguments(Arguments):
    """
    KpfArguments class
    Extends Arguments to support positional arguments in addition
    to keywoard arguments.
    Positional arguments are stored on an internal args list.
    Support is included for indexing the KpfArguments class.  Doing
    so only gives or sets positional arguments.  keyword arguments
    are not accessible via the index.
    append and pop are also supported via passthru.
    """

    def __init__(self, *args, **kwargs):
        Arguments.__init__(self)
        self.name = "undef"
        self._kpfargs = []
        self._kpfargs.extend(args)
        self.__dict__.update(kwargs)

    def __str__(self):
        out = []
        [out.append(f"{arg}") for arg in self._kpfargs]
        [out.append(f"{k}: {v}") for k, v in self.__dict__.items() if k != '_kpfargs']
        return ", ".join(out)

    def __iter__(self):
        return iter(self._kpfargs)
    
    def __getitem__(self, ix):
        return self._kpfargs.__getitem__(ix)
    
    def __setitem__(self, ix, val):
        self._kpfargs.__setitem__(ix, val)

    def insert(self, ix, val):
        self._kpfargs.insert(ix, val)

    def append(self, val):
        self._kpfargs.append(val)
    
    def pop(self):
        return self._kpfargs.pop()

    def __len__(self):
        return len(self._kpfargs)