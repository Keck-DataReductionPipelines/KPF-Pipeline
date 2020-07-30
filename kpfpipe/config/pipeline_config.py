# pipeline_config.py
"""
(explanation here)
"""

import keckdrpframework.config.framework_config as fc

class Struct(object):
    """ Object that supports access by attribute, as well as subscript """

    def __init__(self, arg=None, **kwargs):
        self.arg = []
        if isinstance(arg, tuple):
            self.arg[0] = arg[1]
        elif isinstance(arg, dict):
            for k, v in arg.items():
                setattr(self, k, v)
        for k, v in kwargs:
            setattr(self, k, v)
    
    def __iter__(self):
        for item in self.arg:
            yield item
        for item in self.__dict__:
            if item == 'arg':
                continue
            yield item

    def __getitem__(self, k):
        return self.__dict__[k]

    def getValue(self, k):
        return self[k]

class ConfigClass(fc.ConfigClass):
    """ ConfigClass for KPF Pipeline
    Supports access to config object from within recipes.
    This requires special features beyond the Keck framework's
    ConfigClass
    """

    def __init__(self, cfgfile=None, **kwargs):
        super(ConfigClass, self).__init__(cgfile=cfgfile, **kwargs)
    
    def read(self, cgfile):
        """ override of framework implemnetation """
        def digestItems(sec, known):
            values = self.items(sec)
            secValues = {}
            for k, v in values:
                if k in known:
                    continue
                secValues[k] = self._getType(v)
            return secValues

        def digestItemsStruct(sec, known):
            values = self.items(sec)
            # secValues = {}
            secValues = Struct()
            for k, v in values:
                if k in known:
                    continue
                # secValues[k] = self._getType(v)
                setattr(secValues, k, self._getType(v))
            return secValues

        path = self._getPath(cgfile)
        if path is None:
            return
        super().read(path)

        self.properties.update(digestItems(self.default_section, {}))
        sections = self.sections()

        for sec in sections:
            self.properties[sec] = digestItemsStruct(sec, self.properties)

