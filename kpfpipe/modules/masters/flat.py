from kpfpipe.modules.masters.base import BaseMasterModule

class Flat(BaseMasterModule):
    def __init__(self, l0_file_list, config=None):
        if config is None:
            config = {}
        super().__init__(l0_file_list, config)
