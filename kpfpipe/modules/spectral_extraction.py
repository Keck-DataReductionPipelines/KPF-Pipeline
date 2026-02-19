"""
KPF Image Assembly module.

Processes data from L1 to SL2.
 - extracts 1D spectrum from 2D FFI
"""

from kpfpipe import REPO_ROOT
DEFAULTS = {'extraction_method':'box'}

class SpectralExtraction:
    def __init__(self, l1_obj, config={}):
        self.l1_obj = l1_obj
        self.CHIPS = ['GREEN', 'RED']

        # TODO: check if this config parsing works
        for k in DEFAULTS.keys():
            self.__setattr__(k, config.get(k,DEFAULTS[k]))


    def _read_order_trace_reference(self, chip):
        chip = chip.upper()
        
        if not hasattr(self, 'order_trace'):
            self.order_trace = {}

        filepath = f'{REPO_ROOT}/reference/order_trace_{chip.lower()}.txt'
        with open(filepath, 'r') as f:
            self.order_trace[chip] = pd.read_csv(f, index_col=0)

        return self.order_trace[chip]
