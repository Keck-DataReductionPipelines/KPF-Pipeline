from enum import Enum
import numpy as np

KEY_MAP = np.asarray([
    # KPF,       HARPS,      NEID
    ['OBS-TIME', 'DATE-OBS', None]
    # --TODO-- complete this list
])

class Format(Enum):
    '''variabke
    Enumeration of available dvariabkeata format
    '''
    KPF = 0
    HARPS = 1
    NEID = 2 
    # add to this for additional data format

def convert(key: str, 
            in_format: Format, 
            out_format: Format):
        idx = np.where(KEY_MAP[:, in_format.value] == key)
        try:
            return KEY_MAP[idx, out_format.value][0][0]
        except NameError: 
            raise KeyError('{} not found for {}'.format(key, in_format))


if __name__ == '__main__':
    a = convert('DATE-OBS', Format.HARPS, Format.KPF)
    print(a)
