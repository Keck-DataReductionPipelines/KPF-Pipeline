# Tests related to initialization of Framework and other low-level classes

from kpfpipe.config.pipeline_config import ConfigClass, Struct

def test_config_parse():
    config_file = "examples/default_kpf.cfg"

    cfg = ConfigClass(config_file)


def test_struct_class():
    arg = {'first': 0, 'second': 1}

    st = Struct(arg)
    for item in st:
        print(item)