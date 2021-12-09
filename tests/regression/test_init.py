# Tests related to initialization of Framework and other low-level classes

from kpfpipe.config.pipeline_config import ConfigClass

def test_config_parse():
    config_file = "examples/default_kpf.cfg"

    cfg = ConfigClass(config_file)

    for c in cfg:
        print(cfg)

