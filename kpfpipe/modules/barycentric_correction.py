from kpfpipe.utils.config import ConfigHandler


class BarycentricCorrection:
    def __init__(self, l1_obj, config=None):
        self.l1_obj = l1_obj

        if config is None:
            params = {}
        elif isinstance(config, dict):
            params = config
        elif isinstance(config, ConfigHandler):
            params = config.get_params(["DATA_DIRS", "KPFPIPE", "MODULE_BARYCENTRIC_CORRECTION"])
        else:
            raise TypeError("config must be None, dict, or ConfigHandler")
