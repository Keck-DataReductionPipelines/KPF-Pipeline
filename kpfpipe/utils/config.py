import tomllib
from pathlib import Path

class ConfigHandler:
    """Load a TOML config file, with optional in-memory section overrides.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the TOML config file.
    overrides : dict or None, optional
        Per-section overrides applied after loading; each value is merged into
        the matching section dict, or replaces the section if it is not a dict.
    """

    def __init__(self, path, overrides=None):
        self.path = Path(path)
        self.config = self.load_config()
        if overrides:
            for section, values in overrides.items():
                if section in self.config and isinstance(self.config[section], dict) and isinstance(values, dict):
                    self.config[section].update(values)
                else:
                    self.config[section] = values

    def load_config(self, path=None):
        if path is not None:
            self.path = Path(path)

        with open(self.path, "rb") as f:
            self.config = tomllib.load(f)

        return self.config

    def get_params(self, sections=None):
        if not self.config:
            self.load_config()

        if sections is None:
            sections = ["DATA_DIRS", "KPFPIPE"]

        params = {}
        for section in sections:
            section_cfg = self.config.get(section, {})

            for key, value in section_cfg.items():
                if isinstance(value, dict):
                    for subkey, subval in value.items():
                        params[f"{key}_{subkey}"] = subval
                else:
                    params[key] = value

        return params