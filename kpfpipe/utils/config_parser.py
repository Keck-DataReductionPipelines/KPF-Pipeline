import tomllib
from pathlib import Path

class ConfigHandler:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.config = self.load_config()

    def load_config(self, path: str | Path | None = None):
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