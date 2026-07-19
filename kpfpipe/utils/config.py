"""ConfigHandler: load TOML config files with optional overrides."""

import logging
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)


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
                if (
                    section in self.config
                    and isinstance(self.config[section], dict)
                    and isinstance(values, dict)
                ):
                    self.config[section].update(values)
                else:
                    self.config[section] = values

    def load_config(self, path=None):
        """Load/reload the TOML into ``self.config`` (optional ``path`` override).

        Raises
        ------
        FileNotFoundError
            If the config path does not exist.
        tomllib.TOMLDecodeError
            If the file is not valid TOML.
        """
        if path is not None:
            self.path = Path(path)

        with open(self.path, "rb") as f:
            self.config = tomllib.load(f)

        return self.config

    def get_params(self, sections=None):
        """Flatten ``sections`` into one dict; nested dicts join as ``key_subkey``."""
        if not self.config:
            self.load_config()

        if sections is None:
            sections = ["DATA_DIRS", "TRACES"]

        params = {}
        for section in sections:
            if section not in self.config:
                raise KeyError(f"Config section {section!r} absent from {self.path}")
            section_cfg = self.config[section]
            if section_cfg:
                logger.debug(
                    "Config section %r loaded (%d entries)", section, len(section_cfg)
                )
            else:
                logger.debug(
                    "Config section %r present but empty; no params contributed",
                    section,
                )

            for key, value in section_cfg.items():
                if isinstance(value, dict):
                    for subkey, subval in value.items():
                        params[f"{key}_{subkey}"] = subval
                else:
                    params[key] = value

        return params
