"""
Extension manifest for the KPF data models.

Single home for the extension reference data, the structural twin of
``keyword_registry``: that one owns the header cards, this one owns the
extensions they live on. ``ExtensionManifest`` builds every lookup once in
``__init__``; the module exposes a single instance, ``extension_manifest``.
``KPFDataModel`` (data_models/base.py) surfaces it as a class attribute, so
consumers handed a ``kpf_obj`` reach it through ``kpf.extension_manifest``.

Source of truth: ``config/{data_model}-extensions.csv``, one per data model
(``L0 L1 L2 L4 ML1 ML2-flat ML2-wls``, discovered from the filenames). Each is
the complete, literal statement of that model's shape: every row is created,
with no ``Required`` gate, so ``Required`` is a compliance label and ``HDU`` an
ordinal, neither read here.

Columns read: ``Name`` (the extension), ``DataType`` (its astropy HDU class),
``BitDepth`` (the exact width ``KPFDataModel._assert_bit_depth`` enforces, blank
where unconstrained) and ``Description`` (the EXT_DESCRIPT text).
"""

import importlib.resources
from types import MappingProxyType

import pandas as pd

_kpf_pipe_cfg = importlib.resources.files("kpfpipe.data_models.config")

_SUFFIX = "-extensions.csv"


class ExtensionManifest:
    """Owns the KPF extension manifests and the lookups derived from them.

    Built once at import (the module exposes the singleton
    ``extension_manifest``). All attributes are read-only reference data.
    """

    def __init__(self):
        names = {}
        fits_types = {}
        bit_depths = {}
        descriptions = {}
        paths = sorted(
            (p for p in _kpf_pipe_cfg.iterdir() if p.name.endswith(_SUFFIX)),
            key=lambda p: p.name,
        )
        for path in paths:
            data_model = path.name[: -len(_SUFFIX)]
            table = pd.read_csv(path)
            names[data_model] = tuple(str(n).strip() for n in table["Name"])
            for row in table.itertuples(index=False):
                key = (data_model, str(row.Name).strip())
                fits_types[key] = str(row.DataType).strip()
                bit_depths[key] = None if pd.isna(row.BitDepth) else int(row.BitDepth)
                descriptions[key] = (
                    "" if pd.isna(row.Description) else str(row.Description).strip()
                )
        self._names = MappingProxyType(names)
        self._fits_types = MappingProxyType(fits_types)
        self._bit_depths = MappingProxyType(bit_depths)
        self._descriptions = MappingProxyType(descriptions)

    def names(self, data_model):
        """``data_model``'s extension names, in manifest order."""
        if data_model not in self._names:
            raise ValueError(
                f"unknown data model {data_model!r}; expected one of "
                f"{sorted(self._names)}"
            )
        return self._names[data_model]

    def fits_type(self, data_model, name):
        """The astropy HDU class ``name`` is declared as (``ImageHDU``, ...)."""
        return self._fits_types[(data_model, name)]

    def bit_depth(self, data_model, name):
        """``name``'s declared BitDepth, or None when blank or undeclared."""
        return self._bit_depths.get((data_model, name))

    def description(self, data_model, name):
        """``name``'s EXT_DESCRIPT text, or ``""`` when blank or undeclared."""
        return self._descriptions.get((data_model, name), "")


# Module singleton -- the one manifest instance every consumer reaches through.
extension_manifest = ExtensionManifest()
