"""Frame-type applicability tables for the QC and diagnostics layers.

Which checks run on which exposures. Without this, a pointing check runs on a
Bias and fails, polluting the QUALITY_CONTROL header and the logs with a result
that was never meaningful.

Source of truth: ``config/{class}-applicability.csv``, one per ``QC`` or
``Diagnostics`` subclass, discovered from the filenames. Rows are keyed by
``Method`` (``<Class>.<method>``, the name the base classes' MRO walk yields) and
carry one 0/1 column per frame type. The rest is documentation, checked for drift
by the tests but not read here: ``RequiredData`` (the extensions the check reads,
``|``-separated), ``Description``, and -- on the diagnostics tables, whose methods
each emit several -- a ``Keywords`` column listing every keyword the method writes.
"""

from types import MappingProxyType

import pandas as pd

from kpfpipe.quality_control.config import PATH as _config_path

# Every frame type a check can be declared applicable to; the boolean columns.
FRAME_TYPES = ("Star", "Sun", "Bias", "Dark", "Flat", "LFC", "ThAr", "Etalon")

_SUFFIX = "-applicability.csv"


class Applicability:
    """Owns the applicability tables and the lookups derived from them.

    Built once at import (the module exposes the singleton ``applicability``).
    """

    def __init__(self):
        tables = {}
        paths = sorted(
            (p for p in _config_path.iterdir() if p.name.endswith(_SUFFIX)),
            key=lambda p: p.name,
        )
        for path in paths:
            class_name = path.name[: -len(_SUFFIX)]
            # utf-8-sig: a BOM would otherwise become part of the first column name.
            table = pd.read_csv(path, encoding="utf-8-sig")
            missing = [c for c in FRAME_TYPES if c not in table.columns]
            if missing:
                raise ValueError(f"{path.name} has no {missing} column(s)")
            tables[class_name] = MappingProxyType(
                {
                    self._method_name(row.Method, class_name, path.name): frozenset(
                        frame for frame in FRAME_TYPES if getattr(row, frame)
                    )
                    for row in table.itertuples(index=False)
                }
            )
        self._tables = MappingProxyType(tables)

    @staticmethod
    def _method_name(method, class_name, source):
        """Split a ``<Class>.<method>`` cell, requiring ``<Class>`` to be its own."""
        owner, _, name = str(method).strip().rpartition(".")
        if owner != class_name or not name:
            raise ValueError(
                f"{source}: {method!r} is not a {class_name} method; every row "
                f"must name '{class_name}.<method>'"
            )
        return name

    def _table(self, class_name):
        if class_name not in self._tables:
            raise ValueError(
                f"no config/{class_name}{_SUFFIX}; every QC and diagnostics class "
                "must declare which frame types its checks apply to"
            )
        return self._tables[class_name]

    def methods(self, class_name):
        """``class_name``'s declared method names, unqualified."""
        return frozenset(self._table(class_name))

    def applies(self, class_name, method, frame_type):
        """Whether ``class_name.method`` runs on a ``frame_type`` exposure."""
        table = self._table(class_name)
        if method not in table:
            raise ValueError(f"config/{class_name}{_SUFFIX} has no row for {method!r}")
        return frame_type in table[method]


# Module singleton -- the one instance the base classes reach through.
applicability = Applicability()
