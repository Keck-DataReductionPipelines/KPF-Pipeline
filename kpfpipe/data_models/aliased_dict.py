"""
OrderedDict subclasses with transparent name aliases.

``AliasedOrderedDict`` supports bidirectional alias registration: accessing
d["alias"] transparently resolves to d["canonical_key"]. Generic enough to
upstream into rvdata.

``ChipPrefixDict`` extends it with KPF's GREEN_/RED_ chip-prefix views over the
concatenated order axis, shared by the L2 and L4 data dicts.
"""

from collections import OrderedDict

import numpy as np

from kpfpipe import DETECTOR

NORDER_GREEN = DETECTOR["norder"]["GREEN"]


class AliasedOrderedDict(OrderedDict):
    """
    OrderedDict with transparent name aliases.

    Register an alias with ``register_alias(alias, canonical)``; ``__getitem__``,
    ``__setitem__``, ``__contains__`` and ``get()`` then resolve the alias to the
    canonical key before the lookup.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aliases = {}  # alias → canonical name
        self._reverse = {}  # canonical → set of aliases

    def register_alias(self, alias, canonical):
        """Register alias as a synonym for canonical."""
        self._aliases[alias] = canonical
        self._reverse.setdefault(canonical, set()).add(alias)

    def unregister_alias(self, alias):
        """Remove a previously registered alias."""
        if alias in self._aliases:
            canonical = self._aliases.pop(alias)
            self._reverse.get(canonical, set()).discard(alias)

    def _resolve(self, key):
        """Resolve an alias to its canonical key, or return key unchanged."""
        return self._aliases.get(key, key)

    def aliases_for(self, canonical):
        """Return the set of aliases registered for a canonical key."""
        return self._reverse.get(canonical, set()).copy()

    def __getitem__(self, key):
        return super().__getitem__(self._resolve(key))

    def __setitem__(self, key, value):
        super().__setitem__(self._resolve(key), value)

    def __contains__(self, key):
        return super().__contains__(self._resolve(key))

    def __delitem__(self, key):
        super().__delitem__(self._resolve(key))

    def get(self, key, default=None):
        return super().get(self._resolve(key), default)

    @classmethod
    def from_ordered_dict(cls, od):
        """Create an AliasedOrderedDict from an existing OrderedDict."""
        aliased = cls()
        for key, value in od.items():
            OrderedDict.__setitem__(aliased, key, value)
        return aliased


class ChipPrefixDict(AliasedOrderedDict):
    """Aliased dict with GREEN_/RED_ chip-prefix views over the order axis.

    Each per-order extension holds the green and red orders concatenated (green
    first, on axis 0), so a chip-prefixed key is a numpy view of that chip's
    slice: ``d["GREEN_SCI2_FLUX"]`` is ``d["SCI2_FLUX"][:NORDER_GREEN]``. A write
    to a chip-prefixed key allocates the full concatenated array on first use
    and fills that chip's half.

    Subclasses supply ``_PREFIX_KEYS`` (chip-prefixed key -> (base_key, chip));
    ``_READONLY_BASES`` names base-key suffixes that may only be written whole.
    """

    _PREFIX_KEYS = {}
    _READONLY_BASES = ()

    def _chip_split(self, key):
        """If key is a chip-prefix pattern, return (base_key, chip), else None."""
        return self._PREFIX_KEYS.get(key)

    @staticmethod
    def _chip_view(data, chip):
        """The ``chip`` half of a concatenated green-then-red array."""
        return data[:NORDER_GREEN] if chip == "GREEN" else data[NORDER_GREEN:]

    def __setitem__(self, key, value):
        split = self._chip_split(key)
        if split is None:
            super().__setitem__(key, value)
            return
        base_key, chip = split
        if self._READONLY_BASES and base_key.endswith(self._READONLY_BASES):
            raise KeyError(
                f"chip-prefixed key {key!r} is read-only; write the full table "
                f"via {base_key!r} (rows are green-then-red)"
            )
        resolved = self._resolve(base_key)
        # Allocate the full concatenated array on first write (or if empty);
        # value.shape[1:] keeps this correct for 2-D traces, CCF cubes, and
        # 1-D per-order arrays.
        existing = (
            super().__getitem__(resolved) if super().__contains__(resolved) else None
        )
        if existing is None or np.size(existing) == 0:
            full = np.zeros((DETECTOR["numorder"], *value.shape[1:]), dtype=value.dtype)
            super().__setitem__(resolved, full)
        self._chip_view(super().__getitem__(resolved), chip)[:] = value

    def __getitem__(self, key):
        split = self._chip_split(key)
        if split is not None:
            base_key, chip = split
            return self._chip_view(super().__getitem__(self._resolve(base_key)), chip)
        return super().__getitem__(self._resolve(key))

    def __contains__(self, key):
        split = self._chip_split(key)
        if split is not None:
            return super().__contains__(self._resolve(split[0]))
        return super().__contains__(self._resolve(key))

    def get(self, key, default=None):
        split = self._chip_split(key)
        if split is not None:
            base_key, chip = split
            resolved = self._resolve(base_key)
            if not super().__contains__(resolved):
                return default
            return self._chip_view(super().__getitem__(resolved), chip)
        return super().get(self._resolve(key), default)
