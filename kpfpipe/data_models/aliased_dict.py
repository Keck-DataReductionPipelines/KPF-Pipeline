"""
OrderedDict subclass with transparent name aliases.

Supports bidirectional alias registration: accessing d["alias"] transparently
resolves to d["canonical_key"]. Generic enough to upstream into rvdata.
"""

from collections import OrderedDict


class AliasedOrderedDict(OrderedDict):
    """OrderedDict with transparent name aliases.

    Register an alias with register_alias(alias, canonical). After that,
    __getitem__, __setitem__, __contains__, and get() all resolve the alias
    to the canonical key before performing the lookup.

    Usage:
        d = AliasedOrderedDict()
        d["TRACE3_FLUX"] = some_array
        d.register_alias("SCI2_FLUX", "TRACE3_FLUX")
        d["SCI2_FLUX"]  # returns some_array (same object)
        "SCI2_FLUX" in d  # True
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aliases = {}      # alias → canonical name
        self._reverse = {}      # canonical → set of aliases

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
