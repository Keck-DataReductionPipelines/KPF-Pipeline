"""Shared, tools-free helpers for the processing scripts.

Home of the reduction *registry* (the ``masters``/``science`` -> recipe/config
mapping) and small filesystem helpers reused by the leaf runner (``reduce.py``)
and the batch orchestrators (``masters.py``/``science.py``). This module sits at
the ``scripts/`` layer and depends only on ``kpfpipe`` -- never on ``tools`` --
so the scripts stay runnable without knowledge of the CLI dispatcher above them.
"""

import os

import kpfpipe
from kpfpipe.utils.kpf_utils import is_datecode

_REPO = kpfpipe.REPO_ROOT

# The single source of the reduction -> (recipe, config) mapping. The leaf runner
# resolves a --masters/--science shortcut through it; the orchestrators read the
# resolved config to find the log dir and (masters) the L0 input root.
_SHORTCUTS = {
    "masters": ("recipes/kpf_drp_masters.py", "configs/kpf_drp_masters.toml"),
    "science": ("recipes/kpf_drp_science.py", "configs/kpf_drp_science.toml"),
}


def shortcut_paths(kind):
    """Absolute (recipe, config) paths for a ``--masters``/``--science`` shortcut.

    The single source of the shortcut -> recipe/config mapping (the ``_SHORTCUTS``
    table). Resolves repo-relative, so it works from any cwd. Raises ``KeyError``
    for an unknown ``kind``.
    """
    recipe_rel, config_rel = _SHORTCUTS[kind]
    return (
        os.path.join(_REPO, recipe_rel),
        os.path.join(_REPO, config_rel),
    )


def _datecode_dirs(root, start, end):
    """Sorted datecode subdirs of `root` within the inclusive [start, end] range."""
    return [
        d
        for d in sorted(os.listdir(root))
        if is_datecode(d) and start <= d <= end and os.path.isdir(os.path.join(root, d))
    ]
