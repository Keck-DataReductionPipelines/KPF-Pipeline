"""Shared, tools-free filesystem helper for the processing scripts.

Home of ``_datecode_dirs``, used by the masters orchestrator to expand a datecode
range against the L0 tree. (The default recipe/config paths live in the
``scripts.processing`` package ``__init__``; the process-pool engine in
``_dispatch.py``.) Depends only on
``kpfpipe`` -- never on ``tools`` -- so the scripts stay runnable without knowledge
of the CLI dispatcher above them.
"""

import os

from kpfpipe.utils.kpf_utils import is_datecode


def _datecode_dirs(root, start, end):
    """Sorted datecode subdirs of `root` within the inclusive [start, end] range."""
    return [
        d
        for d in sorted(os.listdir(root))
        if is_datecode(d) and start <= d <= end and os.path.isdir(os.path.join(root, d))
    ]
