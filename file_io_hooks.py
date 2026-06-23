"""WMKO file-I/O hook placeholder.

This module is intentionally tiny: WMKO deployments may replace it to integrate
pipeline output writes with observatory/archive infrastructure. For local
development the hook silently does nothing.
"""


def file_write_hook(
    koaid: str,
    filepath: str,
    start_time: str,
    data_level: str = "lev1",
):
    """No-op file-write hook for non-WMKO local runs."""
    pass
