"""Shared dtype-provenance policy for the test suite (not a test module).

Single source of truth for the float32/float64/uint8/bool matrix the pipeline
must respect at every state — in-memory arrays, module-internal intermediates,
and the FITS-serialized form. EPRV mandates 64-bit for ``*_WAVE``/``BJD_TDB``/
``WAVE_START``/``WAVE_END`` and 8-bit for quality; the rest is KPF policy
(float32 science arrays for performance, float64 for RV/CCF precision).

Guard BOTH directions: never upscale float32->float64 (perf), never downscale
float64->float32 (precision loss -> wrong RVs).
"""

import numpy as np

# --- the policy matrix -----------------------------------------------------

L1_IMAGE = np.float32  # L1 *_CCD/*_VAR, master *_IMG/*_SNR
FLUX = np.float32  # L2 *_FLUX/*_VAR/*_BLAZE
WAVE = np.float64  # *_WAVE everywhere (EPRV, born-64)
BJD = np.float64  # BJD_TDB (EPRV, born-64)
BARYCORR = np.float64  # BARYCORR_KMS/_Z
CCF = np.float64  # L4 CCF cubes
RV_FLOAT = np.float64  # RV/RV_ERR/BERV/WAVE_START/WAVE_END (RV table)
MASK_MEM = np.bool_  # master MASK in memory
MASK_DISK = np.uint8  # master MASK on disk (EPRV 8-bit, BITPIX=8)

# numpy BITPIX <-> dtype, for asserting the on-disk form after a round-trip.
_BITPIX = {
    np.dtype(np.float32): -32,
    np.dtype(np.float64): -64,
    np.dtype(np.uint8): 8,
}


def assert_dtype(arr, expected, label):
    """Assert ``arr`` has ``expected`` precision (kind + itemsize).

    Compares kind/itemsize, not the exact dtype object, so byte-order is
    ignored (FITS round-trips to big-endian, e.g. ``>f4`` is still float32).
    The policy is about precision — float32 vs float64 vs uint8 vs bool — not
    endianness.
    """
    actual = np.asarray(arr).dtype
    exp = np.dtype(expected)
    assert (actual.kind, actual.itemsize) == (exp.kind, exp.itemsize), (
        f"{label}: expected {exp} (kind {exp.kind}, {exp.itemsize}B), got {actual}"
    )


def assert_not_float64(arr, label):
    """L0 guard: native-int or float32 is fine, but never an accidental upcast."""
    actual = np.asarray(arr).dtype
    assert not (actual.kind == "f" and actual.itemsize == 8), (
        f"{label}: must not be float64 (got {actual}) — would upscale L0/L1"
    )


def assert_roundtrip_dtype(
    model_cls, obj, ext, expected, tmp_path, name="rt.fits", expected_disk=None
):
    """Write ``obj`` to FITS, read it back with ``model_cls``, and assert the
    extension ``ext`` round-trips correctly.

    ``expected`` is the in-memory dtype after re-read; ``expected_disk`` is the
    on-disk dtype (BITPIX), defaulting to ``expected``. They differ for MASK,
    which is bool in memory but uint8 (8-bit) on disk.
    """
    import warnings

    expected_disk = expected if expected_disk is None else expected_disk
    out = str(tmp_path / name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Filename .* does not follow")
        obj.to_fits(out)

    from astropy.io import fits

    with fits.open(out) as hdul:
        bitpix = hdul[ext].header["BITPIX"]
    want = _BITPIX[np.dtype(expected_disk)]
    assert bitpix == want, (
        f"{ext} on disk: expected BITPIX {want} ({np.dtype(expected_disk)}), "
        f"got BITPIX {bitpix}"
    )

    reread = model_cls.from_fits(out)
    assert_dtype(reread.data[ext], expected, f"{ext} after round-trip")
    return reread
