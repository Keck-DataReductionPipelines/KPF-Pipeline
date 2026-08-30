"""Shared dtype-provenance names for the test suite (not a test module).

The widths below are *read from* ``data_models/config``, which is where the
float32/float64/uint8 policy is declared and where the data models enforce it.
This module only binds readable names to those declarations, so a test can say
``FLUX`` rather than spelling out a manifest lookup. It states no policy of its
own: change a ``BitDepth`` cell and these follow.

The CSVs are read directly rather than through ``KPFDataModel._bit_depth`` so the
oracle stays independent of the production lookup the tests check.

``BitDepth`` deliberately encodes width, not kind -- ``np.bool_`` and ``np.uint8``
are both 8-bit -- so the ``float``/``bool``/``uint8`` kind is named here.

Builder helpers are assertion-free. ``_dtype_policy.py`` is the one contract module;
if a cross-cutting contract genuinely needs assertions, promote it to a new contract
module and justify the contract.
"""

import importlib.resources

import numpy as np
import pandas as pd

_CFG = importlib.resources.files("kpfpipe.data_models.config")


def _float(table, name):
    """The float dtype ``table``'s ``name`` row declares."""
    rows = pd.read_csv(_CFG / f"{table}.csv")
    bits = int(rows.loc[rows["Name"] == name, "BitDepth"].iloc[0])
    return np.dtype(f"float{bits}")


# --- the policy matrix, as declared in data_models/config ------------------

L1_IMAGE = _float("L1-extensions", "GREEN_CCD")  # L1 *_CCD/*_VAR, master *_IMG/*_SNR
FLUX = _float("L2-extensions", "TRACE1_FLUX")  # L2 *_FLUX/*_VAR/*_BLAZE
WAVE = _float("L2-extensions", "TRACE1_WAVE")  # *_WAVE everywhere (EPRV, born-64)
BJD = _float("L2-extensions", "BJD_TDB")  # BJD_TDB (EPRV, born-64)
BARYCORR = _float("L2-extensions", "BARYCORR_KMS")  # BARYCORR_KMS/_Z
CCF = _float("L4-extensions", "CCF1")  # L4 CCF cubes
RV_FLOAT = _float("L4-RV-columns", "RV")  # RV/RV_ERR/BERV/WAVE_START/WAVE_END
WLS_COEFFS = _float("ML2-wls-extensions", "GREEN_WLS_COEFFS")  # Legendre coefficients
MASK_MEM = np.bool_  # master MASK in memory (8-bit, like the disk form)
MASK_DISK = np.uint8  # master MASK on disk (EPRV 8-bit, BITPIX=8)

# numpy BITPIX <-> dtype, for asserting the on-disk form after a round-trip.
# negative BITPIX is the FITS/IEEE convention for floating point quantities
_BITPIX = {
    np.dtype(np.float32): -32,
    np.dtype(np.float64): -64,
    np.dtype(np.uint8): 8,
}


def assert_dtype(arr, expected, label):
    """Assert ``arr`` has ``expected`` precision (kind + itemsize).

    Kind/itemsize rather than the dtype object, so byte-order is ignored: FITS
    round-trips to big-endian, and ``>f4`` is still float32.
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


# Convention-conforming output names, one per model class, so a round-trip does not
# have to spell one out just to dodge the off-convention write warning. Deliberately
# literal and NOT derived from kpfpipe.utils.io.kpf_filename: the filename convention
# is independently asserted in test_io.py, and calling production here would launder
# that oracle. Keyed on the class name so this module imports no data model.
_ROUNDTRIP_NAMES = {
    "KPF0": "KP.20240113.23249.10.fits",
    "KPF1": "kpf_L1_20240113T102656.fits",
    "KPF2": "kpf_SL2_20240101T000000.fits",
    "KPF4": "kpf_SL4_20240101T000000.fits",
    "KPFMasterL1": "KP.20240113.23249.10_master_bias_L1.fits",
    "KPFMasterL2": "KP.20240113.23249.10_master_thar_L2.fits",
}


def assert_roundtrip_dtype(
    model_cls, obj, ext, expected, tmp_path, name=None, expected_disk=None
):
    """Write ``obj`` to FITS, re-read it, and assert ``ext`` round-trips correctly.

    ``expected`` is the in-memory dtype after re-read; ``expected_disk`` is the
    on-disk dtype (BITPIX), defaulting to ``expected``. They differ for MASK,
    which is bool in memory but uint8 on disk.

    ``name`` overrides the output filename; it defaults to this model's entry in
    ``_ROUNDTRIP_NAMES`` because every level warns on an off-convention write
    (L2/L4 via rvdata, the rest via check_filename_convention).
    """
    expected_disk = expected if expected_disk is None else expected_disk
    if name is None:
        try:
            name = _ROUNDTRIP_NAMES[model_cls.__name__]
        except KeyError:
            raise KeyError(
                f"no conventional round-trip filename for {model_cls.__name__}; "
                "add one to _ROUNDTRIP_NAMES or pass name= explicitly"
            ) from None
    out = str(tmp_path / name)
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
