"""Tests for the WavelengthCalibration module."""

from pathlib import Path

import numpy as np
import pytest

from kpfpipe import DETECTOR
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.masters.level2 import KPFMasterL2
from kpfpipe.modules.wavelength_calibration import WavelengthCalibration
from kpfpipe.utils.astro import air_to_vac
from kpfpipe.utils.config import ConfigHandler

_SCIENCE_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "kpf_drp_science.toml"

# Truth-frame integration data (see TestSpectrumOrientation).
TESTDATA_DIR = Path(__file__).parent / "testdata"
OBS_ID = "KP.20240405.40113.57"  # a bright G-type RV-standard exposure


NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]

_FIBERS = ["SKY", "SCI1", "SCI2", "SCI3", "CAL"]
_CHIPS = ["GREEN", "RED"]

# Small detector width for fast tests; the WLS copy is agnostic to NCOL.
NCOL = 32


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_master_l2(seed=42):
    """Build a KPFMasterL2 with deterministic, distinct per-fiber WAVE arrays.

    Each (chip, fiber) gets its own random-but-reproducible block so we can
    later confirm that exactly the right block lands on the science L2.
    """
    master = KPFMasterL2()
    master.headers["PRIMARY"]["INSTRUME"] = "KPF"
    master.headers["PRIMARY"]["DATE-OBS"] = "2024-04-05T01:00:37"

    rng = np.random.default_rng(seed)
    for chip in _CHIPS:
        norder = NORDER_GREEN if chip == "GREEN" else NORDER_RED
        for fiber in _FIBERS:
            arr = rng.uniform(4000.0, 8000.0, size=(norder, NCOL)).astype(np.float32)
            master.data[f"{chip}_{fiber}_WAVE"] = arr
    return master


def _make_science_l2(wls_path=None):
    """Build a minimal KPF2 for wavecal tests."""
    l2 = KPF2()
    l2.headers["PRIMARY"]["INSTRUME"] = "KPF"
    l2.headers["PRIMARY"]["DATE-OBS"] = "2024-04-05T11:08:33"
    if wls_path is not None:
        # WLSFILE lives in INSTRUMENT_HEADER on L2 (preserves the L1 PRIMARY
        # written by CalibrationAssociation).
        l2.headers["INSTRUMENT_HEADER"]["WLSFILE"] = wls_path
    return l2


@pytest.fixture
def master_wls_path(tmp_path):
    """Write a synthetic KPFMasterL2 to disk and return its path."""
    master = _make_master_l2()
    path = str(tmp_path / "kpf_ML2_20240405T010037.fits")
    master.to_fits(path)
    return path


# ---------------------------------------------------------------------------
# Constructor / config plumbing
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_chips_and_fibers(self):
        mod = WavelengthCalibration(_make_science_l2())
        assert mod.chips == _CHIPS
        assert mod.fibers == _FIBERS

    def test_dict_config_override(self):
        mod = WavelengthCalibration(
            _make_science_l2(),
            config={"chips": ["GREEN"], "fibers": ["SCI2"]},
        )
        assert mod.chips == ["GREEN"]
        assert mod.fibers == ["SCI2"]

    def test_config_handler_accepted(self):
        config = ConfigHandler(str(_SCIENCE_CONFIG_PATH))
        mod = WavelengthCalibration(_make_science_l2(), config=config)
        assert mod.chips == _CHIPS
        assert mod.fibers == _FIBERS

    def test_invalid_config_type(self):
        with pytest.raises(TypeError):
            WavelengthCalibration(_make_science_l2(), config="not a dict")

    def test_results_is_none_before_perform(self):
        mod = WavelengthCalibration(_make_science_l2())
        assert mod._results is None

    def test_wls_path_is_none_before_load(self):
        mod = WavelengthCalibration(_make_science_l2())
        assert mod._wls_path is None


# ---------------------------------------------------------------------------
# load_wls()
# ---------------------------------------------------------------------------


class TestLoadWLS:
    def test_raises_when_wlsfile_missing(self):
        mod = WavelengthCalibration(_make_science_l2())
        with pytest.raises(KeyError, match="WLSFILE"):
            mod.load_wls()

    def test_raises_when_file_does_not_exist(self, tmp_path):
        bogus = str(tmp_path / "does_not_exist.fits")
        mod = WavelengthCalibration(_make_science_l2(wls_path=bogus))
        with pytest.raises(FileNotFoundError, match="Master WLS file not found"):
            mod.load_wls()

    def test_reads_wlsfile_from_instrument_header(self, master_wls_path):
        mod = WavelengthCalibration(_make_science_l2(wls_path=master_wls_path))
        loaded = mod.load_wls()
        assert isinstance(loaded, KPFMasterL2)
        assert mod._wls_path == master_wls_path

    def test_explicit_path_overrides_header(self, master_wls_path):
        # Set WLSFILE to a bogus value to make sure the override wins.
        mod = WavelengthCalibration(_make_science_l2(wls_path="/tmp/bogus.fits"))
        loaded = mod.load_wls(wls_path=master_wls_path)
        assert isinstance(loaded, KPFMasterL2)
        assert mod._wls_path == master_wls_path


# ---------------------------------------------------------------------------
# perform()
# ---------------------------------------------------------------------------


class TestPerform:
    def test_returns_l2_obj(self, master_wls_path):
        l2 = _make_science_l2(wls_path=master_wls_path)
        mod = WavelengthCalibration(l2)
        assert mod.perform() is l2

    def test_adds_receipt_entry(self, master_wls_path):
        l2 = _make_science_l2(wls_path=master_wls_path)
        WavelengthCalibration(l2).perform()
        assert (l2.receipt["Module_Name"] == "wavelength_calibration").any()

    def test_results_populated_after_perform(self, master_wls_path):
        l2 = _make_science_l2(wls_path=master_wls_path)
        mod = WavelengthCalibration(l2)
        mod.perform()
        assert mod._results["wls_path"] == master_wls_path
        assert mod._results["chips"] == _CHIPS
        assert mod._results["fibers"] == _FIBERS

    def test_copies_all_wave_arrays(self, master_wls_path):
        # Every (chip, fiber) WAVE array on the science L2 should match the master.
        l2 = _make_science_l2(wls_path=master_wls_path)
        WavelengthCalibration(l2).perform()

        master = KPFMasterL2.from_fits(master_wls_path)
        for chip in _CHIPS:
            for fiber in _FIBERS:
                key = f"{chip}_{fiber}_WAVE"
                np.testing.assert_array_equal(l2.data[key], master.data[key])

    def test_wave_arrays_are_float64(self, master_wls_path):
        # The fixture master is float32 (4-byte); the science WAVE must be float64.
        master = KPFMasterL2.from_fits(master_wls_path)
        assert master.data["GREEN_SCI2_WAVE"].dtype.itemsize == 4
        l2 = _make_science_l2(wls_path=master_wls_path)
        WavelengthCalibration(l2).perform()
        for chip in _CHIPS:
            for fiber in _FIBERS:
                assert l2.data[f"{chip}_{fiber}_WAVE"].dtype == np.float64

    def test_explicit_path_bypasses_header(self, master_wls_path):
        # WLSFILE header is bogus, but wls_path override is valid → perform() succeeds.
        l2 = _make_science_l2(wls_path="/tmp/bogus.fits")
        WavelengthCalibration(l2).perform(wls_path=master_wls_path)

        master = KPFMasterL2.from_fits(master_wls_path)
        np.testing.assert_array_equal(
            l2.data["GREEN_SCI2_WAVE"], master.data["GREEN_SCI2_WAVE"]
        )

    def test_subset_chips_and_fibers(self, master_wls_path):
        # Only the requested (chip, fiber) blocks are copied; everything
        # else stays zero — both un-requested fibers within the requested
        # chip, and un-requested chips entirely.
        l2 = _make_science_l2(wls_path=master_wls_path)
        WavelengthCalibration(l2).perform(chips=["GREEN"], fibers=["SCI2"])

        master = KPFMasterL2.from_fits(master_wls_path)
        np.testing.assert_array_equal(
            l2.data["GREEN_SCI2_WAVE"], master.data["GREEN_SCI2_WAVE"]
        )
        # Un-requested fiber within the requested chip stays zero.
        assert not np.any(l2.data["GREEN_SCI1_WAVE"])
        # Un-requested chip stays zero.
        assert not np.any(l2.data["RED_SCI2_WAVE"])

    def test_raises_when_wlsfile_missing(self):
        l2 = _make_science_l2()  # no WLSFILE
        with pytest.raises(KeyError, match="WLSFILE"):
            WavelengthCalibration(l2).perform()

    def test_raises_when_master_missing_requested_fiber(self, tmp_path):
        # Master only has SCI2; config asks for all 5 fibers → fail loudly.
        master = KPFMasterL2()
        master.headers["PRIMARY"]["INSTRUME"] = "KPF"
        master.headers["PRIMARY"]["DATE-OBS"] = "2024-04-05T01:00:37"
        rng = np.random.default_rng(7)
        for chip in _CHIPS:
            norder = NORDER_GREEN if chip == "GREEN" else NORDER_RED
            master.data[f"{chip}_SCI2_WAVE"] = rng.uniform(
                4000.0, 8000.0, size=(norder, NCOL)
            ).astype(np.float32)
        master_path = str(tmp_path / "partial_master.fits")
        master.to_fits(master_path)

        l2 = _make_science_l2(wls_path=master_path)
        with pytest.raises(KeyError, match="SKY_WAVE"):
            WavelengthCalibration(l2).perform()


# ---------------------------------------------------------------------------
# Spectrum orientation discriminator (real truth-frame integration)
# ---------------------------------------------------------------------------
#
# Once WavelengthCalibration assigns the master WLS, the extracted flux must be
# co-oriented with that wavelength axis: a strong stellar absorption line has to
# land at its catalog wavelength, NOT at the mirror-image wavelength that a
# flux/wave flip would produce. We test this with deep Fraunhofer lines, whose
# rest wavelengths are textbook-known.
#
# Fraunhofer absorption lines, AIR wavelengths [Angstrom]:
#   https://en.wikipedia.org/wiki/Fraunhofer_lines
# KPF covers ~4457-8700 A, so only H-beta, the Na D doublet, and H-alpha are
# actually observable; the bluer Ca II H&K and H-gamma/H-delta fall below the
# blue cutoff and are skipped automatically (kept here for documentation).

_FRAUNHOFER_AIR = {
    "Ca II K": 3933.66,
    "Ca II H": 3968.47,
    "H-delta": 4101.74,
    "H-gamma": 4340.47,
    "H-beta": 4861.34,
    "Na D2": 5889.95,
    "Na D1": 5895.92,
    "H-alpha": 6562.79,
}

_SCI_FIBERS = ["SCI1", "SCI2", "SCI3"]


def _trough_depth(wave_order, flux_order, lambda_air, halfwin=3.0):
    """Continuum-normalized absorption depth at a catalog line.

    Returns ``1 - min(flux)/continuum`` within +/-``halfwin`` Angstrom of
    ``lambda_air``, or None if the line is not covered by this order or the
    window holds no finite flux. The window is centered between the air and
    vacuum line wavelengths, so it absorbs both the air<->vacuum offset (~1.5 A)
    and the stellar+barycentric Doppler shift (~0.3 A) without needing to know
    which frame the WLS is in. Continuum is the window's 90th percentile -- over
    this narrow a span the blaze is locally flat, so its curvature is
    negligible.
    """
    lambda_vac = float(air_to_vac(np.array([lambda_air]))[0])
    center = 0.5 * (lambda_air + lambda_vac)
    sel = np.abs(np.asarray(wave_order) - center) < halfwin
    if np.count_nonzero(sel) < 5:
        return None
    window = np.asarray(flux_order, dtype=float)[sel]
    if not np.any(np.isfinite(window)):
        return None
    continuum = np.nanpercentile(window, 90)
    if not np.isfinite(continuum) or continuum <= 0:
        return None
    return 1.0 - np.nanmin(window) / continuum


@pytest.mark.skipif(
    not (TESTDATA_DIR / "L0" / "20240405" / f"{OBS_ID}.fits").is_file(),
    reason="truth-frame test data not present",
)
class TestSpectrumOrientation:
    """
    Discriminator: the WLS-calibrated L2 must have its flux co-oriented with the
    assigned wavelength axis.

    For every in-coverage Fraunhofer anchor on every science fiber, the native
    ``(wave, flux)`` pairing must show a DEEPER absorption trough at the catalog
    wavelength than the reversed ``(wave, flux[::-1])`` pairing. A flux/wave flip
    inverts that relationship and fails the test.
    """

    @pytest.fixture(scope="class")
    def science_l2(self):
        # Build the real L0->L2 chain just far enough to get extracted flux with
        # the master WLS assigned. Restricted to the SCI fibers because the truth
        # WLS master only carries those. Upstream modules are imported here, not
        # at module scope, to keep this WLS-application test file's surface small.
        from kpfpipe.data_models import KPF0
        from kpfpipe.modules.calibration_association import CalibrationAssociation
        from kpfpipe.modules.image_assembly import ImageAssembly
        from kpfpipe.modules.image_processing import ImageProcessing
        from kpfpipe.modules.spectral_extraction import SpectralExtraction
        from kpfpipe.utils.pipeline import build_filepath

        config = {
            "KPF_DATA_INPUT": str(TESTDATA_DIR),
            "chips": _CHIPS,
            "fibers": _SCI_FIBERS,
        }
        l0 = KPF0.from_fits(build_filepath(OBS_ID, "L0", data_root=str(TESTDATA_DIR)))
        l1 = ImageAssembly(l0, config).perform()
        l1 = CalibrationAssociation(l1, config).perform(
            ["bias", "dark", "flat", "thar"]
        )
        l1 = ImageProcessing(l1, config).perform()
        l2 = SpectralExtraction(l1, config).perform(verbose=False)
        return WavelengthCalibration(l2, config).perform()

    def _anchor_cases(self, l2):
        """List in-coverage anchors as (name, chip, fiber, order, wave, flux, lambda_air)."""
        cases = []
        for name, lambda_air in _FRAUNHOFER_AIR.items():
            for chip in _CHIPS:
                for fiber in _SCI_FIBERS:
                    wave = np.asarray(l2.data[f"{chip}_{fiber}_WAVE"])
                    flux = np.asarray(l2.data[f"{chip}_{fiber}_FLUX"], dtype=float)
                    for o in range(wave.shape[0]):
                        if wave[o].min() + 4 < lambda_air < wave[o].max() - 4:
                            cases.append(
                                (name, chip, fiber, o, wave[o], flux[o], lambda_air)
                            )
                            break
        return cases

    def test_expected_anchors_in_coverage(self, science_l2):
        """The observable Fraunhofer lines are found; the bluer ones are out of range."""
        names = {c[0] for c in self._anchor_cases(science_l2)}
        assert {"H-beta", "Na D2", "Na D1", "H-alpha"} <= names
        assert names.isdisjoint({"Ca II K", "Ca II H", "H-delta", "H-gamma"})

    def test_flux_co_oriented_with_wave(self, science_l2):
        """Anchors sit in deeper troughs in the native pairing than reversed.

        Orientation is a single global property of the FFI, so the verdict is
        aggregate: the mean native trough depth must clearly exceed the mean
        reversed depth. (A per-anchor comparison is brittle for lines near an
        order center, where the mirror wavelength falls back onto the line.)
        """
        cases = self._anchor_cases(science_l2)
        assert cases, "no Fraunhofer anchor lines found in coverage"

        rows, natives, reverses = [], [], []
        for name, chip, fiber, o, wave, flux, lambda_air in cases:
            native = _trough_depth(wave, flux, lambda_air)
            reverse = _trough_depth(wave, flux[::-1], lambda_air)
            if native is None or reverse is None:
                continue
            natives.append(native)
            reverses.append(reverse)
            rows.append(
                f"  {name} {chip} {fiber} ord{o}: native {native:.3f}  reversed {reverse:.3f}"
            )

        mean_native, mean_reverse = float(np.mean(natives)), float(np.mean(reverses))
        assert mean_native > mean_reverse + 0.15, (
            f"Extracted flux is not co-oriented with the wavelength solution: "
            f"mean native depth {mean_native:.3f} <= mean reversed {mean_reverse:.3f} "
            f"(flux flipped relative to WLS?)\n" + "\n".join(rows)
        )

    def test_strong_anchor_is_deep(self, science_l2):
        """Floor check: the Na D doublet must be a deep trough in the native
        pairing, so the discriminator cannot pass trivially on near-zero flux."""
        deepest = 0.0
        for name, _chip, _fiber, _o, wave, flux, lambda_air in self._anchor_cases(
            science_l2
        ):
            if name in ("Na D1", "Na D2"):
                d = _trough_depth(wave, flux, lambda_air)
                if d is not None:
                    deepest = max(deepest, d)
        assert deepest > 0.7, (
            f"deepest native Na D trough only {deepest:.3f}; expected a deep line"
        )
