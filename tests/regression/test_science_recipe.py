"""Tests for the kpf_drp_science recipe.

Integration tests run the full recipe (L0 -> L1 -> L2 -> L4) against a real star
observation from tests/testdata/L0/20240405/.
"""

import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kpfpipe import DETECTOR
from kpfpipe.data_models.base import KPFDataModel
from kpfpipe.data_models.level2 import KPF2
from kpfpipe.data_models.level4 import KPF4
from kpfpipe.utils.config import ConfigHandler
from kpfpipe.utils.io import kpf_filepath
from recipes._logging import science_run_summary

from ._dtype_policy import CCF, RV_FLOAT, assert_dtype

# ---------------------------------------------------------------------------
# Test data paths and constants
# ---------------------------------------------------------------------------

TESTDATA_DIR = Path(__file__).parent.parent / "testdata"
TESTDATA_L0_DIR = TESTDATA_DIR / "L0" / "20240405"
CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "kpf_drp_science.toml"

OBS_ID = "KP.20240405.40113.57"

NORDER_GREEN = DETECTOR["norder"]["GREEN"]
NORDER_RED = DETECTOR["norder"]["RED"]


def _load_recipe():
    spec = importlib.util.spec_from_file_location(
        "kpf_drp_science",
        Path(__file__).parent.parent.parent / "recipes" / "kpf_drp_science.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Frozen catalog capture (stands in for the live SIMBAD/Gaia query)
# ---------------------------------------------------------------------------
#
# The four CATALOG_RECORD rows AstroQuery returned for OBS_ID on 2026-08-20,
# captured verbatim from one live query (Gaia DR3, SIMBAD).
#
# This is an oracle -- a recording of what production emits -- not a rebuild of
# it: the rows go in through AstroQuery's own writer, and merge_catalog_records
# is what produced the 'kpf-drp' row, so a merge change disagrees with this
# capture rather than silently agreeing with a reimplementation.
#
# To recapture: run AstroQuery(KPF0.from_fits(<L0>), config).perform() in a
# plain python script (NOT under pytest, whose conftest blocks the catalog
# hosts) and dump l0.data["CATALOG_RECORD"] row by row.
_CATALOG_CAPTURE = {
    "wmko": {
        "object": "95128",
        "radec_src": "wmko",
        "plx_src": "wmko",
        "rv_src": "wmko",
        "ra": "10:59:27.4994",
        "dec": "+40:25:50.0140",
        "pmra": 0.0,
        "pmdec": 0.0,
        "parallax": 72.0,
        "rv": 11.1,
        "frame": "icrs",
        "epoch": 2000.0,
        "equinox": 2000.0,
        "color": 0.9100000000000001,
        "color_name": "G-J",
    },
    "gaia": {
        "object": "Gaia DR3 777254360337133312",
        "radec_src": "gaia",
        "plx_src": "gaia",
        "rv_src": "gaia",
        "ra": "10:59:27.5287",
        "dec": "+40:25:49.8035",
        "pmra": -0.3168498990963689,
        "pmdec": 0.05518040036976814,
        "parallax": 72.00696109116399,
        "rv": 11.23812198638916,
        "frame": "icrs",
        "epoch": 2016.0,
        "equinox": 2000.0,
        "color": 0.7958617210388184,
        "color_name": "Gaia BP-RP",
    },
    "simbad": {
        "object": "HD 95128",
        "radec_src": "simbad",
        "plx_src": "simbad",
        "rv_src": "simbad",
        "ra": "10:59:27.9728",
        "dec": "+40:25:48.9206",
        "pmra": -0.31685,
        "pmdec": 0.05518,
        "parallax": 72.007,
        "rv": 11.142,
        "frame": "icrs",
        "epoch": 2000.0,
        "equinox": 2000.0,
        "color": None,
        "color_name": "",
    },
    "kpf-drp": {
        "object": "Gaia DR3 777254360337133312",
        "radec_src": "gaia",
        "plx_src": "gaia",
        "rv_src": "gaia",
        "ra": "10:59:27.5287",
        "dec": "+40:25:49.8035",
        "pmra": -0.3168498990963689,
        "pmdec": 0.05518040036976814,
        "parallax": 72.00696109116399,
        "rv": 11.23812198638916,
        "frame": "icrs",
        "epoch": 2016.0,
        "equinox": 2000.0,
        "color": 0.7958617210388184,
        "color_name": "Gaia BP-RP",
    },
}


class _FrozenAstroQuery:
    """AstroQuery stand-in that replays _CATALOG_CAPTURE instead of querying.

    Writes the frozen rows through the real module's own writer and header
    setter, so the L0 this hands back carries the same CATALOG_RECORD schema,
    presence flags and receipt entry a live run produces -- without the
    network. AstroQuery's own contract (queries, merge, schema) is tested in
    test_astro_query.py with Gaia and SIMBAD mocked.
    """

    def __init__(self, l0_obj, config=None):
        # Imported here, not at module scope: astro_query pulls in
        # astroquery/astropy (~2 s) and opens a connection at import.
        from kpfpipe.modules.astro_query import AstroQuery

        self._aq = AstroQuery(l0_obj, config)

    def perform(self):
        aq = self._aq
        for source, record in _CATALOG_CAPTURE.items():
            aq._write_catalog_record(source, dict(record))
        for keyword, value in aq._catalog_primary_cards().items():
            aq.l0_obj.set_keyword(keyword, value)
        aq.l0_obj.receipt_add_entry("astro_query", "", "PASS")
        return aq.l0_obj


# ---------------------------------------------------------------------------
# Science recipe integration (real L0 star data from tests/testdata/)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.requires_testdata
class TestScienceRecipe:
    @pytest.fixture(scope="class")
    def recipe_output(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("science_out")

        config = ConfigHandler(
            str(CONFIG_PATH),
            overrides={
                "DATA_DIRS": {
                    "KPF_DATA_INPUT": str(TESTDATA_DIR),
                    "KPF_MASTERS_OUTPUT": str(TESTDATA_DIR),
                    "KPF_SCIENCE_OUTPUT": str(tmp_path),
                }
            },
        )
        args = argparse.Namespace(obs_id=OBS_ID)

        recipe = _load_recipe()
        # The recipe module object is built fresh by _load_recipe() and dropped
        # with this fixture, so swapping the stage in place needs no patcher.
        # AstroQuery is the recipe's one live-network stage; _FrozenAstroQuery
        # replays a recording of it rather than querying. See _CATALOG_CAPTURE.
        recipe.AstroQuery = _FrozenAstroQuery
        recipe.main(config, args)

        out_path = kpf_filepath(OBS_ID, "L2", data_root=str(tmp_path))
        return out_path

    @pytest.fixture(scope="class")
    def l2(self, recipe_output):
        """The written L2, read once and shared read-only across the class."""
        return KPF2.from_fits(recipe_output)

    @pytest.fixture(scope="class")
    def l4(self, recipe_output):
        """The L4 the same recipe run wrote, read once for the class.

        Keyed off recipe_output so the recipe still runs exactly once; the
        data root is the L2 path's grandparent (as the QLP tests below do).
        """
        data_root = str(Path(recipe_output).parents[2])
        return KPF4.from_fits(kpf_filepath(OBS_ID, "L4", data_root=data_root))

    def test_output_file_exists(self, recipe_output):
        assert os.path.isfile(recipe_output), (
            f"Expected output not found: {recipe_output}"
        )

    def test_output_filename_format(self, recipe_output):
        assert os.path.basename(recipe_output) == "kpf_SL2_20240405T110833.fits"

    def test_output_is_valid_kpf2(self, l2):
        # isinstance() would only restate from_fits' return type; the level and
        # the extracted-spectrum extensions are what the L2 write must carry.
        assert l2.level == 2
        for ext in (
            "GREEN_SCI2_FLUX",
            "GREEN_SCI2_WAVE",
            "RED_SCI2_FLUX",
            "RED_SCI2_WAVE",
        ):
            assert np.asarray(l2.data[ext]).ndim == 2, f"{ext} is not a 2-D trace"

    def test_flux_positive(self, l2):
        assert np.nanmedian(l2.data["GREEN_SCI2_FLUX"]) > 0
        assert np.nanmedian(l2.data["RED_SCI2_FLUX"]) > 0

    def test_receipt_chain(self, l2):
        modules = l2.receipt["FUNCTION"].values
        assert "image_assembly" in modules
        assert "calibration_association" in modules
        assert "spectral_extraction" in modules
        assert "wavelength_calibration" in modules
        assert "barycentric_correction" in modules

    def test_barycorr_extensions_populated(self, l2):
        norder = DETECTOR["numorder"]
        for ext in ("BJD_TDB", "BARYCORR_KMS", "BARYCORR_Z"):
            arr = np.asarray(l2.data[ext])
            assert arr.shape == (norder,), f"{ext} shape {arr.shape} != ({norder},)"
            assert np.all(np.isfinite(arr)), f"{ext} has non-finite values"

    def test_per_ccd_barycorr_keywords(self, l2):
        homes = {
            "BJDGREEN": "BJD_TDB",
            "BJDRED": "BJD_TDB",
            "BVGREEN": "BARYCORR_KMS",
            "BVRED": "BARYCORR_KMS",
            "BZGREEN": "BARYCORR_Z",
            "BZRED": "BARYCORR_Z",
        }
        for key, ext in homes.items():
            hdr = l2.headers[ext]
            assert key in hdr, f"{key} missing from {ext}"
            assert np.isfinite(float(hdr.get(key))), f"{key} not finite"

    def test_calibration_headers_set(self, l2):
        # Master paths land in the RECEIPT table, ages on QUALITY_CONTROL.
        entry = l2.receipt_read_entry("calibration_association")
        qc = l2.headers["QUALITY_CONTROL"]
        # bias/dark use a full path + float AGE. Flat association is not wired
        # up until flat processing exists.
        for cal_type in ("bias", "dark"):
            assert entry[f"{cal_type}file"] is not None
            assert f"{cal_type.upper()}AGE" in qc
        assert entry["flatfile"] is None
        assert "FLATAGE" not in qc
        # thar follows the same convention; WLSAGE is in days.
        assert entry["wlsfile"].endswith("_master_thar_L2.fits")
        assert isinstance(qc.get("WLSAGE"), float)

    def test_catalog_record_reaches_the_science_cards(self, l2):
        """The merged catalog row is copied onto every science fiber's C*# cards.

        Expectations come from the capture itself, so this checks the overlay
        wiring -- each canonical column landing on its keyword, for all three
        science traces -- and stays indifferent to what the record contains.

        The pairs are spelled out rather than read from _CATALOG_CARD_BASES: a
        test that imported the mapping would agree with a renamed keyword
        instead of catching it.
        """
        prim = l2.headers["PRIMARY"]
        expected = _CATALOG_CAPTURE["kpf-drp"]
        # SCI1-3 are traces 2-4 and carry identical astrometry.
        for column, base in (
            ("object", "CID"),
            ("radec_src", "CSRC"),
            ("ra", "CRA"),
            ("dec", "CDEC"),
            ("pmra", "CPMR"),
            ("pmdec", "CPMD"),
            ("parallax", "CPLX"),
            ("rv", "CRV"),
            ("epoch", "CEPCH"),
            ("equinox", "CEQNX"),
            ("color", "CCLR"),
            ("color_name", "CCLRN"),
        ):
            for trace in (2, 3, 4):
                card = f"{base}{trace}"
                assert prim[card] == expected[column], card

    def test_l4_rv_products_are_deterministic(self, l4):
        """The same frames and masters must reproduce the same numbers.

        A reproducibility pin on the pipeline, not a claim about the target: it
        catches numerical drift and any loss of run-to-run determinism through
        the barycentric and CCF chain.

        These values are the only ones here coupled to what goes in: the truth
        frames, the masters, and _CATALOG_CAPTURE. If this goes red, confirm all
        three are unchanged before reading it as a pipeline regression --
        different input needs a deliberate re-pin, not a fix.
        """
        prim = l4.headers["PRIMARY"]
        assert prim["RV"] == pytest.approx(11.290367394940835, abs=1e-6)
        assert prim["BERV"] == pytest.approx(-18.507945234900816, abs=1e-6)
        assert prim["BJDTDB"] == pytest.approx(2460405.9689188506, abs=1e-9)

    def test_provenance_keywords_set(self, l2):
        # The EPRV DRPTAG and its WMKO counterpart DRPVERNO both sit on PRIMARY,
        # as do the rest of the provenance cards.
        prim = l2.headers["PRIMARY"]
        version = importlib.metadata.version("kpfpipe")
        assert prim.get("DRPTAG") == version
        assert prim.get("DRPVERNO") == version
        assert prim.get("PROGID") and prim.get("KOAID")
        # BarycentricCorrection is the last module to run before the L2 write.
        assert prim.get("DRPSTATU") == "Barycentric Correction module complete"

    @pytest.mark.parametrize(
        "ext, blue_end, red_end",
        [
            ("GREEN_SCI2_WAVE", (4450, 4470), (5990, 6010)),
            ("RED_SCI2_WAVE", (5980, 5995), (8690, 8710)),
        ],
    )
    def test_wave_arrays_populated(self, l2, ext, blue_end, red_end):
        # A single non-zero element would pass on a WLS copied from the wrong
        # fiber or chip. Every order must instead run blue->red and the chip's
        # coverage must land where the detector actually sees.
        wave = np.asarray(l2.data[ext])
        assert np.all(np.diff(wave, axis=1) > 0), f"{ext} is not strictly ascending"
        assert blue_end[0] < wave.min() < blue_end[1]
        assert red_end[0] < wave.max() < red_end[1]

    def test_l4_output_filename_format(self, recipe_output):
        path = kpf_filepath(OBS_ID, "L4", data_root=str(Path(recipe_output).parents[2]))
        assert os.path.isfile(path), f"Expected L4 output not found: {path}"
        assert os.path.basename(path) == "kpf_SL4_20240405T110833.fits"

    def test_l4_receipt_chain(self, l4):
        # The L2's receipt stops at barycentric_correction, so these two entries
        # are what distinguishes the final product from the intermediate one.
        modules = l4.receipt["FUNCTION"].values
        assert "cross_correlation" in modules
        assert "radial_velocity" in modules

    def test_l4_rv_products_present(self, l4):
        # Structural only, by decision: the RV extensions exist, name their
        # method, and are born in the dtypes the EPRV standard requires. No
        # radial-velocity VALUE is pinned anywhere in this suite.
        assert l4.headers["PRIMARY"].get("RVMETHOD") == "CCF"
        for fiber in ("SCI1", "SCI2", "SCI3"):
            assert_dtype(l4.data[f"{fiber}_CCF"], CCF, f"{fiber}_CCF")
            table = l4.data[f"{fiber}_RV"]
            assert set(table.colnames) >= {"RV", "RV_ERR", "BJD_TDB", "BERV"}
            for col in ("RV", "RV_ERR", "BJD_TDB", "BERV"):
                assert_dtype(np.asarray(table[col]), RV_FLOAT, f"{fiber}_RV[{col}]")

    def test_qlp_l0_pngs_exist(self, recipe_output):
        qlp_dir = Path(recipe_output).parents[2] / "QLP" / "20240405" / OBS_ID / "L0"
        assert (qlp_dir / f"{OBS_ID}_L0_stitched_image_green_zoomable.png").is_file()
        assert (qlp_dir / f"{OBS_ID}_L0_stitched_image_red_zoomable.png").is_file()

    def test_qlp_l1_pngs_exist(self, recipe_output):
        qlp_dir = Path(recipe_output).parents[2] / "QLP" / "20240405" / OBS_ID / "L1"
        assert (qlp_dir / f"{OBS_ID}_L1_image_green_zoomable.png").is_file()
        assert (qlp_dir / f"{OBS_ID}_L1_image_red_zoomable.png").is_file()

    def test_qlp_l2_pngs_exist(self, recipe_output):
        qlp_dir = Path(recipe_output).parents[2] / "QLP" / "20240405" / OBS_ID / "L2"
        assert (qlp_dir / f"{OBS_ID}_L2_snr_per_order_green_zoomable.png").is_file()
        assert (qlp_dir / f"{OBS_ID}_L2_snr_per_order_red_zoomable.png").is_file()

    def test_qlp_l4_pngs_exist(self, recipe_output):
        qlp_dir = Path(recipe_output).parents[2] / "QLP" / "20240405" / OBS_ID / "L4"
        assert (qlp_dir / f"{OBS_ID}_L4_ccf_grid_green_zoomable.png").is_file()
        assert (qlp_dir / f"{OBS_ID}_L4_ccf_grid_red_zoomable.png").is_file()


# ---------------------------------------------------------------------------
# Recipe wiring (no testdata: every collaborator stubbed, real main() called)
# ---------------------------------------------------------------------------


def _wire_science_recipe(tmp_path, monkeypatch):
    """Load the real recipe with every collaborator replaced by a recorder.

    Returns ``(recipe, config, args, record)``. ``record["calls"]`` holds one
    ``(stage, input_tag, args)`` tuple per collaborator in call order -- the
    list order IS the ordering assertion -- alongside the products written
    (``record["written"]``) and the object handed to the run summary.

    Stubs return tagged sentinels rather than real products, so each stage's
    input identifies which stage produced it.
    """
    recipe = _load_recipe()
    record = {"calls": [], "written": [], "summary": None}

    class Product:
        def __init__(self, tag):
            self.tag = tag

        def to_fits(self, path):
            record["written"].append((self.tag, path))

    def _tag(obj):
        return getattr(obj, "tag", obj)

    def _stage(name, produces):
        class StubStage:
            def __init__(self, obj, config=None, **kwargs):
                self._obj = obj

            def perform(self, *args, **kwargs):
                record["calls"].append((name, _tag(self._obj), args))
                return Product(produces)

        return StubStage

    def _checkpoint(name):
        class StubCheckpoint:
            def __init__(self, obj):
                self._obj = obj

            def run(self):
                record["calls"].append((name, _tag(self._obj), ()))

        return StubCheckpoint

    def _plot(name):
        class StubPlot:
            def __init__(self, obj, output_dir=None, obs_id=None):
                record["calls"].append((name, _tag(obj), (output_dir,)))

            def run(self, which):
                pass

        return StubPlot

    class StubKPF0:
        @staticmethod
        def from_fits(path, standardize=False):
            record["calls"].append(("from_fits", path, (standardize,)))
            return Product("l0")

    monkeypatch.setattr(recipe, "KPF0", StubKPF0)
    for name, produces in [
        ("AstroQuery", "l0"),
        ("ImageAssembly", "l1"),
        ("CalibrationAssociation", "l1"),
        ("ImageProcessing", "l1"),
        ("SpectralExtraction", "l2"),
        ("WavelengthCalibration", "l2"),
        ("BarycentricCorrection", "l2"),
        ("CrossCorrelation", "l4"),
        ("RadialVelocity", "l4"),
    ]:
        monkeypatch.setattr(recipe, name, _stage(name, produces))
    for name in ("CheckpointL0", "CheckpointL1", "CheckpointL2", "CheckpointL4"):
        monkeypatch.setattr(recipe, name, _checkpoint(name))
    for name in ("PlotL0", "PlotL1", "PlotL2", "PlotL4"):
        monkeypatch.setattr(recipe, name, _plot(name))

    def capture_summary(l4, elapsed):
        record["summary"] = _tag(l4)
        return ""

    monkeypatch.setattr(recipe, "science_run_summary", capture_summary)

    config = ConfigHandler(
        str(CONFIG_PATH),
        overrides={
            "DATA_DIRS": {
                "KPF_DATA_INPUT": str(tmp_path),
                "KPF_MASTERS_OUTPUT": str(tmp_path),
                "KPF_SCIENCE_OUTPUT": str(tmp_path),
            }
        },
    )
    return recipe, config, argparse.Namespace(obs_id=OBS_ID), record


class TestScienceRecipeWiring:
    """The stage hand-off chain, driven without testdata or FITS I/O."""

    @pytest.fixture
    def run(self, tmp_path, monkeypatch):
        recipe, config, args, record = _wire_science_recipe(tmp_path, monkeypatch)
        recipe.main(config, args)
        record["tmp_path"] = tmp_path
        return record

    def test_stages_run_in_order(self, run):
        # The load standardizes (next test), so every stage reads one PRIMARY,
        # the EPRV one. CheckpointL0 runs BEFORE ImageAssembly on purpose: QCL0
        # writes the L0 QC flags that to_kpf1 propagates into L1/L2/L4.
        assert [call[0] for call in run["calls"]] == [
            "from_fits",
            "AstroQuery",
            "PlotL0",
            "CheckpointL0",
            "ImageAssembly",
            "CalibrationAssociation",
            "ImageProcessing",
            "PlotL1",
            "CheckpointL1",
            "SpectralExtraction",
            "WavelengthCalibration",
            "BarycentricCorrection",
            "PlotL2",
            "CheckpointL2",
            "CrossCorrelation",
            "RadialVelocity",
            "PlotL4",
            "CheckpointL4",
        ]

    def test_the_load_standardizes(self, run):
        # The conversion is no longer a stage of its own; the recipe gets it by
        # loading with standardize=True, so nothing downstream sees a native PRIMARY.
        load = next(call for call in run["calls"] if call[0] == "from_fits")
        assert load[2] == (True,)

    def test_each_stage_receives_the_previous_product(self, run):
        # A mis-wired hand-off (SpectralExtraction fed the L0, RadialVelocity fed
        # the pre-CCF L2) is invisible to a pass/fail run on real data.
        inputs = {call[0]: call[1] for call in run["calls"]}
        assert inputs["ImageAssembly"] == "l0"
        assert inputs["SpectralExtraction"] == "l1"
        assert inputs["BarycentricCorrection"] == "l2"
        assert inputs["CrossCorrelation"] == "l2"
        assert inputs["RadialVelocity"] == "l4"

    def test_calibration_masters_requested(self, run):
        # Flat association is deliberately absent until flat processing exists.
        calls = {call[0]: call[2] for call in run["calls"]}
        assert calls["CalibrationAssociation"] == (["bias", "dark", "thar"],)

    def test_products_written_to_their_convention_paths(self, run):
        data_root = str(run["tmp_path"])
        assert run["written"] == [
            ("l2", kpf_filepath(OBS_ID, "L2", data_root=data_root)),
            ("l4", kpf_filepath(OBS_ID, "L4", data_root=data_root)),
        ]

    def test_summary_reads_the_l4(self, run):
        # The end-of-run verdict quotes the combined RV, so it must be handed the
        # L4 and not the L2 written a few lines earlier.
        assert run["summary"] == "l4"


# ---------------------------------------------------------------------------
# Science recipe error paths
# ---------------------------------------------------------------------------


class TestScienceRecipeErrors:
    def test_missing_l0_file_raises(self, tmp_path):
        config = ConfigHandler(
            str(CONFIG_PATH),
            overrides={
                "DATA_DIRS": {
                    "KPF_DATA_INPUT": str(tmp_path),
                    "KPF_MASTERS_OUTPUT": str(tmp_path),
                    "KPF_SCIENCE_OUTPUT": str(tmp_path),
                }
            },
        )
        args = argparse.Namespace(obs_id=OBS_ID)
        recipe = _load_recipe()
        # rvdata raises a bare IOError (an OSError) for a missing L0 file; match=
        # on the obs_id is what pins the failure to the L0 read, since other
        # OSErrors (dir creation, FITS writes) would also satisfy a bare type check.
        with pytest.raises(OSError, match=OBS_ID):
            recipe.main(config, args)

    def test_missing_obs_id_raises(self, tmp_path):
        config = ConfigHandler(
            str(CONFIG_PATH),
            overrides={
                "DATA_DIRS": {
                    "KPF_DATA_INPUT": str(tmp_path),
                    "KPF_MASTERS_OUTPUT": str(tmp_path),
                    "KPF_SCIENCE_OUTPUT": str(tmp_path),
                }
            },
        )
        args = argparse.Namespace(obs_id=None)
        recipe = _load_recipe()
        with pytest.raises(SystemExit, match="--obs_id is required"):
            recipe.main(config, args)


class _FakeL4:
    """Minimal stand-in for a finished KPF4: what science_run_summary reads."""

    def __init__(self, obs_id, headers, receipt):
        self.obs_id = obs_id
        self.headers = headers
        self.receipt = receipt

    # The parse under test is production code, so borrow it rather than mock it.
    receipt_read_entry = KPFDataModel.receipt_read_entry


class TestScienceSummary:
    """Unit tests for the science_run_summary() formatter."""

    def _l4(self):
        headers = {
            "RECEIPT": {"ORIGID": "KP.20240405.40113.57"},
            "PRIMARY": {"RV": 11.290158, "RVERR": 0.000156, "BJDTDB": 2460405.968919},
        }
        receipt = pd.DataFrame(
            [
                ("from_fits", "fn=/in/L0/20240405/KP.20240405.40113.57.fits, foo=None"),
                (
                    "calibration_association",
                    "biasfile=/m/20240405/KP.20240405.03637.74_master_bias_L1.fits, "
                    "darkfile=/m/20240405/KP.20240405.03637.74_master_dark_L1.fits, "
                    "flatfile=None, "
                    "wlsfile=/m/20240405/KP.20240405.63499.95_master_thar_L2.fits",
                ),
                (
                    "to_fits",
                    "out_filepath=/out/L2/20240405/kpf_SL2_20240405T110833.fits",
                ),
                (
                    "to_fits",
                    "out_filepath=/out/L4/20240405/kpf_SL4_20240405T110833.fits",
                ),
            ],
            columns=["FUNCTION", "ARGS"],
        )
        return _FakeL4("KP.20240405.40113.57", headers, receipt)

    def test_all_fields_from_l4(self):
        text = science_run_summary(self._l4(), 92.4)
        assert "run summary: KP.20240405.40113.57" in text
        # Input/output paths come from the RECEIPT table, shown as basenames.
        assert "inputs:   KP.20240405.40113.57.fits" in text
        assert "/in/" not in text
        assert (
            "outputs:  kpf_SL2_20240405T110833.fits  kpf_SL4_20240405T110833.fits"
            in text
        )
        # Masters from the calibration_association receipt entry.
        assert "bias=KP.20240405.03637.74_master_bias_L1.fits" in text
        assert "thar=KP.20240405.63499.95_master_thar_L2.fits" in text
        # RV km/s, error m/s (0.000156 km/s -> 0.156 m/s).
        assert "+11.29016 km/s  err 0.156 m/s  @ BJD_TDB 2460405.968919" in text
        assert "elapsed:  92.4 s" in text
        # Internal blank-line padding so the block stands out in the log.
        assert text.startswith("\n\n") and text.endswith("\n\n")

    def test_no_science_combine_and_no_receipt(self):
        # RV absent (or FITS UNDEFINED) -> n/a; no receipt table -> paths n/a.
        l4 = _FakeL4(
            "obs",
            {"RECEIPT": {}, "PRIMARY": {"RV": None}, "QUALITY_CONTROL": {}},
            None,
        )
        text = science_run_summary(l4, 1.0)
        assert "RV:       n/a (no science combine)" in text
        assert "inputs:   n/a" in text
        assert "outputs:  n/a" in text
        assert "bias=n/a" in text
