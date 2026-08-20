"""Tests for the KPF1 (assembled FFI / L1) data model, the L0->L1 transform, and
the KPFMasterL1 calibration product. Synthetic FITS fixtures throughout -- no real
KPF data needed.
"""

import importlib.metadata
import logging
import re

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from kpfpipe.data_models.level0 import KPF0
from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.masters import KPFMasterL1
from kpfpipe.data_models.masters.base import KPFMasterModel
from kpfpipe.modules.astro_query import AstroQuery
from kpfpipe.utils.astro import compute_redshift

from ._catalog import SOURCES, catalog_record_table

# synthetic_l0_file, synthetic_l0_minimal, synthetic_l1_file fixtures live in
# tests/conftest.py

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_masters_l1_file(tmp_path_factory):
    """Minimal synthetic Masters L1 FITS file (module-scoped: consumers only read
    it via from_fits)."""
    rng = np.random.default_rng(20240113)
    fn = str(
        tmp_path_factory.mktemp("ml1") / "KP.20240113.23249.10_master_bias_L1.fits"
    )

    primary = fits.PrimaryHDU()
    primary.header["INSTRUME"] = "KPF"
    primary.header["DATE-OBS"] = "2024-01-13T10:26:56"
    primary.header["DATALVL"] = "ML1"

    def img(name):
        return fits.ImageHDU(data=rng.random((32, 32)).astype(np.float32), name=name)

    def mask(name):
        return fits.ImageHDU(data=np.ones((32, 32), dtype=np.uint8), name=name)

    hdul = fits.HDUList(
        [
            primary,
            img("GREEN_IMG"),
            img("GREEN_SNR"),
            mask("GREEN_MASK"),
            img("RED_IMG"),
            img("RED_SNR"),
            mask("RED_MASK"),
        ]
    )
    hdul.writeto(fn, overwrite=True)
    hdul.close()

    return fn


class TestKPF1:
    def test_from_fits(self, synthetic_l1_file):
        l1 = KPF1.from_fits(synthetic_l1_file)
        assert l1.level == 1
        assert "GREEN_CCD" in l1.extensions
        assert "GREEN_VAR" in l1.extensions
        assert "RED_CCD" in l1.extensions
        assert "RED_VAR" in l1.extensions
        assert l1.data["GREEN_CCD"].shape == (32, 32)

    def test_required_extensions_created(self):
        l1 = KPF1()
        assert "PRIMARY" in l1.extensions
        assert "GREEN_CCD" in l1.extensions
        assert "GREEN_VAR" in l1.extensions
        assert "RED_CCD" in l1.extensions
        assert "RED_VAR" in l1.extensions
        assert "RECEIPT" in l1.extensions

    def test_round_trip(self, synthetic_l1_file, tmp_path):
        l1 = KPF1.from_fits(synthetic_l1_file)
        original_green = l1.data["GREEN_CCD"].copy()

        out_fn = str(tmp_path / "kpf_L1_20240113T102656.fits")
        l1.to_fits(out_fn)

        l1_reread = KPF1.from_fits(out_fn)
        np.testing.assert_array_almost_equal(
            l1_reread.data["GREEN_CCD"], original_green, decimal=4
        )

    def test_receipt_tracking(self, synthetic_l1_file, tmp_path):
        l1 = KPF1.from_fits(synthetic_l1_file)
        assert len(l1.receipt) >= 1

        out_fn = str(tmp_path / "kpf_L1_20240113T102656.fits")
        l1.to_fits(out_fn)
        assert "to_fits" in l1.receipt["FUNCTION"].values

    def test_receipt_survives_roundtrip(self, synthetic_l1_file, tmp_path):
        # rvdata's _create_hdul serializes self.data["RECEIPT"], so
        # KPFDataModel._create_hdul syncs self.receipt into it before writing;
        # without that the in-memory history is silently lost on write.
        l1 = KPF1.from_fits(synthetic_l1_file)
        l1.receipt_add_entry("image_assembly", "", "PASS")
        out_fn = str(tmp_path / "kpf_L1_20240113T102656.fits")
        l1.to_fits(out_fn)

        modules = KPF1.from_fits(out_fn).receipt["FUNCTION"].values
        assert "image_assembly" in modules
        assert "to_fits" in modules

    def test_generate_filename(self, synthetic_l1_file):
        l1 = KPF1.from_fits(synthetic_l1_file)
        l1.obs_id = "KP.20240113.37616.10"  # 37616 s of day = 10:26:56 UT
        assert l1.generate_standard_filename() == "kpf_L1_20240113T102656.fits"

    def test_datalvl_header(self, synthetic_l1_file, tmp_path):
        l1 = KPF1.from_fits(synthetic_l1_file)
        out_fn = str(tmp_path / "kpf_L1_20240113T102656.fits")
        l1.to_fits(out_fn)

        with fits.open(out_fn) as hdul:
            assert hdul["PRIMARY"].header["DATALVL"] == "L1"

    def test_raises_on_unknown_extension(self, tmp_path):
        fn = str(tmp_path / "unknown_ext.fits")
        primary = fits.PrimaryHDU()
        primary.header["DATE-OBS"] = "2024-01-13T00:00:00"
        weird = fits.ImageHDU(data=np.zeros((4, 4)))
        weird.name = "WEIRD_EXTENSION"
        hdul = fits.HDUList([primary, weird])
        hdul.writeto(fn, overwrite=True)
        hdul.close()

        with pytest.raises(
            ValueError, match="Non-standard extension 'WEIRD_EXTENSION'"
        ):
            KPF1.from_fits(fn)

    def test_to_fits_warns_on_nonconforming_name_but_writes(
        self, caplog, synthetic_l1_file, tmp_path
    ):
        # The filename advisory is warn-only, as at L0: the write still proceeds.
        l1 = KPF1.from_fits(synthetic_l1_file)
        out = str(tmp_path / "not_kpf_convention.fits")
        with caplog.at_level(logging.WARNING):
            out_path = l1.to_fits(out)
        assert "does not follow the KPF L1 naming" in caplog.text
        assert out_path == out


class TestL1PrimarySeed:
    """KPF1.__init__ seeds the EPRV Required PRIMARY skeleton from the registry,
    mirroring rvdata's RV2.__init__ (which KPF2/KPF4 inherit but KPF1 cannot, as
    L1 is not an EPRV level). This makes KWRDPRL1 a meaningful presence check."""

    @staticmethod
    def _required_l1_primary():
        # The EPRV L2 PRIMARY set is tagged Level 1 (KPF requires it from L1).
        reg = KPF1.keyword_registry
        return {k for k, lvl in reg.required["PRIMARY"].items() if lvl <= 1}

    def test_fresh_kpf1_carries_eprv_skeleton(self):
        l1 = KPF1()
        present = set(l1.headers["PRIMARY"])
        assert self._required_l1_primary() <= present

    def test_seed_is_typed_with_comments(self):
        l1 = KPF1()
        prim = l1.headers["PRIMARY"]
        # Boolean/UInt EPRV datatypes parse to real Python types, not strings.
        assert prim["ISSOLAR"] is False
        # The comment comes from the EPRV description (registry), not the caller.
        assert prim.comments["INSTRUME"] == "Instrument name"

    def test_datalvl_corrected_to_l1(self):
        # The seed defaults DATALVL (EPRV Required) to "UNKNOWN"; __init__ fixes it.
        assert KPF1().headers["PRIMARY"]["DATALVL"] == "L1"

    def test_compliance_tags_from_rvdata_pin(self):
        # EPRVTAG/VOCLASS default to "UNKNOWN" in the EPRV CSV; _DEFAULT_OVERRIDES
        # derives them from the pinned rv-data-standard release. EPRVTAG is the
        # version ("v0.4.0"); VOCLASS encodes the release month
        # ("EPRVSTANDARD<YYYY.MM>").
        prim = KPF1().headers["PRIMARY"]
        version = importlib.metadata.version("rv-data-standard")
        assert prim["EPRVTAG"] == f"v{version}"
        assert re.fullmatch(r"EPRVSTANDARD\d{4}\.\d{2}", prim["VOCLASS"])

    def test_seed_matches_registry_lookup(self):
        # The 40 seeded keys are exactly the registry's eprv_primary_seed.
        assert (
            set(KPF1.keyword_registry.eprv_primary_seed) == self._required_l1_primary()
        )

    def test_converted_l1_has_all_required(self, synthetic_l0_file):
        # A converted L1 carries every EPRV-Required PRIMARY keyword, which is
        # what makes QCL1's KWRDPRL1 presence check meaningful.
        l1 = KPF0.from_fits(synthetic_l0_file).to_kpf1()
        assert self._required_l1_primary() <= set(l1.headers["PRIMARY"])

    def test_native_overlay_is_typed(self, synthetic_l0_file):
        # _map_header coerces native/header_map-default values to their EPRV
        # DataType and preserves the seeded comment.
        prim = KPF0.from_fits(synthetic_l0_file).to_kpf1().headers["PRIMARY"]
        assert prim["NUMTRACE"] == 5 and isinstance(prim["NUMTRACE"], int)
        assert isinstance(prim["OBSALT"], float)
        assert prim["ISSOLAR"] is False
        assert prim.comments["NUMTRACE"]  # comment survived the typed overlay

    def test_master_l1_not_seeded(self):
        # KPFMasterL1 bypasses KPF1.__init__ (its __init__ chains straight to
        # KPFDataModel), so masters stay out of EPRV scope -- no EPRV science
        # skeleton. Masters carry their own minimal PRIMARY: only DATALVL ("ML1").
        prim = set(KPFMasterL1().headers["PRIMARY"])
        assert prim == {"DATALVL"}
        assert not (self._required_l1_primary() & prim) - {"DATALVL"}


class TestHeaderMapFiberRealignment:
    """rvdata's header_map numbers per-trace keywords CAL-first (1=CAL..5=SKY);
    _load_header_map realigns them to KPF's SKY-first numbering (1=SKY..5=CAL) by
    swapping index 1<->5 for the fiber-indexed families. Keywords that also end in
    1/5 but are not fiber-indexed (EXSNR wavelength bands) must be left alone."""

    @staticmethod
    def _source(standard):
        # (INSTRUMENT, DEFAULT) for a STANDARD key in the sanitized header_map, or
        # (None, None) if the row was dropped.
        hm = KPF1.keyword_registry.header_map
        row = hm[hm["STANDARD"].astype(str).str.strip() == standard]
        if row.empty:
            return None, None
        return str(row.iloc[0]["INSTRUMENT"]).strip(), str(
            row.iloc[0]["DEFAULT"]
        ).strip()

    def test_trace_native_cards_swapped_to_sky_first(self):
        # SKY is trace 1, CAL is trace 5 (KPF), so their native OBJ cards land there.
        assert self._source("TRACE1")[0] == "SKY-OBJ"
        assert self._source("TRACE5")[0] == "CAL-OBJ"

    def test_catalog_block_swapped(self):
        # The CAL last-source card moves to index 5; the "sky" catalog defaults move
        # from index 5 to index 1 (SKY).
        assert self._source("CLSRC5")[0] == "CAL-OBJ"
        assert self._source("CLSRC1")[0] != "CAL-OBJ"
        assert self._source("CSRC1")[1] == "sky"
        assert self._source("CID1")[1] == "sky"

    def test_exposure_meter_snr_not_swapped(self):
        # Guard: EXSNR ends in 1/5 (wavelength band 452/852nm), not a fiber index,
        # so it must be untouched -- catches an over-broad swap.
        assert self._source("EXSNR1")[0] == "SNRSC452"
        assert self._source("EXSNR5")[0] == "SNRSC852"

    def test_sci_catalog_source_cells_blanked(self):
        # The SCI-fiber (2,3,4) catalog C*# cards come from to_kpf1's CATALOG_RECORD
        # overlay, so their raw header_map source cells are blanked; SKY(1)/CAL(5) not.
        hm = KPF1.keyword_registry.header_map
        for base in ("CID", "CSRC", "CRA", "CDEC", "CPMR", "CRV", "CZ", "CLSRC"):
            for i in (2, 3, 4):
                r = hm[hm["STANDARD"].astype(str).str.strip() == f"{base}{i}"].iloc[0]
                assert pd.isna(r["INSTRUMENT"]) and pd.isna(r["DEFAULT"])
        assert self._source("CSRC1")[1] == "sky"  # SKY default intact
        assert self._source("CRV5")[1] == "0"  # CAL default intact
        assert self._source("CLSRC5")[0] == "CAL-OBJ"  # CAL native card intact


class TestToKpf1:
    def test_to_kpf1_creates_kpf1(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert l1.level == 1
        assert isinstance(l1, KPF1)

    def test_to_kpf1_copies_primary_header(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert l1.headers["PRIMARY"]["INSTRUME"] == "KPF"
        assert l1.headers["PRIMARY"]["DATE-OBS"] == "2024-01-13T10:26:56"
        assert l1.headers["PRIMARY"]["OBJECT"] == "HD_10700"

    # Canonical CATALOG_RECORD row (EPRV C*# format) overlaid onto the SCI cards.
    _KPF_DRP = {
        "object": "Gaia123",
        "radec_src": "gaia",
        "plx_src": "gaia",
        "rv_src": "gaia",
        "ra": "12:00:00.0000",
        "dec": "+40:00:00.000",
        "pmra": 0.5,
        "pmdec": -0.3,
        "parallax": 50.0,
        "rv": 10.0,
        "frame": "icrs",
        "epoch": 2016.0,
        "equinox": 2000.0,
        "color": 1.23,
        "color_name": "Gaia BP-RP",
    }

    @staticmethod
    def _l0_with_catalog(record):
        # A science KPF0 carrying a canonical 'kpf-drp' CATALOG_RECORD row, no network.
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = "Object"
        AstroQuery(l0)._write_catalog_record("kpf-drp", record)
        return l0

    def test_catalog_overlay_populates_sci_cards(self):
        # Direct copy of the canonical row onto every SCI fiber (2,3,4), no conversion.
        p = self._l0_with_catalog(dict(self._KPF_DRP)).to_kpf1().headers["PRIMARY"]
        for i in (2, 3, 4):
            assert p[f"CID{i}"] == "Gaia123"
            assert p[f"CSRC{i}"] == "gaia"
            assert p[f"CRA{i}"] == "12:00:00.0000"
            assert p[f"CDEC{i}"] == "+40:00:00.000"
            assert p[f"CPMR{i}"] == 0.5
            assert p[f"CPMD{i}"] == -0.3
            assert p[f"CPLX{i}"] == 50.0
            assert p[f"CRV{i}"] == 10.0
            assert p[f"CZ{i}"] == pytest.approx(compute_redshift(10.0 * u.km / u.s))
            assert p[f"CEPCH{i}"] == 2016.0
            assert p[f"CEQNX{i}"] == 2000.0
            assert p[f"CCLR{i}"] == 1.23
            assert p[f"CCLRN{i}"] == "Gaia BP-RP"

    def test_catalog_overlay_skips_missing_optional(self):
        # parallax/rv absent from the canonical row -> those cards stay blank; the
        # coherent position block is still written.
        record = {
            **self._KPF_DRP,
            "parallax": None,
            "rv": None,
            "color": None,
            "color_name": None,
        }
        p = self._l0_with_catalog(record).to_kpf1().headers["PRIMARY"]
        assert not p.get("CPLX2")
        assert not p.get("CRV2")
        assert not p.get("CZ2")  # z derives from rv, so it is blank when rv is absent
        assert not p.get("CCLR2")
        assert not p.get("CCLRN2")
        assert p["CRA2"] == "12:00:00.0000"

    def test_science_frame_without_catalog_warns_and_leaves_blank(self, caplog):
        # An empty CATALOG_RECORD on a science frame means AstroQuery never ran:
        # to_kpf1 succeeds with blank SCI C*# cards but warns about the downstream fail.
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = "Object"
        with caplog.at_level(logging.WARNING):
            p = l0.to_kpf1().headers["PRIMARY"]
        assert "CATALOG_RECORD is empty" in caplog.text
        assert "will fail downstream" in caplog.text
        for kw in ("CRA2", "CID2", "CSRC2", "CPMR2"):
            assert not p.get(kw)

    def test_calibration_frame_without_catalog_leaves_sci_cards_blank(self):
        # A calibration frame carries no target astrometry, so an empty
        # CATALOG_RECORD is expected: to_kpf1 succeeds with blank SCI C*# cards.
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = "Bias"
        p = l0.to_kpf1().headers["PRIMARY"]
        for kw in ("CRA2", "CID2", "CSRC2", "CPMR2"):
            assert not p.get(kw)

    def test_catalog_overlay_warns_on_mixed_sources(self, caplog):
        # rv from a different catalog than the position -> WARNING at PRIMARY commit.
        record = {**self._KPF_DRP, "rv_src": "simbad"}
        with caplog.at_level(logging.WARNING):
            self._l0_with_catalog(record).to_kpf1()
        assert "mixed sources" in caplog.text

    def test_catalog_overlay_single_source_no_warning(self, caplog):
        # All provenance labels agree (gaia) -> no mixed-source warning.
        with caplog.at_level(logging.WARNING):
            self._l0_with_catalog(dict(self._KPF_DRP)).to_kpf1()
        assert "mixed sources" not in caplog.text

    def test_to_kpf1_converts_native_to_eprv(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        prim = l1.headers["PRIMARY"]
        assert prim.get("OBSTYPE") == "Object"  # IMTYPE -> OBSTYPE
        assert prim.get("EXPTIME") == 300.0  # ELAPSED -> EXPTIME
        assert prim.get("OBSERVER") == "Smith"  # GROBSERV -> OBSERVER
        # Raw native names must not remain on the EPRV PRIMARY.
        assert "IMTYPE" not in prim
        assert "ELAPSED" not in prim
        assert "GROBSERV" not in prim

    def test_to_kpf1_preserves_raw_header_in_instrument_header(self, synthetic_l0_file):
        # INSTRUMENT_HEADER is a verbatim copy of the raw L0 PRIMARY.
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert "INSTRUMENT_HEADER" in l1.extensions
        inst = l1.headers["INSTRUMENT_HEADER"]
        assert inst["IMTYPE"] == "Object"
        assert inst["ELAPSED"] == 300.0
        assert inst["GROBSERV"] == "Smith"
        assert inst["INSTRUME"] == "KPF"
        # Pipeline-stamped DRP provenance lives on RECEIPT, never on the raw
        # PRIMARY snapshot, so INSTRUMENT_HEADER stays pure instrument metadata.
        assert "DRPVERNO" not in inst
        assert "DRPSTATU" not in inst

    def test_to_kpf1_filters_non_registry_headermap_keys(self, tmp_path):
        # _map_header emits only registered keywords, so header_map's non-standard
        # STANDARD keys (here PARANG <- PARANTEL) are dropped rather than leaked
        # onto the EPRV PRIMARY; the raw value survives in INSTRUMENT_HEADER.
        fn = str(tmp_path / "KP.20240113.00009.00.fits")
        p = fits.PrimaryHDU()
        p.header["INSTRUME"] = "KPF"
        p.header["OFNAME"] = "KP.20240113.00009.00.fits"
        p.header["PROGNAME"] = "K123"
        p.header["PARANTEL"] = 108.03  # header_map maps PARANTEL -> non-standard PARANG
        fits.HDUList([p]).writeto(fn)
        l1 = KPF0.from_fits(fn).to_kpf1()
        assert "PARANG" not in l1.headers["PRIMARY"]
        assert l1.headers["INSTRUMENT_HEADER"]["PARANTEL"] == 108.03

    def test_to_kpf1_fixes_value_bugs(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        prim = l1.headers["PRIMARY"]
        assert prim.get("NUMORDER") == 67  # 35 green + 32 red, not 65
        # JD_UTC is the full Julian Date of DATE-OBS (not a raw MJD).
        assert prim.get("JD_UTC") == pytest.approx(2460322.93537, abs=1e-3)
        version = importlib.metadata.version("kpfpipe")
        assert prim.get("DRPTAG") == version  # EPRV version keyword stays on PRIMARY
        # DRPVERNO lives on RECEIPT, not PRIMARY.
        assert prim.get("DRPVERNO") is None
        assert l1.headers["RECEIPT"].get("DRPVERNO") == version

    def test_map_header_is_pure_tabular_except_jd_utc(self, synthetic_l0_file):
        # Those values are correct on the L1 PRIMARY, but _map_header itself does
        # not special-case NUMORDER/DRPTAG/DATALVL (they ride the seed / model
        # level); JD_UTC is the one transform it performs.
        out = KPF0.from_fits(synthetic_l0_file)._map_header()
        assert "NUMORDER" not in out  # seeded (registry _DEFAULT_OVERRIDES)
        assert "DRPTAG" not in out  # seeded (registry _DEFAULT_OVERRIDES)
        assert "DATALVL" not in out  # set by KPF1.__init__ (model level)
        assert "JD_UTC" in out  # the one per-frame transform kept in _map_header

    def test_to_kpf1_forwards_program_ids(self, synthetic_l0_file):
        # PROGID/KOAID are stamped to the L0 RECEIPT at read and forwarded to the
        # L1 RECEIPT, never onto PRIMARY.
        l1 = KPF0.from_fits(synthetic_l0_file).to_kpf1()
        receipt = l1.headers["RECEIPT"]
        assert receipt.get("PROGID") == "K123"
        assert receipt.get("KOAID") == "KP.20240113.23249.10.fits"
        assert "PROGID" not in l1.headers["PRIMARY"]

    def test_to_kpf1_forwards_drpstatus(self, synthetic_l0_file):
        # The to_kpf1 receipt is denylisted, so the ingest-time DRPSTATU survives
        # until the first real module runs.
        receipt = KPF0.from_fits(synthetic_l0_file).to_kpf1().headers["RECEIPT"]
        assert receipt.get("DRPSTATU") == "File ingested into KPF-DRP"

    def test_to_kpf1_copies_passthrough_extensions(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        # CA_HK and TELEMETRY were in the synthetic file
        assert "CA_HK" in l1.extensions
        assert "TELEMETRY" in l1.extensions
        np.testing.assert_array_equal(l1.data["CA_HK"], l0.data["CA_HK"])

    def test_to_kpf1_skips_missing_extensions(self, synthetic_l0_minimal):
        l0 = KPF0.from_fits(synthetic_l0_minimal)
        l1 = l0.to_kpf1()
        assert "CA_HK" not in l1.extensions
        assert "TELEMETRY" not in l1.extensions

    def test_to_kpf1_leaves_ccd_empty(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert "GREEN_CCD" in l1.extensions
        assert "RED_CCD" in l1.extensions
        # Extensions exist but ImageAssembly has not populated them yet.
        assert len(l1.data["GREEN_CCD"]) == 0
        assert len(l1.data["RED_CCD"]) == 0

    def test_to_kpf1_carries_receipt(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert len(l1.receipt) >= 2  # from_fits + to_kpf1
        assert "to_kpf1" in l1.receipt["FUNCTION"].values

    def test_to_kpf1_copies_obs_id(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert l1.obs_id == "KP.20240113.23249.10"

    def test_to_kpf1_drops_amp_extensions(self, synthetic_l0_file):
        l0 = KPF0.from_fits(synthetic_l0_file)
        l1 = l0.to_kpf1()
        assert "GREEN_AMP1" not in l1.extensions
        assert "GREEN_AMP2" not in l1.extensions
        assert "RED_AMP1" not in l1.extensions


class TestCatalogRecordPassthrough:
    """AstroQuery's CATALOG_RECORD rows and presence flags ride L0 -> L1 unchanged.

    The L1 -> L2 and L2 -> L4 hops have the same class in test_data_models_l{2,4}.py.
    """

    @staticmethod
    def _l0_with_catalog_table(rv=-16.6):
        l0 = KPF0()
        l0.headers["PRIMARY"]["IMTYPE"] = "Object"
        l0.set_data("CATALOG_RECORD", catalog_record_table(rv=rv))
        l0.set_keyword("GAIACR", 1)
        return l0

    def test_rows_and_flags_reach_l1(self):
        # Beyond the C*# overlay, L1 keeps every source row -- not just the merged
        # one -- so the astrometry stays auditable.
        l1 = self._l0_with_catalog_table().to_kpf1()
        assert [str(s) for s in l1.data["CATALOG_RECORD"]["source"]] == list(SOURCES)
        assert l1.headers["CATALOG_RECORD"]["GAIACR"] == 1

    def test_catalog_record_roundtrip(self, tmp_path):
        # CATALOG_RECORD is registered in L1-extensions.csv, so it reads back at
        # all (an unlisted extension raises), and a missing rv comes back NaN
        # rather than masked.
        fn = str(tmp_path / "kpf_L1_20240405T000000.fits")
        self._l0_with_catalog_table(rv=None).to_kpf1().to_fits(fn)
        back = KPF1.from_fits(fn)
        assert [str(s) for s in back.data["CATALOG_RECORD"]["source"]] == list(SOURCES)
        assert back.headers["CATALOG_RECORD"]["GAIACR"] == 1
        rv = back.data["CATALOG_RECORD"][0]["rv"]
        assert rv is not np.ma.masked and np.isnan(rv)


class TestDrpStatus:
    """DRPSTATU advances to '<Module Name> module complete' via the
    receipt_add_entry override; data-model conversion/IO receipts are denylisted
    so it names the last real science/masters stage."""

    def test_module_receipt_updates_status(self, synthetic_l0_file):
        l1 = KPF0.from_fits(synthetic_l0_file).to_kpf1()
        l1.receipt_add_entry("image_assembly", "", "PASS")
        status = l1.headers["RECEIPT"].get("DRPSTATU")
        assert status == "Image Assembly module complete"

    def test_master_receipt_updates_status(self, synthetic_l0_file):
        l1 = KPF0.from_fits(synthetic_l0_file).to_kpf1()
        l1.receipt_add_entry("master_bias", "", "PASS")
        status = l1.headers["RECEIPT"].get("DRPSTATU")
        assert status == "Master Bias module complete"

    def test_internal_receipts_do_not_change_status(self, synthetic_l0_file):
        l1 = KPF0.from_fits(synthetic_l0_file).to_kpf1()
        l1.receipt_add_entry("radial_velocity", "", "PASS")
        for internal in ("to_kpf2", "to_kpf4", "to_fits", "from_fits"):
            l1.receipt_add_entry(internal, "", "PASS")
        status = l1.headers["RECEIPT"].get("DRPSTATU")
        assert status == "Radial Velocity module complete"


class TestKPFMasterL1:
    def test_required_extensions_created(self):
        m = KPFMasterL1()
        for ext in [
            "PRIMARY",
            "GREEN_IMG",
            "GREEN_SNR",
            "GREEN_MASK",
            "RED_IMG",
            "RED_SNR",
            "RED_MASK",
            "RECEIPT",
        ]:
            assert ext in m.extensions

    def test_no_science_extensions(self):
        m = KPFMasterL1()
        for ext in ["GREEN_CCD", "GREEN_VAR", "RED_CCD", "RED_VAR", "CA_HK"]:
            assert ext not in m.extensions

    def test_inherits_from_kpf1(self):
        m = KPFMasterL1()
        assert isinstance(m, KPF1)

    def test_inherits_from_kpf_master_model(self):
        m = KPFMasterL1()
        assert isinstance(m, KPFMasterModel)

    def test_class_attributes(self):
        assert KPFMasterL1._DATALVL == "ML1"

    def test_from_fits(self, synthetic_masters_l1_file):
        m = KPFMasterL1.from_fits(synthetic_masters_l1_file)
        assert "GREEN_IMG" in m.extensions
        assert "GREEN_SNR" in m.extensions
        assert "GREEN_MASK" in m.extensions
        assert "RED_IMG" in m.extensions
        assert m.data["GREEN_IMG"].shape == (32, 32)

    def test_round_trip(self, synthetic_masters_l1_file, tmp_path):
        m = KPFMasterL1.from_fits(synthetic_masters_l1_file)
        original = m.data["GREEN_IMG"].copy()

        out_fn = str(tmp_path / "KP.20240113.23249.10_master_bias_L1.fits")
        m.to_fits(out_fn)

        m2 = KPFMasterL1.from_fits(out_fn)
        np.testing.assert_array_almost_equal(m2.data["GREEN_IMG"], original, decimal=4)

    def test_datalvl_header_in_fits(self, synthetic_masters_l1_file, tmp_path):
        m = KPFMasterL1.from_fits(synthetic_masters_l1_file)
        out_fn = str(tmp_path / "KP.20240113.23249.10_master_bias_L1.fits")
        m.to_fits(out_fn)
        with fits.open(out_fn) as hdul:
            assert hdul["PRIMARY"].header["DATALVL"] == "ML1"

    def test_generate_filename(self):
        m = KPFMasterL1()
        m.set_input_files(
            ["KP.20240113.23249.10.fits", "KP.20240113.23310.00.fits"], "bias"
        )
        assert (
            m.generate_standard_filename() == "KP.20240113.23249.10_master_bias_L1.fits"
        )

    def test_generate_filename_requires_inputs(self):
        # A master can never produce a non-compliant name: with no recorded
        # inputs / type, generate_standard_filename raises rather than falling
        # back to a KOAID-less name.
        with pytest.raises(ValueError):
            KPFMasterL1().generate_standard_filename()

    def test_no_warning_on_known_extensions(self, caplog, synthetic_masters_l1_file):
        with caplog.at_level(logging.WARNING):
            KPFMasterL1.from_fits(synthetic_masters_l1_file)
        assert "Non-standard extension" not in caplog.text

    def test_set_input_files(self):
        m = KPFMasterL1()
        files = ["/data/a.fits", "/data/b.fits", "/data/c.fits"]
        m.set_input_files(files, "bias")
        assert m.data["INPUT_FILES"]["FILENAME"].tolist() == files
        assert m.headers["PRIMARY"]["MASTYPE"] == "bias"
