"""
Tests for KPF header storage and serialization (data_models/headers.py + base.py).

KPF stores every extension header as an ``astropy.io.fits.Header``, so reads use
``header.get(key)`` / ``header[key]`` and writes use ``header[key] = (value,
comment)`` natively. These tests pin two things the rest of the pipeline relies
on: (1) every header is a ``fits.Header`` from construction onward, and (2) a
commented PRIMARY card survives ``to_fits`` -> ``from_fits`` with its comment
intact -- the regression guard for the lossless-PRIMARY serialization in
``KPFDataModel._create_hdul`` / ``KPF2._create_hdul`` / ``KPF4._create_hdul``.

HeaderConverter (WMKO->EPRV conversion/validation) is exercised end-to-end by the
to_kpf1/to_kpf2 tests in test_data_models_l{1,2,4}.py.
"""

from astropy.io import fits

from kpfpipe.data_models.level1 import KPF1
from kpfpipe.data_models.level2 import KPF2


class TestHeaderStorage:
    """Every extension header is a fits.Header, for fresh and read-back objects."""

    def test_fresh_l1_headers_are_fits_headers(self):
        l1 = KPF1()
        assert isinstance(l1.headers["PRIMARY"], fits.Header)

    def test_fresh_l2_headers_are_fits_headers(self):
        kpf2 = KPF2()
        assert isinstance(kpf2.headers["PRIMARY"], fits.Header)
        # A non-PRIMARY extension header too (created via create_extension).
        assert isinstance(kpf2.headers["TRACE1_FLUX"], fits.Header)

    def test_tuple_write_sets_value_and_comment(self):
        # The documented write path: header[key] = (value, comment).
        l1 = KPF1()
        l1.headers["PRIMARY"]["BIASAGE"] = (-0.5, "[day] bias age")
        assert l1.headers["PRIMARY"]["BIASAGE"] == -0.5
        assert l1.headers["PRIMARY"].comments["BIASAGE"] == "[day] bias age"

    def test_get_returns_scalar_and_honors_default(self):
        l1 = KPF1()
        l1.headers["PRIMARY"]["DATE-OBS"] = ("2024-01-13T10:26:56", "obs start")
        assert l1.headers["PRIMARY"].get("DATE-OBS") == "2024-01-13T10:26:56"
        assert l1.headers["PRIMARY"].get("NOPE", "fallback") == "fallback"
        assert l1.headers["PRIMARY"].get("NOPE") is None


class TestPrimaryCommentRoundTrip:
    """A commented PRIMARY write survives to_fits -> from_fits with its comment.

    rvdata's base _create_hdul serializes PRIMARY by iterating ``.items()``, which
    drops a fits.Header's comments; KPF's _create_hdul overrides rebuild the
    PRIMARY HDU so the comments survive. These tests would fail without that fix.
    """

    def test_l1_primary_comment_round_trips(self, tmp_path):
        l1 = KPF1()
        l1.headers["PRIMARY"]["DATE-OBS"] = "2024-01-13T10:26:56"
        l1.headers["PRIMARY"]["BIASAGE"] = (-0.5, "[day] bias age")

        fn = str(tmp_path / "header_roundtrip_l1.fits")
        l1.to_fits(fn)

        prim = KPF1.from_fits(fn).headers["PRIMARY"]
        assert prim.get("BIASAGE") == -0.5
        assert prim.comments["BIASAGE"] == "[day] bias age"

    def test_l2_primary_comment_round_trips(self, tmp_path):
        kpf2 = KPF2()
        kpf2.headers["PRIMARY"]["CCFRV"] = (1.2345, "[km/s] Combined radial velocity")

        fn = str(tmp_path / "kpf_SL2_20240113T102656.fits")
        kpf2.to_fits(fn)

        prim = KPF2.from_fits(fn).headers["PRIMARY"]
        assert prim.get("CCFRV") == 1.2345
        assert prim.comments["CCFRV"] == "[km/s] Combined radial velocity"


class TestUndefinedRoundTrip:
    """A None value (non-finite RV) becomes a FITS UNDEFINED card, not a crash."""

    def test_none_value_round_trips_as_undefined(self, tmp_path):
        l1 = KPF1()
        l1.headers["PRIMARY"]["DATE-OBS"] = "2024-01-13T10:26:56"
        # radial_velocity writes None for a non-finite fit; astropy stores it as
        # an UNDEFINED card. The value must round-trip as a blank/undefined card
        # (read back as None), never as a finite number.
        l1.headers["PRIMARY"]["CCFRV"] = None

        fn = str(tmp_path / "header_undefined_l1.fits")
        l1.to_fits(fn)

        prim = KPF1.from_fits(fn).headers["PRIMARY"]
        assert "CCFRV" in prim
        val = prim["CCFRV"]
        assert val is None or isinstance(val, fits.card.Undefined)
