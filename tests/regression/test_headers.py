"""
Tests for HeaderParser (the value/comment normalizer) in data_models/headers.py.

HeaderConverter (native↔EPRV conversion/validation) is exercised end-to-end by
the to_kpf1/to_kpf2 tests in test_data_models_l{1,2,4}.py; here we pin the
read/write parsing contract that the rest of the pipeline relies on.
"""

from collections import OrderedDict

from astropy.io import fits

from kpfpipe.data_models.headers import HeaderParser
from kpfpipe.data_models.level1 import KPF1


class TestHeaderParserGet:
    """get() returns the scalar regardless of the header's storage form."""

    def test_fits_header(self):
        hdr = fits.Header()
        hdr["KEY"] = (1.5, "a comment")
        assert HeaderParser.get(hdr, "KEY") == 1.5

    def test_tuple_ordereddict(self):
        hdr = OrderedDict({"KEY": (1.5, "a comment")})
        assert HeaderParser.get(hdr, "KEY") == 1.5

    def test_bare_scalar(self):
        assert HeaderParser.get({"KEY": 1.5}, "KEY") == 1.5

    def test_missing_returns_default(self):
        assert HeaderParser.get({}, "KEY", "fallback") == "fallback"
        assert HeaderParser.get({}, "KEY") is None

    def test_present_none_is_not_default(self):
        # A stored None (e.g. a non-finite RV written as FITS UNDEFINED) is
        # returned as None; the default fires only on absence.
        assert HeaderParser.get({"KEY": None}, "KEY", "fallback") is None


class TestHeaderParserSet:
    """set() chooses the card form from whether a comment is supplied."""

    def test_with_comment_writes_tuple(self):
        hdr = {}
        HeaderParser.set(hdr, "KEY", 5, "the comment")
        assert hdr["KEY"] == (5, "the comment")

    def test_without_comment_writes_bare_value(self):
        hdr = {}
        HeaderParser.set(hdr, "KEY", 5)
        assert hdr["KEY"] == 5

    def test_set_then_get_on_fits_header(self):
        # A commented write works on any fits.Header extension; get() reads the
        # scalar back, comment retained on the card.
        hdr = fits.Header()
        HeaderParser.set(hdr, "KEY", 5, "the comment")
        assert HeaderParser.get(hdr, "KEY") == 5
        assert hdr.comments["KEY"] == "the comment"


class TestHeaderParserRoundTrip:
    """A commented PRIMARY write survives to_fits -> from_fits with its comment."""

    def test_primary_comment_round_trips(self, tmp_path):
        l1 = KPF1()
        l1.headers["PRIMARY"]["DATE-OBS"] = "2024-01-13T10:26:56"
        HeaderParser.set(l1.headers["PRIMARY"], "BIASAGE", -0.5, "[day] bias age")

        fn = str(tmp_path / "header_roundtrip_l1.fits")
        l1.to_fits(fn)

        prim = KPF1.from_fits(fn).headers["PRIMARY"]
        assert HeaderParser.get(prim, "BIASAGE") == -0.5
        assert prim.comments["BIASAGE"] == "[day] bias age"
