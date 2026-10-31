"""Tests for kpfpipe.utils.kpf: KPF timestamp/obs_id conversion utilities."""

from datetime import datetime

import pytest

from kpfpipe.utils.kpf import (
    get_datecode,
    get_obs_id,
    get_timestamp,
    hst_to_utc,
    instrument_era,
    is_datecode,
    is_obs_id,
    is_timestamp,
    kpf_timestamp_to_datetime,
    kpf_timestamp_to_eprv_timestamp,
    utc_to_hst,
)


class TestUtcToHst:
    def test_midday_no_rollover(self):
        # 12:00 UTC = 02:00 HST same day (43200 - 36000 = 7200)
        assert utc_to_hst("20240405.43200.00") == "20240405.07200.00"

    def test_rollover_to_previous_day(self):
        # 03:00 UTC = 17:00 HST previous day (10800 - 36000 < 0)
        assert utc_to_hst("20240405.10800.00") == "20240404.61200.00"

    def test_frame_str_preserved(self):
        assert utc_to_hst("20240405.43200.71").endswith(".71")

    def test_exact_offset_boundary(self):
        # 10:00 UTC = 00:00 HST (36000 - 36000 = 0)
        assert utc_to_hst("20240405.36000.00") == "20240405.00000.00"


class TestHstToUtc:
    def test_midday_no_rollover(self):
        # 02:00 HST = 12:00 UTC same day (7200 + 36000 = 43200)
        assert hst_to_utc("20240405.07200.00") == "20240405.43200.00"

    def test_rollover_to_next_day(self):
        # 17:00 HST = 03:00 UTC next day
        # (61200 + 36000 = 97200 -> 97200 - 86400 = 10800)
        assert hst_to_utc("20240404.61200.00") == "20240405.10800.00"

    def test_frame_str_preserved(self):
        assert hst_to_utc("20240405.07200.71").endswith(".71")

    def test_exact_offset_boundary(self):
        # 00:00 HST = 10:00 UTC (0 + 36000 = 36000)
        assert hst_to_utc("20240405.00000.00") == "20240405.36000.00"

    def test_roundtrip(self):
        ts = "20240405.40113.57"
        assert hst_to_utc(utc_to_hst(ts)) == ts


class TestKpfTimestampToEprv:
    @pytest.mark.parametrize(
        "timestamp, expected",
        [
            (
                "20240405.40113.57",
                "20240405T110833",
            ),  # 40113 s = 11:08:33; frame dropped
            ("20240405.00000.00", "20240405T000000"),  # midnight
            ("20240405.86399.00", "20240405T235959"),  # 86399 s = 23:59:59
            ("20240405.03600.00", "20240405T010000"),  # one hour
        ],
    )
    def test_conversion(self, timestamp, expected):
        assert kpf_timestamp_to_eprv_timestamp(timestamp) == expected


class TestGetObsId:
    def test_extracts_from_bare_obs_id(self):
        assert get_obs_id("KP.20240405.40113.57") == "KP.20240405.40113.57"

    def test_extracts_from_path(self):
        assert (
            get_obs_id("/data/L0/20240405/KP.20240405.40113.57.fits")
            == "KP.20240405.40113.57"
        )

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match="No obs_id found"):
            get_obs_id("not_an_obs_id.fits")


class TestGetDatecode:
    def test_extracts_from_obs_id(self):
        assert get_datecode("KP.20240405.40113.57") == "20240405"

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match="Cannot extract datecode"):
            get_datecode("not_an_obs_id")


class TestGetTimestamp:
    def test_extracts_from_obs_id(self):
        assert get_timestamp("KP.20240405.40113.57") == "20240405.40113.57"

    def test_extracts_from_path(self):
        assert (
            get_timestamp("/data/L0/20240405/KP.20240405.40113.57.fits")
            == "20240405.40113.57"
        )

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match="No KPF timestamp found"):
            get_timestamp("notimestamp.fits")


# ------------------------------------------------------------------
# Validation: predicates reject semantically invalid input
# ------------------------------------------------------------------


class TestPredicates:
    @pytest.mark.parametrize(
        "predicate, value, expected",
        [
            (is_obs_id, "KP.20240405.40113.57", True),
            (is_obs_id, "KP.20249999.40113.57", False),  # month 99: not a date
            (is_obs_id, "KP.20240405.99999.57", False),  # 99999 s > 86399
            (is_obs_id, "20240405.40113.57", False),  # wrong format
            (is_obs_id, None, False),
            (is_obs_id, 12345, False),
            (is_datecode, "20240405", True),
            (is_datecode, "20249999", False),  # month 99: not a date
            (is_datecode, "2024-04-05", False),  # wrong format
            (is_datecode, None, False),
            (is_datecode, 20240405, False),
            (is_timestamp, "20240405.40113.57", True),
            (is_timestamp, "20249999.40113.57", False),  # month 99: not a date
            (is_timestamp, "20240405.99999.57", False),  # 99999 s > 86399
            (is_timestamp, "20240405T110833", False),  # wrong format
            (is_timestamp, None, False),
        ],
    )
    def test_classifies_input(self, predicate, value, expected):
        assert predicate(value) is expected


# ------------------------------------------------------------------
# Validation: extractors and converters raise on malformed or semantically
# invalid timestamps rather than producing wrong output (a day rollover from
# seconds > 86399, or an hour >= 24)
# ------------------------------------------------------------------


_INVALID = "Invalid KPF timestamp"
_NOT_STRING = "must be a string"


class TestTimestampValidation:
    @pytest.mark.parametrize(
        "func, value, match",
        [
            (get_obs_id, "KP.20249999.40113.57.fits", _INVALID),
            (get_obs_id, "KP.20240405.99999.57.fits", _INVALID),
            (get_obs_id, None, _NOT_STRING),
            (get_datecode, "KP.20249999.40113.57", _INVALID),
            (get_datecode, "KP.20240405.99999.57", _INVALID),
            (get_datecode, None, _NOT_STRING),
            (get_timestamp, "KP.20249999.40113.57.fits", _INVALID),
            (get_timestamp, "KP.20240405.99999.57.fits", _INVALID),
            (get_timestamp, None, _NOT_STRING),
            (kpf_timestamp_to_datetime, "20240405.99999.57", _INVALID),
            (kpf_timestamp_to_datetime, "20249999.40113.57", _INVALID),
            (kpf_timestamp_to_datetime, "not-a-timestamp", _INVALID),
            (utc_to_hst, "20240405.99999.57", _INVALID),
            (utc_to_hst, "20240405T110833", _INVALID),
            (hst_to_utc, "20240405.99999.57", _INVALID),
            (hst_to_utc, "not-a-timestamp", _INVALID),
            (kpf_timestamp_to_eprv_timestamp, "20240405.99999.57", _INVALID),
            (kpf_timestamp_to_eprv_timestamp, "20249999.40113.57", _INVALID),
        ],
    )
    def test_invalid_input_raises(self, func, value, match):
        with pytest.raises(ValueError, match=match):
            func(value)


class TestInstrumentEra:
    # Era 2.0 starts 2024-02-23 12:00:01 UT; 1.5 ends at 12:00:00.
    @pytest.mark.parametrize(
        "obs_time, era",
        [
            (datetime(2024, 2, 23, 12, 0, 0), "1.5"),
            (datetime(2024, 2, 23, 12, 0, 1), "2.0"),
            (datetime(2024, 1, 1), "1.0"),
        ],
    )
    def test_era_at_boundary(self, obs_time, era):
        assert instrument_era(obs_time) == era

    def test_uncovered_time_raises(self):
        with pytest.raises(ValueError, match="No KPF instrument era covers"):
            instrument_era(datetime(2999, 1, 1))

    @pytest.mark.xfail(
        strict=True,
        raises=ValueError,
        reason="prod bug: kpf.py:171 / level0.py:256 -- reference eras leave a 1 s "
        "gap between rows, so a sub-second timestamp inside it has no era",
    )
    def test_sub_second_time_at_a_boundary_has_an_era(self):
        # Frame timestamps carry sub-second precision (DATE-OBS is to the ms).
        assert instrument_era(datetime(2024, 2, 23, 12, 0, 0, 500000)) in {"1.5", "2.0"}
