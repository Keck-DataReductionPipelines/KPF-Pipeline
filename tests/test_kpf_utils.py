"""
Tests for kpfpipe.utils.kpf timestamp conversion utilities.
"""

import pytest

from kpfpipe.utils.kpf import (
    utc_to_hst,
    hst_to_utc,
    kpf_timestamp_to_eprv_timestamp,
    eprv_timestamp_to_kpf_timestamp,
    get_obs_id,
    get_datecode,
    get_timestamp,
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
        # 17:00 HST = 03:00 UTC next day (61200 + 36000 = 97200 -> 97200 - 86400 = 10800)
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

    def test_basic_conversion(self):
        # 40113s = 11:08:33
        assert kpf_timestamp_to_eprv_timestamp("20240405.40113.57") == "20240405T110833"

    def test_midnight(self):
        assert kpf_timestamp_to_eprv_timestamp("20240405.00000.00") == "20240405T000000"

    def test_end_of_day(self):
        # 86399s = 23:59:59
        assert kpf_timestamp_to_eprv_timestamp("20240405.86399.00") == "20240405T235959"

    def test_one_hour(self):
        assert kpf_timestamp_to_eprv_timestamp("20240405.03600.00") == "20240405T010000"

    def test_frame_field_dropped(self):
        # Frame field should not appear in output
        result = kpf_timestamp_to_eprv_timestamp("20240405.40113.57")
        assert "57" not in result
        assert "." not in result


class TestEprvTimestampToKpf:

    def test_basic_conversion(self):
        # 11:08:33 = 11*3600 + 8*60 + 33 = 40113
        assert eprv_timestamp_to_kpf_timestamp("20240405T110833") == "20240405.40113.00"

    def test_midnight(self):
        assert eprv_timestamp_to_kpf_timestamp("20240405T000000") == "20240405.00000.00"

    def test_end_of_day(self):
        assert eprv_timestamp_to_kpf_timestamp("20240405T235959") == "20240405.86399.00"

    def test_one_hour(self):
        assert eprv_timestamp_to_kpf_timestamp("20240405T010000") == "20240405.03600.00"

    def test_frame_field_is_zero(self):
        assert eprv_timestamp_to_kpf_timestamp("20240405T110833").endswith(".00")

    def test_roundtrip(self):
        # Round-trip loses frame field (becomes .00)
        ts = "20240405.40113.57"
        assert eprv_timestamp_to_kpf_timestamp(kpf_timestamp_to_eprv_timestamp(ts)) == "20240405.40113.00"


class TestGetObsId:

    def test_extracts_from_bare_obs_id(self):
        assert get_obs_id("KP.20240405.40113.57") == "KP.20240405.40113.57"

    def test_extracts_from_path(self):
        assert get_obs_id("/data/L0/20240405/KP.20240405.40113.57.fits") == "KP.20240405.40113.57"

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
        assert get_timestamp("/data/L0/20240405/KP.20240405.40113.57.fits") == "20240405.40113.57"

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match="No KPF timestamp found"):
            get_timestamp("notimestamp.fits")
