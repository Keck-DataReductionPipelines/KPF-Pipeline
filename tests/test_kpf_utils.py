"""
Tests for kpfpipe.utils.kpf timestamp conversion utilities.
"""

import pytest

from kpfpipe.utils.kpf import (
    utc_to_hst,
    hst_to_utc,
    kpf_timestamp_to_eprv_timestamp,
    eprv_timestamp_to_kpf_timestamp,
    kpf_timestamp_to_datetime,
    get_obs_id,
    get_datecode,
    get_timestamp,
    get_seconds_since_j2000,
    is_obs_id,
    is_datecode,
    is_timestamp,
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
        assert (
            eprv_timestamp_to_kpf_timestamp(kpf_timestamp_to_eprv_timestamp(ts))
            == "20240405.40113.00"
        )


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


# ---------------------------------------------------------------------------
# Validation: predicates reject semantically invalid input
# ---------------------------------------------------------------------------


class TestIsObsId:
    def test_valid_obs_id(self):
        assert is_obs_id("KP.20240405.40113.57") is True

    def test_rejects_bad_date(self):
        # Format matches but month 99 is not a real date.
        assert is_obs_id("KP.20249999.40113.57") is False

    def test_rejects_seconds_out_of_range(self):
        # SSSSS = 99999 > 86399.
        assert is_obs_id("KP.20240405.99999.57") is False

    def test_rejects_wrong_format(self):
        assert is_obs_id("20240405.40113.57") is False

    def test_rejects_non_string(self):
        assert is_obs_id(None) is False
        assert is_obs_id(12345) is False


class TestIsDatecode:
    def test_valid_datecode(self):
        assert is_datecode("20240405") is True

    def test_rejects_bad_date(self):
        # Format matches but month 99 is not a real date.
        assert is_datecode("20249999") is False

    def test_rejects_wrong_format(self):
        assert is_datecode("2024-04-05") is False

    def test_rejects_non_string(self):
        assert is_datecode(None) is False
        assert is_datecode(20240405) is False


class TestIsTimestamp:
    def test_valid_timestamp(self):
        assert is_timestamp("20240405.40113.57") is True

    def test_rejects_bad_date(self):
        assert is_timestamp("20249999.40113.57") is False

    def test_rejects_seconds_out_of_range(self):
        assert is_timestamp("20240405.99999.57") is False

    def test_rejects_wrong_format(self):
        assert is_timestamp("20240405T110833") is False

    def test_rejects_non_string(self):
        assert is_timestamp(None) is False


# ---------------------------------------------------------------------------
# Validation: extractors raise on semantically invalid embedded timestamps
# ---------------------------------------------------------------------------


class TestGetObsIdValidation:
    def test_raises_on_invalid_date(self):
        with pytest.raises(ValueError, match="Invalid date"):
            get_obs_id("KP.20249999.40113.57.fits")

    def test_raises_on_seconds_out_of_range(self):
        with pytest.raises(ValueError, match="seconds-past-midnight"):
            get_obs_id("KP.20240405.99999.57.fits")

    def test_raises_on_non_string(self):
        with pytest.raises(ValueError, match="must be a string"):
            get_obs_id(None)


class TestGetDatecodeValidation:
    def test_raises_on_invalid_date(self):
        with pytest.raises(ValueError, match="Invalid date"):
            get_datecode("KP.20249999.40113.57")

    def test_raises_on_seconds_out_of_range(self):
        with pytest.raises(ValueError, match="seconds-past-midnight"):
            get_datecode("KP.20240405.99999.57")

    def test_raises_on_non_string(self):
        with pytest.raises(ValueError, match="must be a string"):
            get_datecode(None)


class TestGetTimestampValidation:
    def test_raises_on_invalid_date(self):
        with pytest.raises(ValueError, match="Invalid date"):
            get_timestamp("KP.20249999.40113.57.fits")

    def test_raises_on_seconds_out_of_range(self):
        # The path contains a syntactically-matching but invalid timestamp.
        with pytest.raises(ValueError, match="seconds-past-midnight"):
            get_timestamp("KP.20240405.99999.57.fits")

    def test_raises_on_non_string(self):
        with pytest.raises(ValueError, match="must be a string"):
            get_timestamp(None)


# ---------------------------------------------------------------------------
# Validation: converters reject malformed input rather than silently
# producing wrong output (the day-rollover and hh>=24 slips).
# ---------------------------------------------------------------------------


class TestKpfTimestampToDatetimeValidation:
    def test_raises_on_seconds_out_of_range(self):
        # Previously: silently rolled over into the next day.
        with pytest.raises(ValueError, match="seconds-past-midnight"):
            kpf_timestamp_to_datetime("20240405.99999.57")

    def test_raises_on_invalid_date(self):
        with pytest.raises(ValueError, match="Invalid date"):
            kpf_timestamp_to_datetime("20249999.40113.57")

    def test_raises_on_wrong_format(self):
        with pytest.raises(ValueError, match="Invalid KPF timestamp format"):
            kpf_timestamp_to_datetime("not-a-timestamp")


class TestUtcToHstValidation:
    def test_raises_on_seconds_out_of_range(self):
        with pytest.raises(ValueError, match="seconds-past-midnight"):
            utc_to_hst("20240405.99999.57")

    def test_raises_on_wrong_format(self):
        with pytest.raises(ValueError, match="Invalid KPF timestamp format"):
            utc_to_hst("20240405T110833")


class TestHstToUtcValidation:
    def test_raises_on_seconds_out_of_range(self):
        with pytest.raises(ValueError, match="seconds-past-midnight"):
            hst_to_utc("20240405.99999.57")

    def test_raises_on_wrong_format(self):
        with pytest.raises(ValueError, match="Invalid KPF timestamp format"):
            hst_to_utc("not-a-timestamp")


class TestKpfTimestampToEprvValidation:
    def test_raises_on_seconds_out_of_range(self):
        # Previously: produced 'YYYYMMDDT273739' with hh=27.
        with pytest.raises(ValueError, match="seconds-past-midnight"):
            kpf_timestamp_to_eprv_timestamp("20240405.99999.57")

    def test_raises_on_invalid_date(self):
        with pytest.raises(ValueError, match="Invalid date"):
            kpf_timestamp_to_eprv_timestamp("20249999.40113.57")


class TestEprvTimestampToKpfValidation:
    def test_raises_on_non_T_separator(self):
        # Previously: silently accepted any char at position 8.
        with pytest.raises(ValueError, match="Invalid EPRV timestamp format"):
            eprv_timestamp_to_kpf_timestamp("20240405A110833")

    def test_raises_on_hours_out_of_range(self):
        # Previously: emitted an invalid KPF timestamp going out.
        with pytest.raises(ValueError, match="time-of-day"):
            eprv_timestamp_to_kpf_timestamp("20240405T256059")

    def test_raises_on_minutes_out_of_range(self):
        with pytest.raises(ValueError, match="time-of-day"):
            eprv_timestamp_to_kpf_timestamp("20240405T126059")

    def test_raises_on_seconds_out_of_range(self):
        with pytest.raises(ValueError, match="time-of-day"):
            eprv_timestamp_to_kpf_timestamp("20240405T120060")

    def test_raises_on_invalid_date(self):
        with pytest.raises(ValueError, match="Invalid date"):
            eprv_timestamp_to_kpf_timestamp("20249999T110833")

    def test_raises_on_short_input(self):
        with pytest.raises(ValueError, match="Invalid EPRV timestamp format"):
            eprv_timestamp_to_kpf_timestamp("short")


class TestGetSecondsSinceJ2000:
    def test_basic(self):
        # J2000.0 itself: 2000-01-01 12:00 UTC = '20000101.43200.00'
        assert get_seconds_since_j2000("20000101.43200.00") == 0

    def test_monotonic_across_year_boundary(self):
        # Dec 31 23:59:00 -> Jan 1 00:00:00 should differ by 60s exactly.
        end = get_seconds_since_j2000("20231231.86340.00")
        start_next_year = get_seconds_since_j2000("20240101.00000.00")
        assert start_next_year - end == 60

    def test_raises_on_invalid_timestamp(self):
        with pytest.raises(ValueError, match="seconds-past-midnight"):
            get_seconds_since_j2000("KP.20240405.99999.57.fits")

    def test_raises_when_no_timestamp_found(self):
        with pytest.raises(ValueError, match="No KPF timestamp found"):
            get_seconds_since_j2000("notimestamp.fits")
