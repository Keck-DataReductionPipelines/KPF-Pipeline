"""
Unit tests for CalibrationAssociation.
"""

import pytest

from kpfpipe.modules.calibration_association import CalibrationAssociation

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class MockL1:
    def __init__(self, date_obs="2024-04-05T11:08:33"):
        self.headers = {"PRIMARY": {"DATE-OBS": date_obs}}
        self._receipt = []

    def receipt_add_entry(self, name, status):
        self._receipt.append((name, status))


def _make_module(tmp_path, date_obs="2024-04-05T11:08:33"):
    l1 = MockL1(date_obs)
    return CalibrationAssociation(l1, config={"KPF_DATA_INPUT": str(tmp_path)})


_LEVEL_BY_CAL_TYPE = {
    "bias": "L1",
    "dark": "L1",
    "flat": "L1",
    "thar": "L2",
}


def _stub_master(directory, obs_id, cal_type):
    """Create a zero-byte stub master file with the correct naming convention."""
    level = _LEVEL_BY_CAL_TYPE[cal_type]
    path = directory / f"{obs_id}_master_{cal_type}_{level}.fits"
    path.touch()
    return path


# ---------------------------------------------------------------------------
# TestFindMasterFiles
# ---------------------------------------------------------------------------


class TestFindMasterFiles:
    def test_returns_matching_files_within_window(self, tmp_path):
        d = tmp_path / "masters" / "20240405"
        d.mkdir(parents=True)
        _stub_master(d, "KP.20240405.03637.74", "bias")

        mod = _make_module(tmp_path)
        result = mod._find_master_files("bias", "2024-04-05T11:08:33")

        assert len(result) == 1
        assert result[0][1] == "20240405.03637.74"

    def test_searches_previous_day_by_default(self, tmp_path):
        # Default window is [-1, 0]; a file from the previous day should appear.
        d = tmp_path / "masters" / "20240404"
        d.mkdir(parents=True)
        _stub_master(d, "KP.20240404.79200.00", "bias")

        mod = _make_module(tmp_path)
        result = mod._find_master_files("bias", "2024-04-05T11:08:33")

        assert len(result) == 1
        assert "20240404" in result[0][1]

    def test_excludes_files_outside_window(self, tmp_path):
        # Two days back is outside the default [-1, 0] window.
        d = tmp_path / "masters" / "20240403"
        d.mkdir(parents=True)
        _stub_master(d, "KP.20240403.03637.74", "bias")

        mod = _make_module(tmp_path)
        result = mod._find_master_files("bias", "2024-04-05T11:08:33")

        assert result == []

    def test_masters_search_window_days_override_expands_range(self, tmp_path):
        d = tmp_path / "masters" / "20240403"
        d.mkdir(parents=True)
        _stub_master(d, "KP.20240403.03637.74", "bias")

        mod = _make_module(tmp_path)
        result = mod._find_master_files(
            "bias", "2024-04-05T11:08:33", masters_search_window_days=[-2, 0]
        )

        assert len(result) == 1

    def test_returns_empty_when_no_files(self, tmp_path):
        mod = _make_module(tmp_path)
        result = mod._find_master_files("bias", "2024-04-05T11:08:33")
        assert result == []

    def test_returns_sorted_by_timestamp(self, tmp_path):
        d = tmp_path / "masters" / "20240405"
        d.mkdir(parents=True)
        _stub_master(d, "KP.20240405.50000.00", "bias")
        _stub_master(d, "KP.20240405.03637.74", "bias")

        mod = _make_module(tmp_path)
        result = mod._find_master_files("bias", "2024-04-05T11:08:33")

        assert result[0][1] < result[1][1]

    def test_ignores_wrong_cal_type(self, tmp_path):
        d = tmp_path / "masters" / "20240405"
        d.mkdir(parents=True)
        _stub_master(d, "KP.20240405.03637.74", "dark")

        mod = _make_module(tmp_path)
        result = mod._find_master_files("bias", "2024-04-05T11:08:33")

        assert result == []


# ---------------------------------------------------------------------------
# TestSelectNearest
# ---------------------------------------------------------------------------


class TestSelectNearest:
    def test_returns_single_candidate(self, tmp_path):
        mod = _make_module(tmp_path)
        result = mod._select_nearest(
            "2024-04-05T11:08:33",
            [
                (
                    "/data/masters/20240405/KP.20240405.03637.74_master_bias_L1.fits",
                    "20240405.03637.74",
                )
            ],
        )
        assert (
            result == "/data/masters/20240405/KP.20240405.03637.74_master_bias_L1.fits"
        )

    def test_selects_nearest_of_two(self, tmp_path):
        mod = _make_module(tmp_path)
        # Science frame at 40113s (~11:08 UTC).
        # Candidate A at 03637s (~01:00 UTC) — delta ~7.1 hours.
        # Candidate B at 36000s (~10:00 UTC) — delta ~1.1 hours.
        result = mod._select_nearest(
            "2024-04-05T11:08:33",
            [
                (
                    "/masters/KP.20240405.03637.74_master_bias_L1.fits",
                    "20240405.03637.74",
                ),
                (
                    "/masters/KP.20240405.36000.00_master_bias_L1.fits",
                    "20240405.36000.00",
                ),
            ],
        )
        assert "KP.20240405.36000.00" in result

    def test_returns_none_for_empty_list(self, tmp_path):
        mod = _make_module(tmp_path)
        assert mod._select_nearest("2024-04-05T11:08:33", []) is None

    def test_prefers_same_day_over_previous_day(self, tmp_path):
        mod = _make_module(tmp_path)
        # Science at 2024-04-05 02:00 UTC (7200s).
        # Previous-day master at 23:00 HST = 23:00 local ≈ 23*3600 = 82800s
        # on 2024-04-04.
        # Same-day master at 00:30 UTC on 2024-04-05 (1800s).
        result = mod._select_nearest(
            "2024-04-05T02:00:00",
            [
                (
                    "/masters/KP.20240404.82800.00_master_bias_L1.fits",
                    "20240404.82800.00",
                ),
                (
                    "/masters/KP.20240405.01800.00_master_bias_L1.fits",
                    "20240405.01800.00",
                ),
            ],
        )
        assert "KP.20240405.01800.00" in result


# ---------------------------------------------------------------------------
# TestPerform
# ---------------------------------------------------------------------------


class TestPerform:
    @pytest.fixture
    def masters_dir(self, tmp_path):
        d = tmp_path / "masters" / "20240405"
        d.mkdir(parents=True)
        for cal_type in ("bias", "dark", "flat"):
            _stub_master(d, "KP.20240405.03637.74", cal_type)
        return tmp_path

    def test_returns_l1_obj(self, masters_dir):
        mod = _make_module(masters_dir)
        result = mod.perform(["bias"])
        assert result is mod.l1_obj

    def test_adds_receipt_entry(self, masters_dir):
        mod = _make_module(masters_dir)
        mod.perform(["bias"])
        assert ("calibration_association", "PASS") in mod.l1_obj._receipt

    def test_sets_biasfile_header(self, masters_dir):
        mod = _make_module(masters_dir)
        mod.perform(["bias"])
        assert (
            mod.l1_obj.headers["PRIMARY"]["BIASFILE"]
            == "KP.20240405.03637.74_master_bias_L1.fits"
        )

    def test_sets_biasdir_header(self, masters_dir):
        mod = _make_module(masters_dir)
        mod.perform(["bias"])
        expected_dir = str(masters_dir / "masters" / "20240405")
        assert mod.l1_obj.headers["PRIMARY"]["BIASDIR"] == expected_dir

    def test_sets_agebias_same_day(self, masters_dir):
        mod = _make_module(masters_dir)
        mod.perform(["bias"])
        assert mod.l1_obj.headers["PRIMARY"]["AGEBIAS"] == 0

    def test_sets_agebias_previous_day(self, tmp_path):
        d = tmp_path / "masters" / "20240404"
        d.mkdir(parents=True)
        _stub_master(d, "KP.20240404.79200.00", "bias")

        mod = _make_module(tmp_path)
        mod.perform(["bias"])
        assert mod.l1_obj.headers["PRIMARY"]["AGEBIAS"] == 1

    def test_sets_headers_for_dark_and_flat(self, masters_dir):
        mod = _make_module(masters_dir)
        mod.perform(["bias", "dark", "flat"])
        for prefix in ("BIAS", "DARK", "FLAT"):
            assert f"{prefix}FILE" in mod.l1_obj.headers["PRIMARY"]
            assert f"{prefix}DIR" in mod.l1_obj.headers["PRIMARY"]
            assert f"AGE{prefix}" in mod.l1_obj.headers["PRIMARY"]

    def test_sets_headers_for_thar(self, masters_dir):
        # Legacy WLS convention: WLSFILE holds the full path (no WLSDIR), and
        # AGEWLS is float days with sign = (master_dt - obs_dt). Master at
        # 2024-04-05 01:00:37 UTC vs obs at 2024-04-05 11:08:33 UTC gives
        # delta = -10h 07m 56s = -36476 s = -0.422176 days.
        d = masters_dir / "masters" / "20240405"
        _stub_master(d, "KP.20240405.03637.74", "thar")

        mod = _make_module(masters_dir)
        mod.perform(["bias", "thar"])
        h = mod.l1_obj.headers["PRIMARY"]
        assert h["WLSFILE"] == str(d / "KP.20240405.03637.74_master_thar_L2.fits")
        assert "WLSDIR" not in h
        assert isinstance(h["AGEWLS"], float)
        assert h["AGEWLS"] == pytest.approx(-0.422176, abs=1e-5)

    def test_raises_on_unknown_cal_type(self, masters_dir):
        mod = _make_module(masters_dir)
        with pytest.raises(ValueError, match="bogus"):
            mod.perform(["bogus"])

    def test_raises_when_no_master_found(self, tmp_path):
        mod = _make_module(tmp_path)
        with pytest.raises(FileNotFoundError, match="bias"):
            mod.perform(["bias"])

    def test_raises_on_first_missing_cal_type(self, masters_dir):
        # Only bias exists; dark should trigger the error.
        d = masters_dir / "masters" / "20240405"
        for f in d.glob("*_master_dark_L1.fits"):
            f.unlink()

        mod = _make_module(masters_dir)
        with pytest.raises(FileNotFoundError, match="dark"):
            mod.perform(["bias", "dark"])

    def test_masters_search_window_days_override(self, tmp_path):
        # Master is 2 days before the science frame; only found with wider window.
        d = tmp_path / "masters" / "20240403"
        d.mkdir(parents=True)
        _stub_master(d, "KP.20240403.03637.74", "bias")

        mod = _make_module(tmp_path)
        with pytest.raises(FileNotFoundError):
            mod.perform(["bias"])  # default window doesn't reach 2 days back

        mod2 = _make_module(tmp_path)
        mod2.perform(["bias"], masters_search_window_days=[-2, 0])  # should succeed
        assert "BIASFILE" in mod2.l1_obj.headers["PRIMARY"]
