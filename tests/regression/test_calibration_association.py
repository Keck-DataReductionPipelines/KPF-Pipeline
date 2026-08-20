"""Unit tests for CalibrationAssociation."""

import logging

import pytest
from astropy.io import fits

from kpfpipe.modules.calibration_association import CalibrationAssociation

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class MockL1:
    def __init__(self, date_obs="2024-04-05T11:08:33"):
        # Headers are fits.Header, mirroring the real KPF data models.
        primary = fits.Header()
        primary["DATE-OBS"] = date_obs
        self.obs_id = "KP.20240405.40113.57"
        self.headers = {
            "PRIMARY": primary,
            "INSTRUMENT_HEADER": fits.Header(),
            "RECEIPT": fits.Header(),
            "QUALITY_CONTROL": fits.Header(),
        }
        self._receipt = []

    def receipt_add_entry(self, name, args, status):
        self._receipt.append((name, args, status))

    def set_keyword(self, key, value):
        # Mirror the real routing: L1-headers.csv routes every {PREFIX}FILE
        # master path to RECEIPT, and that is all CalibrationAssociation writes.
        # Fail loud on anything else rather than inventing a PRIMARY fallback.
        if not key.endswith("FILE"):
            raise KeyError(f"{key!r} is not routed by this mock; extend it")
        self.headers["RECEIPT"][key] = value


def _make_module(tmp_path, date_obs="2024-04-05T11:08:33"):
    l1 = MockL1(date_obs)
    return CalibrationAssociation(l1, config={"KPF_MASTERS_OUTPUT": str(tmp_path)})


_LEVEL_BY_CAL_TYPE = {
    "bias": "L1",
    "dark": "L1",
    "flat": "L1",
    "thar": "L2",
}


def _stub_master(directory, obs_id, cal_type):
    """Zero-byte stub master following the production naming convention."""
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
        # Default window is [-1, 0] days.
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

    def test_warns_and_drops_master_with_unparseable_timestamp(self, caplog, tmp_path):
        d = tmp_path / "masters" / "20240405"
        d.mkdir(parents=True)
        _stub_master(d, "KP.20240405.03637.74", "bias")
        # Matches the *_master_bias_L1.fits glob but has no KPF timestamp.
        (d / "nostamp_master_bias_L1.fits").touch()

        mod = _make_module(tmp_path)
        with caplog.at_level(logging.WARNING):
            result = mod._find_master_files("bias", "2024-04-05T11:08:33")
        assert "unparseable timestamp" in caplog.text

        assert len(result) == 1
        assert result[0][0].endswith("KP.20240405.03637.74_master_bias_L1.fits")

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
        # Timestamps are seconds of day UTC: the science frame is at 40113s, so
        # 36000s is ~1.1 h away and 03637s ~7.1 h.
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

    def test_selects_a_master_taken_after_the_frame(self, tmp_path):
        # _select_nearest minimizes |dt|, so a master observed *after* the frame
        # wins when it is closer; the signed {PREFIX}AGE DiagL1 recomputes is
        # what makes that intentional. A past-only filter would be a plausible
        # "fix" that silently changes which master calibrates which frame.
        mod = _make_module(tmp_path)
        # Science at 40113s; candidates 3 h before (29313s) and 1 h after (43713s).
        result = mod._select_nearest(
            "2024-04-05T11:08:33",
            [
                (
                    "/masters/KP.20240405.29313.00_master_bias_L1.fits",
                    "20240405.29313.00",
                ),
                (
                    "/masters/KP.20240405.43713.00_master_bias_L1.fits",
                    "20240405.43713.00",
                ),
            ],
        )
        assert "KP.20240405.43713.00" in result

    def test_prefers_same_day_over_previous_day(self, tmp_path):
        mod = _make_module(tmp_path)
        # Timestamps are seconds of day UTC: science at 7200s on 04-05, with
        # candidates 3 h earlier (82800s on 04-04) and 1.5 h earlier (1800s).
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
        assert ("calibration_association", "", "PASS") in mod.l1_obj._receipt

    def test_sets_biasfile_header(self, masters_dir):
        # BIASFILE is the master's full path (no separate BIASDIR).
        mod = _make_module(masters_dir)
        mod.perform(["bias"])
        expected = str(
            masters_dir
            / "masters"
            / "20240405"
            / "KP.20240405.03637.74_master_bias_L1.fits"
        )
        assert mod.l1_obj.headers["RECEIPT"].get("BIASFILE") == expected

    def test_sets_headers_for_dark_and_flat(self, masters_dir):
        # {PREFIX}AGE is recomputed downstream by DiagL1 from the path this
        # module writes (covered in test_diagnostics.py), so it must be absent
        # here; likewise {PREFIX}DIR, since {PREFIX}FILE holds the full path.
        mod = _make_module(masters_dir)
        mod.perform(["bias", "dark", "flat"])
        for prefix in ("BIAS", "DARK", "FLAT"):
            assert f"{prefix}FILE" in mod.l1_obj.headers["RECEIPT"]
            assert f"{prefix}DIR" not in mod.l1_obj.headers["RECEIPT"]
            assert f"{prefix}AGE" not in mod.l1_obj.headers["QUALITY_CONTROL"]

    def test_sets_headers_for_thar(self, masters_dir):
        # WLSFILE holds the full path (no WLSDIR); WLSAGE comes from DiagL1.
        d = masters_dir / "masters" / "20240405"
        _stub_master(d, "KP.20240405.03637.74", "thar")

        mod = _make_module(masters_dir)
        mod.perform(["bias", "thar"])
        receipt = mod.l1_obj.headers["RECEIPT"]
        assert receipt.get("WLSFILE") == str(
            d / "KP.20240405.03637.74_master_thar_L2.fits"
        )
        assert "WLSDIR" not in receipt
        assert "WLSAGE" not in mod.l1_obj.headers["QUALITY_CONTROL"]

    def test_raises_on_unknown_cal_type(self, masters_dir):
        mod = _make_module(masters_dir)
        with pytest.raises(ValueError, match="bogus"):
            mod.perform(["bogus"])

    def test_raises_when_no_master_found(self, tmp_path):
        mod = _make_module(tmp_path)
        with pytest.raises(FileNotFoundError, match="bias"):
            mod.perform(["bias"])

    def test_raises_on_first_missing_cal_type(self, masters_dir):
        # dark is removed, so it is the type that fails.
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
        with pytest.raises(FileNotFoundError, match="No 'bias' master found"):
            mod.perform(["bias"])  # default window doesn't reach 2 days back

        mod2 = _make_module(tmp_path)
        mod2.perform(["bias"], masters_search_window_days=[-2, 0])
        assert "BIASFILE" in mod2.l1_obj.headers["RECEIPT"]


# ---------------------------------------------------------------------------
# Fail-loudly paths (construction + cal_type validation)
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_bad_config_type_raises(self):
        with pytest.raises(
            TypeError, match="config must be None, dict, or ConfigHandler"
        ):
            CalibrationAssociation(MockL1(), config="not-a-config")

    def test_unsupported_cal_type_raises(self, tmp_path):
        # cal_type is validated before date_obs is used, so the value is irrelevant.
        mod = _make_module(tmp_path)
        with pytest.raises(ValueError, match="unsupported cal_type"):
            mod._find_master_files("nonsense", None)
