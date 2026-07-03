"""Tests for the pipeline logging setup (kpfpipe/utils/logger.py)."""

import logging
import re
import time
import warnings

import pytest

from kpfpipe.utils import logger as kpflog
from kpfpipe.utils.config import ConfigHandler
from tools.cli import resolve_logging

_UT_FROZEN = time.struct_time((2026, 7, 2, 14, 3, 22, 2, 183, 0))


@pytest.fixture(autouse=True)
def _teardown():
    """Always tear down after each test.

    Critical for the captureWarnings bridge: pytest wraps every test in
    ``catch_warnings``, so ``captureWarnings(False)`` must run inside the
    same test context or ``logging._warnings_showwarning`` goes stale and a
    later ``captureWarnings(True)`` silently no-ops.
    """
    yield
    kpflog.teardown_logging()


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class TestGetLevel:
    def test_maps_names_any_case(self):
        assert kpflog.get_level("debug") == logging.DEBUG
        assert kpflog.get_level("INFO") == logging.INFO
        assert kpflog.get_level("Warning") == logging.WARNING

    def test_unknown_level_raises(self):
        with pytest.raises(ValueError, match="unknown log level"):
            kpflog.get_level("chatty")


class TestBuildLogPath:
    def test_layout_and_ut_components(self, tmp_path):
        path = kpflog.build_log_path(
            str(tmp_path), "science", "KP.20240923.33129.48", start_time=_UT_FROZEN
        )
        fn = "kpf_science_KP.20240923.33129.48_20260702T140322.log"
        assert path == str(tmp_path / "20260702" / fn)

    def test_empty_log_dir_raises(self):
        with pytest.raises(ValueError, match="log_dir"):
            kpflog.build_log_path("", "science", "x")


class TestSetupLogging:
    def test_creates_file_and_returns_path(self, tmp_path):
        path = kpflog.setup_logging(str(tmp_path), "science", "KP.1.2.3")
        datecode = time.strftime("%Y%m%d", time.gmtime())
        assert path.startswith(str(tmp_path / datecode))
        assert re.search(r"kpf_science_KP\.1\.2\.3_\d{8}T\d{6}\.log$", path)
        assert _read(path) == ""  # created, empty until a record arrives

    def test_record_format(self, tmp_path):
        path = kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        logging.getLogger("kpfpipe.x").info("hello world")
        line = _read(path).splitlines()[0]
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z "
            r"INFO\s+kpfpipe\.x: hello world",
            line,
        )

    def test_ut_timestamps(self, tmp_path):
        # Assert the converter rather than comparing wall clocks (TZ-robust).
        kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        handlers = logging.getLogger().handlers
        (handler,) = [h for h in handlers if h.name == "kpfpipe_file"]
        assert handler.formatter.converter is time.gmtime

    def test_level_filtering(self, tmp_path):
        path = kpflog.setup_logging(
            str(tmp_path), "science", "t", level="INFO", console=False
        )
        logging.getLogger("kpfpipe.x").debug("too quiet")
        logging.getLogger("kpfpipe.x").info("loud enough")
        text = _read(path)
        assert "too quiet" not in text
        assert "loud enough" in text

    def test_unique_filename_on_collision(self, tmp_path, monkeypatch):
        monkeypatch.setattr(kpflog.time, "gmtime", lambda *a: _UT_FROZEN)
        first = kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        second = kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        assert second == f"{first}.1"

    def test_repeated_setup_no_duplicate_handlers(self, tmp_path):
        root = logging.getLogger()
        before = len(root.handlers)
        first = kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        kpflog.setup_logging(str(tmp_path), "masters", "20240923", console=False)
        assert len(root.handlers) == before + 1  # one file handler, never two
        (handler,) = [h for h in root.handlers if h.name == "kpfpipe_file"]
        assert handler.baseFilename != first  # the survivor is the second file

    def test_console_handler_optional(self, tmp_path):
        root = logging.getLogger()
        kpflog.setup_logging(str(tmp_path), "science", "t", console=True)
        assert any(h.name == "kpfpipe_console" for h in root.handlers)
        kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        assert not any(h.name == "kpfpipe_console" for h in root.handlers)

    def test_third_party_pins(self, tmp_path):
        kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        assert logging.getLogger("matplotlib").level == logging.WARNING
        assert logging.getLogger("PIL").level == logging.WARNING

    def test_bad_level_raises_before_side_effects(self, tmp_path):
        with pytest.raises(ValueError, match="unknown log level"):
            kpflog.setup_logging(str(tmp_path), "science", "t", level="chatty")
        assert list(tmp_path.iterdir()) == []


class TestTeardown:
    def test_teardown_removes_handlers_and_restores_showwarning(self, tmp_path):
        root = logging.getLogger()
        before_handlers = list(root.handlers)
        before_showwarning = warnings.showwarning
        kpflog.setup_logging(str(tmp_path), "science", "t")
        kpflog.teardown_logging()
        assert root.handlers == before_handlers
        assert warnings.showwarning is before_showwarning

    def test_teardown_restores_root_level(self, tmp_path):
        root = logging.getLogger()
        before = root.level
        kpflog.setup_logging(str(tmp_path), "science", "t", level="DEBUG")
        kpflog.teardown_logging()
        assert root.level == before

    def test_teardown_safe_when_nothing_installed(self):
        kpflog.teardown_logging()  # must not raise


class TestIOChokepoints:
    """Every FITS write/read emits one INFO record (DRP-RUN-08)."""

    def test_to_fits_and_from_fits_log_records(self, tmp_path, caplog):
        from kpfpipe.data_models.level4 import KPF4

        path = str(tmp_path / "rt_l4.fits")
        with caplog.at_level(logging.INFO, logger="kpfpipe"):
            KPF4().to_fits(path)
            KPF4.from_fits(path)
        assert f"wrote KPF4 to {path}" in caplog.text
        assert f"reading KPF4 from {path}" in caplog.text


def _config(tmp_path, body):
    path = tmp_path / "config.toml"
    path.write_text(body)
    return path


class TestResolveLogging:
    _LOGGER_TOML = '[LOGGER]\nlog_directory = "/logs/"\nlog_level = "DEBUG"\n'

    def test_resolves_config_and_tokens(self, tmp_path):
        config = ConfigHandler(_config(tmp_path, self._LOGGER_TOML))
        params = resolve_logging(
            config, "/repo/recipes/kpf_drp_science.py", "KP.1.2.3", None
        )
        assert params == {
            "log_dir": "/logs/",
            "recipe_name": "science",
            "target": "KP.1.2.3",
            "level": "DEBUG",
            "console": True,
        }

    def test_datecode_target_and_defaults(self, tmp_path):
        config = ConfigHandler(_config(tmp_path, '[LOGGER]\nlog_directory = "/l/"\n'))
        params = resolve_logging(config, "kpf_drp_masters.py", None, "20240923")
        assert params["recipe_name"] == "masters"
        assert params["target"] == "20240923"
        assert params["level"] == "INFO"  # default when config omits log_level

    def test_no_target_falls_back_to_run(self, tmp_path):
        config = ConfigHandler(_config(tmp_path, '[LOGGER]\nlog_directory = "/l/"\n'))
        assert resolve_logging(config, "custom.py", None, None)["target"] == "run"

    def test_missing_log_directory_raises(self, tmp_path):
        config = ConfigHandler(_config(tmp_path, "[LOGGER]\n"))
        with pytest.raises(ValueError, match="no log directory configured"):
            resolve_logging(config, "kpf_drp_science.py", "KP.1.2.3", None)

    def test_cli_override_wins(self, tmp_path):
        # The CLI passes --log_dir/--log_level through ConfigHandler overrides.
        config = ConfigHandler(
            _config(tmp_path, self._LOGGER_TOML),
            overrides={"LOGGER": {"log_directory": "/elsewhere/", "log_level": "INFO"}},
        )
        params = resolve_logging(config, "kpf_drp_science.py", "KP.1.2.3", None)
        assert params["log_dir"] == "/elsewhere/"
        assert params["level"] == "INFO"


class TestWarningsBridge:
    @pytest.mark.filterwarnings("always")
    def test_capturewarnings_lands_in_file_with_source(self, tmp_path):
        path = kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        warnings.warn("bridged boom", stacklevel=1)  # 1: attribute to THIS line
        text = _read(path)
        assert "WARNING  py.warnings:" in text
        assert "bridged boom" in text
        assert "test_logger.py" in text  # file:lineno source identification

    @pytest.mark.filterwarnings("always")
    def test_pytest_warns_coexistence(self, tmp_path):
        # Inside pytest.warns the recorder wins: the assertion passes and the
        # record does NOT reach the log (captureWarnings only swaps
        # showwarning; pytest.warns uses catch_warnings(record=True)).
        path = kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        with pytest.warns(UserWarning, match="recorded boom"):
            warnings.warn("recorded boom", stacklevel=2)
        assert "recorded boom" not in _read(path)
