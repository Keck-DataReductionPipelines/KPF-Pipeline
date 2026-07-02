"""Tests for the pipeline logging setup (kpfpipe/utils/logger.py)."""

import logging
import re
import time
import warnings

import pytest

from kpfpipe.utils import logger as kpflog

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
