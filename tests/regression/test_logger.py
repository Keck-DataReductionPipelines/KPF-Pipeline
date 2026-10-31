"""Tests for the pipeline logging setup (kpfpipe/utils/logger.py)."""

import logging
import re
import sys
import time
import warnings

import pytest

from kpfpipe.utils import logger as kpflog

_UT_FROZEN = time.struct_time((2026, 7, 2, 14, 3, 22, 2, 183, 0))
_RUN_ID = "run_20260702T140322"


@pytest.fixture(autouse=True)
def _teardown():
    """Tear down after each test.

    pytest wraps every test in ``catch_warnings``, so ``captureWarnings(False)``
    must run inside the same test context or ``logging._warnings_showwarning``
    goes stale and a later ``captureWarnings(True)`` silently no-ops.
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
        path = kpflog.log_filename(
            str(tmp_path), "science", "KP.20240923.33129.48", start_time=_UT_FROZEN
        )
        fn = "kpf_science_KP.20240923.33129.48_20260702T140322.log"
        assert path == str(tmp_path / fn)

    def test_empty_log_dir_raises(self):
        with pytest.raises(ValueError, match="log_dir"):
            kpflog.log_filename("", "science", "x")


class TestSetupLogging:
    def test_creates_file_and_returns_path(self, tmp_path, monkeypatch):
        # Freeze the clock: recomputing the stamp after the call races UT
        # midnight, a once-a-day flake in a CI-scheduled suite.
        monkeypatch.setattr(kpflog.time, "gmtime", lambda *a: _UT_FROZEN)
        path = kpflog.setup_logging(str(tmp_path), "science", "KP.1.2.3", _RUN_ID)
        assert path.startswith(str(tmp_path))
        assert re.search(r"kpf_science_KP\.1\.2\.3_\d{8}T\d{6}\.log$", path)
        assert _read(path) == ""  # created, empty until a record arrives

    def test_record_format(self, tmp_path):
        path = kpflog.setup_logging(
            str(tmp_path), "science", "t", _RUN_ID, console=False
        )
        logging.getLogger("kpfpipe.x").info("hello world")
        line = _read(path).splitlines()[0]
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z "
            r"INFO\s+kpfpipe\.x: hello world",
            line,
        )

    def test_ut_timestamps(self, tmp_path):
        # Assert the converter rather than comparing wall clocks (TZ-robust).
        kpflog.setup_logging(str(tmp_path), "science", "t", _RUN_ID, console=False)
        handlers = logging.getLogger().handlers
        (handler,) = [h for h in handlers if h.name == "kpfpipe_file"]
        assert handler.formatter.converter is time.gmtime

    def test_level_filtering(self, tmp_path):
        path = kpflog.setup_logging(
            str(tmp_path), "science", "t", _RUN_ID, level="INFO", console=False
        )
        logging.getLogger("kpfpipe.x").debug("too quiet")
        logging.getLogger("kpfpipe.x").info("loud enough")
        text = _read(path)
        assert "too quiet" not in text
        assert "loud enough" in text

    def test_later_setup_for_the_same_target_appends(self, tmp_path):
        # The masters batch runs each night once per stage under one run id; the
        # night keeps the stamped log its first stage created.
        first = kpflog.setup_logging(
            str(tmp_path), "masters", "20240405", _RUN_ID, console=False
        )
        logging.getLogger("kpfpipe.x").info("bias stage")
        second = kpflog.setup_logging(
            str(tmp_path), "masters", "20240405", _RUN_ID, console=False
        )
        logging.getLogger("kpfpipe.x").info("dark stage")
        assert second == first
        assert re.search(r"kpf_masters_20240405_\d{8}T\d{6}\.log$", first)
        text = _read(first)
        assert "bias stage" in text and "dark stage" in text

    def test_unique_filename_on_collision(self, tmp_path):
        # Two instances creating the same log in the same second never share it.
        path = str(tmp_path / "kpf_science_t_20260702T140322.log")
        first = kpflog._open_file_handler(path)
        second = kpflog._open_file_handler(path)
        first.close()
        second.close()
        assert second.baseFilename == f"{first.baseFilename}.1"

    def test_exhausted_collision_retries_raises(self, tmp_path, monkeypatch):
        # The retry loop must terminate: turning it into `while True` would hang
        # a run forever instead of failing it.
        monkeypatch.setattr(kpflog, "_MAX_COLLISION_RETRIES", 1)
        path = str(tmp_path / "kpf_science_t_20260702T140322.log")
        handlers = [kpflog._open_file_handler(path) for _ in range(2)]
        for handler in handlers:
            handler.close()
        with pytest.raises(FileExistsError, match="unique log file"):
            kpflog._open_file_handler(path)

    def test_repeated_setup_no_duplicate_handlers(self, tmp_path):
        root = logging.getLogger()
        before = len(root.handlers)
        first = kpflog.setup_logging(
            str(tmp_path), "science", "t", _RUN_ID, console=False
        )
        kpflog.setup_logging(
            str(tmp_path), "masters", "20240923", _RUN_ID, console=False
        )
        assert len(root.handlers) == before + 1
        (handler,) = [h for h in root.handlers if h.name == "kpfpipe_file"]
        assert handler.baseFilename != first  # the survivor is the second file

    def test_console_handler_optional(self, tmp_path):
        root = logging.getLogger()
        kpflog.setup_logging(str(tmp_path), "science", "t", _RUN_ID, console=True)
        assert any(h.name == "kpfpipe_console" for h in root.handlers)
        kpflog.setup_logging(str(tmp_path), "science", "t", _RUN_ID, console=False)
        assert not any(h.name == "kpfpipe_console" for h in root.handlers)

    def test_console_defaults_to_stderr(self, tmp_path):
        # The leaf runner's console echo stays on stderr (stream=None default).
        kpflog.setup_logging(str(tmp_path), "science", "t", _RUN_ID, console=True)
        (console,) = [
            h for h in logging.getLogger().handlers if h.name == "kpfpipe_console"
        ]
        assert console.stream is sys.stderr

    def test_third_party_pins(self, tmp_path):
        kpflog.setup_logging(str(tmp_path), "science", "t", _RUN_ID, console=False)
        assert logging.getLogger("matplotlib").level == logging.WARNING
        assert logging.getLogger("PIL").level == logging.WARNING

    def test_bad_level_raises_before_side_effects(self, tmp_path):
        with pytest.raises(ValueError, match="unknown log level"):
            kpflog.setup_logging(str(tmp_path), "science", "t", _RUN_ID, level="chatty")
        assert list(tmp_path.iterdir()) == []

    def test_writes_into_the_run_directory(self, tmp_path, monkeypatch):
        # log_dir is the configured *parent*; the run gets its own directory under
        # it, so a whole process tree sharing one run_id logs side by side.
        monkeypatch.setattr(kpflog.time, "gmtime", lambda *a: _UT_FROZEN)
        path = kpflog.setup_logging(
            str(tmp_path), "science", "t", _RUN_ID, console=False
        )
        assert path == str(tmp_path / _RUN_ID / "kpf_science_t_20260702T140322.log")

    def test_mints_its_own_run_directory_when_unset(self, tmp_path, monkeypatch):
        # A standalone `kpfpipe run`: the leaf passes its --run_id straight through,
        # so an unset one is minted here rather than on the fly in the script.
        monkeypatch.setattr(kpflog.time, "gmtime", lambda *a: _UT_FROZEN)
        path = kpflog.setup_logging(str(tmp_path), "science", "t", console=False)
        assert path == str(
            tmp_path / "run_20260702T140322" / "kpf_science_t_20260702T140322.log"
        )

    def test_missing_log_dir_raises(self, tmp_path):
        # The guard must precede the run-dir join: os.path.join("", run_id) is a
        # truthy *relative* path, so log_filename's own check would never fire and
        # an unset log_dir would silently write beside the cwd (DRP-RUN-07).
        for unset in ("", None):
            with pytest.raises(ValueError, match="log_dir"):
                kpflog.setup_logging(unset, "science", "t", _RUN_ID, console=False)
        assert not list(tmp_path.iterdir())


class TestSetupBatchLogging:
    """A per-invocation ``_batch_`` log echoed to stdout so an operator can watch
    fan-out progress live."""

    def test_mints_the_run_id_and_returns_it_with_the_path(self, tmp_path, monkeypatch):
        # Frozen for the same UT-midnight reason as TestSetupLogging's twin.
        monkeypatch.setattr(kpflog.time, "gmtime", lambda *a: _UT_FROZEN)
        run_id, path = kpflog.setup_batch_logging(str(tmp_path), "masters")
        assert run_id == "masters_20260702T140322"
        assert path == str(tmp_path / run_id / "kpf_masters_batch_20260702T140322.log")
        assert _read(path) == ""  # created, empty until a record arrives

    def test_forwarded_run_id_is_used_verbatim(self, tmp_path):
        # What a launching script passes down: join its run, never mint another.
        parent = "timeseries_20240405T010203"
        run_id, path = kpflog.setup_batch_logging(
            str(tmp_path), "masters", parent, console=False
        )
        assert run_id == parent
        assert path.startswith(str(tmp_path / parent) + "/")

    def test_module_logger_record_lands_in_file(self, tmp_path):
        _, path = kpflog.setup_batch_logging(str(tmp_path), "masters", console=False)
        logging.getLogger("scripts.process.masters").info("dispatching 3 job(s)")
        assert "dispatching 3 job(s)" in _read(path)

    def test_console_echo_is_stdout(self, tmp_path):
        kpflog.setup_batch_logging(str(tmp_path), "science")
        (console,) = [
            h for h in logging.getLogger().handlers if h.name == "kpfpipe_console"
        ]
        assert console.stream is sys.stdout

    def test_filter_on_console_only_not_file(self, tmp_path):
        kpflog.setup_batch_logging(str(tmp_path), "science")
        root = logging.getLogger()
        (console,) = [h for h in root.handlers if h.name == "kpfpipe_console"]
        (file_h,) = [h for h in root.handlers if h.name == "kpfpipe_file"]
        assert any(
            isinstance(flt, kpflog._BatchConsoleFilter) for flt in console.filters
        )
        assert not file_h.filters

    def test_library_info_off_console_but_kept_in_file(self, tmp_path, capsys):
        # Library INFO chatter is trimmed from the live terminal echo while the
        # driver's narration and any WARNING still show; the file keeps everything.
        # thirdparty.io is a neutral name -- astropy sets propagate=False on its
        # logger, so its records never reach the root handlers.
        _, path = kpflog.setup_batch_logging(str(tmp_path), "masters")
        logging.getLogger("scripts.process.masters").info("driver narration")
        logging.getLogger("thirdparty.io").info("library chatter")
        logging.getLogger("thirdparty.io").warning("library warning")

        out = capsys.readouterr().out
        assert "driver narration" in out
        assert "library chatter" not in out
        assert "library warning" in out
        text = _read(path)
        assert {"driver narration", "library chatter", "library warning"} <= {
            line.split(": ", 1)[-1].strip() for line in text.splitlines()
        }


class TestBatchConsoleFilter:
    """Sub-WARNING records pass only from ``scripts.*``/``__main__``; WARNING and
    above always pass."""

    _flt = kpflog._BatchConsoleFilter()

    def _rec(self, name, level):
        return logging.LogRecord(name, level, __file__, 1, "m", None, None)

    def test_info_from_scripts_passes(self):
        assert self._flt.filter(self._rec("scripts.process.masters", logging.INFO))
        assert self._flt.filter(self._rec("scripts.plot.timeseries", logging.INFO))

    def test_info_from_main_passes(self):
        # A driver launched as `python -m scripts.process.<name>` logs as __main__.
        assert self._flt.filter(self._rec("__main__", logging.INFO))

    def test_info_from_library_dropped(self):
        assert not self._flt.filter(self._rec("astropy.io.fits", logging.INFO))
        assert not self._flt.filter(self._rec("kpfpipe.utils.io", logging.INFO))
        assert not self._flt.filter(self._rec("py.warnings", logging.INFO))

    def test_debug_is_source_filtered_too(self):
        # Below WARNING covers DEBUG as well, not just INFO.
        assert not self._flt.filter(self._rec("astropy", logging.DEBUG))
        assert self._flt.filter(self._rec("scripts.process.science", logging.DEBUG))

    def test_warning_and_above_always_pass(self):
        # Per-unit failures and warnings from anywhere are never hidden.
        assert self._flt.filter(self._rec("astropy.io.fits", logging.WARNING))
        assert self._flt.filter(self._rec("py.warnings", logging.WARNING))
        assert self._flt.filter(self._rec("kpfpipe.x", logging.ERROR))


class TestTeardown:
    def test_teardown_removes_handlers_and_restores_showwarning(self, tmp_path):
        root = logging.getLogger()
        before_handlers = list(root.handlers)
        before_showwarning = warnings.showwarning
        kpflog.setup_logging(str(tmp_path), "science", "t", _RUN_ID)
        kpflog.teardown_logging()
        assert root.handlers == before_handlers
        assert warnings.showwarning is before_showwarning

    def test_teardown_restores_root_level(self, tmp_path):
        root = logging.getLogger()
        before = root.level
        kpflog.setup_logging(str(tmp_path), "science", "t", _RUN_ID, level="DEBUG")
        kpflog.teardown_logging()
        assert root.level == before

    def test_teardown_safe_when_nothing_installed(self):
        kpflog.teardown_logging()  # must not raise


class TestWarningsBridge:
    @pytest.mark.filterwarnings("always")
    def test_capturewarnings_lands_in_file_with_source(self, tmp_path):
        path = kpflog.setup_logging(
            str(tmp_path), "science", "t", _RUN_ID, console=False
        )
        warnings.warn("bridged boom", stacklevel=1)  # 1: attribute to THIS line
        text = _read(path)
        assert "WARNING  py.warnings:" in text
        assert "bridged boom" in text
        assert "test_logger.py" in text

    @pytest.mark.filterwarnings("always")
    def test_pytest_warns_coexistence(self, tmp_path):
        # Inside pytest.warns the recorder wins and the record never reaches the
        # log: captureWarnings only swaps showwarning, while pytest.warns uses
        # catch_warnings(record=True).
        path = kpflog.setup_logging(
            str(tmp_path), "science", "t", _RUN_ID, console=False
        )
        with pytest.warns(UserWarning, match="recorded boom"):
            warnings.warn("recorded boom", stacklevel=2)
        assert "recorded boom" not in _read(path)
