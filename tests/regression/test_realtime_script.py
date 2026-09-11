"""Tests for scripts/processing/realtime.py: the continuous-mode L0 watcher.

Covers arg parsing, UT night selection, the settled-frame scan, IMTYPE
classification, the exactly-once ledger, and one-pass (``--once``) runs with the
subprocess launcher stubbed: what gets dispatched, what gets skipped, what the
heartbeat status file and the run.json say, and that a second pass over the same
tree dispatches nothing. Synthetic L0 trees only -- no real reduction runs.
"""

import json
import os

import pytest

from kpfpipe.utils import run_record as rr
from scripts.processing import realtime as rt

from ._scripts import write_l0_tree

# scripts/CLI/tools-layer suite: excluded from `make test-fast`.
pytestmark = pytest.mark.cli

_NIGHT = "20240405"
# 2024-04-05 00:30 UT, so "today" is _NIGHT and yesterday is 20240404.
_NOW = 1712277000.0


# ---------------------------------------------------------------------------
# parse_args / watch_datecodes
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self, monkeypatch):
        monkeypatch.setattr(rt, "_default_science_jobs", lambda: 3)
        ns = rt.parse_args([])
        assert ns.poll_interval == 15.0
        assert ns.settle == 60.0
        assert ns.nights == 2
        assert ns.once is False
        assert ns.jobs == 3
        assert ns.job_timeout == 1200
        assert ns.status_file is None and ns.ledger is None

    @pytest.mark.parametrize(
        "argv",
        [
            ["--poll_interval", "0"],
            ["--settle", "-1"],
            ["--nights", "0"],
            ["--jobs", "0"],
            ["--job_timeout", "0"],
        ],
    )
    def test_invalid_values_exit_two(self, argv):
        with pytest.raises(SystemExit) as exc:
            rt.parse_args(argv)
        assert exc.value.code == 2

    def test_output_dir_fans_out_to_log_dir(self):
        ns = rt.parse_args(["--output_dir", "/out"])
        assert ns.log_dir == "/out/logs"


class TestWatchDatecodes:
    def test_today_then_yesterday_ut(self):
        assert rt.watch_datecodes(_NOW, nights=2) == ["20240405", "20240404"]

    def test_single_night(self):
        assert rt.watch_datecodes(_NOW, nights=1) == ["20240405"]


# ---------------------------------------------------------------------------
# scan_settled_frames / classify_frame
# ---------------------------------------------------------------------------


def _age(path, seconds, now=_NOW):
    os.utime(path, (now - seconds, now - seconds))


class TestScanSettledFrames:
    def test_only_settled_kp_fits_files(self, tmp_path):
        oid_old = write_l0_tree(tmp_path, _NIGHT, 100)
        oid_new = write_l0_tree(tmp_path, _NIGHT, 200)
        night = tmp_path / "L0" / _NIGHT
        _age(night / f"{oid_old}.fits", 120)
        _age(night / f"{oid_new}.fits", 10)
        (night / "notes.txt").write_text("x")
        (night / "KP.junk.fits").mkdir()  # a directory, not a file
        got = rt.scan_settled_frames([str(night)], settle=60, now=_NOW)
        assert [os.path.basename(p) for p, _m, _s in got] == [f"{oid_old}.fits"]
        assert got[0][2] > 0  # size recorded

    def test_missing_dir_is_empty(self, tmp_path):
        assert rt.scan_settled_frames([str(tmp_path / "nope")], 60, _NOW) == []

    def test_zero_settle_takes_everything(self, tmp_path):
        oid = write_l0_tree(tmp_path, _NIGHT, 100)
        night = tmp_path / "L0" / _NIGHT
        _age(night / f"{oid}.fits", 0)
        assert len(rt.scan_settled_frames([str(night)], 0, _NOW)) == 1


class TestClassifyFrame:
    def test_object_is_science(self, tmp_path):
        oid = write_l0_tree(tmp_path, _NIGHT, 100, imtype="Object")
        assert rt.classify_frame(str(tmp_path / "L0" / _NIGHT / f"{oid}.fits")) == (
            "science",
            None,
        )

    def test_bias_is_cal_with_reason(self, tmp_path):
        oid = write_l0_tree(tmp_path, _NIGHT, 100, imtype="Bias", obj="autocal-bias")
        kind, reason = rt.classify_frame(str(tmp_path / "L0" / _NIGHT / f"{oid}.fits"))
        assert kind == "cal" and reason == "IMTYPE=Bias"

    def test_unreadable_is_cal_with_reason(self, tmp_path):
        bad = tmp_path / "KP.20240405.00001.00.fits"
        bad.write_bytes(b"not a fits file")
        kind, reason = rt.classify_frame(str(bad))
        assert kind == "cal" and reason.startswith("unreadable header")


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class TestLedger:
    def test_roundtrip_and_counts(self, tmp_path):
        led = rt.Ledger(str(tmp_path / "ledger.json")).load()
        k1 = rt.Ledger.key("/a", 1.5, 10)
        k2 = rt.Ledger.key("/b", 2.5, 10)
        led.mark(k1, state="running")
        led.mark(k2, state="skipped")
        led.mark(k1, state="succeeded", exit_status=0)
        again = rt.Ledger(str(tmp_path / "ledger.json")).load()
        assert again.seen(k1) and again.seen(k2)
        assert again.entries[k1]["exit_status"] == 0
        assert again.counts() == {
            "skipped": 1,
            "running": 0,
            "succeeded": 1,
            "failed": 0,
            "dispatched": 1,
        }

    def test_changed_mtime_is_a_new_key(self):
        assert rt.Ledger.key("/a", 1.0, 10) != rt.Ledger.key("/a", 2.0, 10)

    def test_wrong_schema_raises(self, tmp_path):
        p = tmp_path / "ledger.json"
        p.write_text('{"schema": 99, "entries": {}}')
        with pytest.raises(ValueError):
            rt.Ledger(str(p)).load()


# ---------------------------------------------------------------------------
# --once end to end (launcher stubbed)
# ---------------------------------------------------------------------------


class _FakeRunOne:
    """Stand-in for _dispatch._run_one: records argv, returns a scripted rc."""

    def __init__(self, rc_for=None):
        self.calls = []
        self.rc_for = rc_for or {}

    def __call__(self, argv, timeout=None, launch_interval=0.0):
        self.calls.append(argv)
        obs_id = argv[argv.index("-o") + 1]
        return self.rc_for.get(obs_id, 0), ""


def _setup_tree(tmp_path, now=_NOW):
    """Two aged Object frames + one aged Bias frame under L0/{tonight}."""
    data = tmp_path / "data"
    o1 = write_l0_tree(data, _NIGHT, 100, imtype="Object")
    o2 = write_l0_tree(data, _NIGHT, 200, imtype="Object")
    b1 = write_l0_tree(data, _NIGHT, 300, imtype="Bias", obj="autocal-bias")
    night = data / "L0" / _NIGHT
    for oid in (o1, o2, b1):
        _age(night / f"{oid}.fits", 600, now)
    return data, o1, o2, b1


def _once(monkeypatch, tmp_path, data, fake, extra=(), now=_NOW):
    """Run realtime --once with the launcher stubbed and a temp log dir."""
    log_dir = tmp_path / "logs"
    fake_log = log_dir / _NIGHT / "kpf_realtime_batch_x.log"
    fake_log.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rt, "configure_runtime", lambda: None)
    monkeypatch.setattr(rt, "setup_batch_logging", lambda *a, **k: str(fake_log))
    monkeypatch.setattr(rt, "_run_one", fake)
    monkeypatch.setattr(rt.time, "time", lambda: now)
    monkeypatch.setattr(rr, "git_sha", lambda repo_root=None: None)
    monkeypatch.delenv(rr.PARENT_ENV, raising=False)
    argv = [
        "--once",
        "--input_dir",
        str(data),
        "--kpf_masters_output",
        str(tmp_path / "m"),
        "--kpf_science_output",
        str(tmp_path / "s"),
        "--log_dir",
        str(log_dir),
        "--jobs",
        "2",
        "--settle",
        "60",
        *extra,
    ]
    code = 0
    try:
        rt.main(argv)
    except SystemExit as exc:
        code = exc.code
    return code, log_dir, str(fake_log)


class TestOnce:
    def test_dispatches_science_skips_cal_writes_status_and_record(
        self, monkeypatch, tmp_path
    ):
        data, o1, o2, b1 = _setup_tree(tmp_path)
        fake = _FakeRunOne()
        code, log_dir, fake_log = _once(monkeypatch, tmp_path, data, fake)
        assert code == 0

        dispatched = sorted(a[a.index("-o") + 1] for a in fake.calls)
        assert dispatched == sorted([o1, o2])
        # The leaf is told the same dir overrides the watcher got.
        assert "--kpf_data_input" in fake.calls[0]

        ledger = rt.Ledger(str(log_dir / "realtime_ledger.json")).load()
        states = {e["obs_id"]: e["state"] for e in ledger.entries.values()}
        assert states == {o1: "succeeded", o2: "succeeded", b1: "skipped"}

        with open(log_dir / "realtime_status.json") as fh:
            status = json.load(fh)
        assert status["schema"] == rt.STATUS_SCHEMA
        assert status["files_seen"] == 3
        assert status["dispatched"] == 2
        assert status["skipped"] == 1
        assert status["succeeded"] == 2
        assert status["failed"] == 0
        assert status["running"] == 0 and status["queued"] == 0
        assert status["watched_dirs"][0].endswith(os.path.join("L0", _NIGHT))
        assert status["last_dispatch_utc"] is not None
        assert status["run_json"] == rr.run_json_path(fake_log)

        rec = rr.read_run_record(rr.run_json_path(fake_log))
        assert rec["kind"] == "realtime"
        assert rec["status"] == "succeeded"
        assert rec["counts"] == {"done": 2, "failed": 0, "skipped": 1}
        assert os.environ.get(rr.PARENT_ENV) is None

    def test_second_pass_dispatches_nothing(self, monkeypatch, tmp_path):
        data, *_ = _setup_tree(tmp_path)
        first = _FakeRunOne()
        _once(monkeypatch, tmp_path, data, first)
        second = _FakeRunOne()
        code, log_dir, _ = _once(monkeypatch, tmp_path, data, second)
        assert code == 0
        assert len(first.calls) == 2
        assert second.calls == []

    def test_failed_frame_gives_exit_one_and_failed_ledger_state(
        self, monkeypatch, tmp_path
    ):
        data, o1, o2, _b1 = _setup_tree(tmp_path)
        fake = _FakeRunOne(rc_for={o2: 1})
        code, log_dir, fake_log = _once(monkeypatch, tmp_path, data, fake)
        assert code == 1
        ledger = rt.Ledger(str(log_dir / "realtime_ledger.json")).load()
        states = {e["obs_id"]: e["state"] for e in ledger.entries.values()}
        assert states[o1] == "succeeded" and states[o2] == "failed"
        rec = rr.read_run_record(rr.run_json_path(fake_log))
        assert rec["status"] == "failed"
        assert rec["counts"] == {"done": 1, "failed": 1, "skipped": 1}

    def test_unsettled_frame_waits_for_a_later_tick(self, monkeypatch, tmp_path):
        data, o1, o2, _b1 = _setup_tree(tmp_path)
        night = data / "L0" / _NIGHT
        _age(night / f"{o2}.fits", 10)  # landed 10 s ago: not settled yet
        fake = _FakeRunOne()
        _once(monkeypatch, tmp_path, data, fake)
        assert [a[a.index("-o") + 1] for a in fake.calls] == [o1]
        # 5 minutes later, the same tree: only the newly settled frame goes out.
        fake2 = _FakeRunOne()
        _once(monkeypatch, tmp_path, data, fake2, now=_NOW + 300)
        assert [a[a.index("-o") + 1] for a in fake2.calls] == [o2]
